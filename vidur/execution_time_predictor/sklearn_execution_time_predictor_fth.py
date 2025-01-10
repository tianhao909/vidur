import hashlib
import os
import pickle
from abc import abstractmethod
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from fasteners import InterProcessReaderWriterLock
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from vidur.config import (
    BaseExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    ReplicaConfig,
)
from vidur.entities import Batch
from vidur.execution_time_predictor.base_execution_time_predictor import (
    BaseExecutionTimePredictor,
)
from vidur.logger import init_logger

logger = init_logger(__name__)


class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):  # 定义一个名为 SklearnExecutionTimePredictor 的类，继承自 BaseExecutionTimePredictor 类
    def __init__(
        self,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
    ) -> None:  # 初始化方法，定义类的构造函数，接受多个配置对象作为参数
        super().__init__(
            predictor_config=predictor_config,
            replica_config=replica_config,
            replica_scheduler_config=replica_scheduler_config,
            metrics_config=metrics_config,
        )  # 调用父类的初始化方法以便继承属性
        os.makedirs(self._cache_dir, exist_ok=True)  # 创建缓存目录，如果目录不存在则创建

        # These overheads are only for GQA models
        self._attention_prefill_batching_overhead_fraction = (  # 设置注意力预填充批量开销分数
            (self._config.attention_prefill_batching_overhead_fraction)
            if self._model_config.num_q_heads > self._model_config.num_kv_heads
            else 0
        )
        self._attention_decode_batching_overhead_fraction = (  # 设置注意力解码批量开销分数
            (self._config.attention_decode_batching_overhead_fraction)
            if self._model_config.num_q_heads > self._model_config.num_kv_heads
            else 0
        )
        if self._replica_scheduler_provider == "orca":  # 检查复制调度器提供程序是否为 "orca"
            self._max_tokens = (  # 设置最大token数
                self._config.prediction_max_tokens_per_request
                * self._config.prediction_max_batch_size
            )
        else:
            self._max_tokens = self._config.prediction_max_tokens_per_request  # 设置最大token数

        num_workers = (  # 计算工人数
            self._replica_config.num_pipeline_stages
            * self._replica_config.tensor_parallel_size
        )
        devices_per_node = self._replica_config.node_config.num_devices_per_node  # 获取每个节点的设备数
        assert (
            num_workers < devices_per_node or num_workers % devices_per_node == 0
        ), "Number of workers should be less than devices per node or a multiple of devices per node"  # 断言工人数应小于设备数或设备数的倍数

        self._is_multi_node = num_workers > devices_per_node  # 判断是否为多节点

        (
            self._compute_input_file,
            self._attention_input_file,
            self._all_reduce_input_file,
            self._send_recv_input_file,
            self._cpu_overhead_input_file,
        ) = self._get_input_files()  # 获取输入文件路径

        self._models = self._train_models()  # 训练模型
        self._predictions = self._predict_from_models()  # 从模型中进行预测

    def _get_input_files(self) -> Tuple[str, str, str, str, str]:  # 获取输入文件的方法
        input_files = [
            self._config.compute_input_file,
            self._config.attention_input_file,
            self._config.all_reduce_input_file,
            self._config.send_recv_input_file,
            self._config.cpu_overhead_input_file,
        ]  # 定义输入文件列表
        for i in range(len(input_files)):  # 遍历输入文件列表
            input_files[i] = (
                input_files[i]
                .replace("{DEVICE}", self._replica_config.device)
                .replace("{MODEL}", self._model_config.get_name())
                .replace("{NETWORK_DEVICE}", self._replica_config.network_device)
            )  # 替换文件路径中的占位符

        return tuple(input_files)  # 返回输入文件路径的元组

    def _load_compute_df(self, file_path: str) -> pd.DataFrame:  # 加载计算数据的方法，接受文件路径作为参数
        df = self._read_input_file(file_path)  # 读取输入文件并返回数据帧
        df = df.drop_duplicates()  # 删除重复项

        logger.debug(f"Length of complete compute df: {len(df)} {file_path}")  # 记录日志，打印数据帧的长度和文件路径
        logger.debug(f"self._num_q_heads: {self._model_config.num_q_heads}")  # 打印模型的 num_q_heads 配置
        logger.debug(f"self._embedding_dim: {self._model_config.embedding_dim}")  # 打印模型的 embedding_dim 配置
        logger.debug(f"self._mlp_hidden_dim: {self._model_config.mlp_hidden_dim}")  # 打印模型的 mlp_hidden_dim 配置
        logger.debug(f"self._use_gated_mlp: {self._model_config.use_gated_mlp}")  # 打印模型的 use_gated_mlp 配置
        logger.debug(f"self._vocab_size: {self._model_config.vocab_size}")  # 打印模型的 vocab_size 配置
        logger.debug(
            f"self._num_tensor_parallel_workers: {self._replica_config.tensor_parallel_size}"
        )  # 打印模型的 num_tensor_parallel_workers 配置

        df = df[
            (df["n_head"] == self._model_config.num_q_heads)
            & (df["n_kv_head"] == self._model_config.num_kv_heads)
            & (df["n_embd"] == self._model_config.embedding_dim)
            & (df["n_expanded_embd"] == self._model_config.mlp_hidden_dim)
            & (df["use_gated_mlp"] == self._model_config.use_gated_mlp)
            & (df["vocab_size"] == self._model_config.vocab_size)
            & (
                df["num_tensor_parallel_workers"]
                == self._replica_config.tensor_parallel_size
            )
        ]  # 筛选符合模型配置的数据

        for column in [
            "time_stats.post_attention_layernorm.median",
            "time_stats.add.median",
            "time_stats.input_layernorm.median",
        ]:  # 遍历需要检查的列名
            if column not in df.columns:  # 如果列名不在数据中
                df[column] = 0  # 则填充为0
            else:
                df.fillna({column: 0}, inplace=True)  # 否则将空值填充为0
        return df  # 返回数据帧

    def _load_attention_df(self, file_path: str) -> pd.DataFrame:  # 加载注意力数据的方法
        df = pd.read_csv(file_path)  # 读取CSV文件
        df = df.drop_duplicates()  # 删除重复项

        for column in [
            "time_stats.attn_kv_cache_save.median",
        ]:  # 遍历需要检查的列名
            if column not in df.columns:  # 如果列名不在数据中
                df[column] = 0  # 则填充为0
            else:
                df.fillna({column: 0}, inplace=True)  # 否则将空值填充为0

        return df[
            (df["n_embd"] == self._model_config.embedding_dim)
            & (df["n_q_head"] == self._model_config.num_q_heads)
            & (df["n_kv_head"] == self._model_config.num_kv_heads)
            & (df["block_size"] == self._block_size)
            & (
                df["num_tensor_parallel_workers"]
                == self._replica_config.tensor_parallel_size
            )
        ]  # 筛选符合模型配置的数据，并返回

    def _load_all_reduce_df(self, file_path: str) -> pd.DataFrame:  # 加载All Reduce数据的方法
        df = self._read_input_file(file_path)  # 读取输入文件并返回数据帧
        return df[
            (df["num_workers"] == self._replica_config.tensor_parallel_size)
            & (df["devices_per_node"] == self._replica_config.tensor_parallel_size)
            & (df["collective"] == "all_reduce")
        ]  # 筛选符合条件的数据，并返回

    def _load_send_recv_df(self, file_path: str) -> pd.DataFrame:  # 加载Send/Recv数据的方法
        if self._is_multi_node:  # 判断是否为多节点
            devices_per_node = 1  # 如果是，则设备数为1
        else:
            devices_per_node = 2  # 否则设备数为2

        df = self._read_input_file(file_path)  # 读取输入文件并返回数据帧
        filtered_df = df[
            (df["collective"] == "send_recv")
            & (df["devices_per_node"] == devices_per_node)
        ]  # 筛选符合条件的数据
        return filtered_df  # 返回筛选后的数据

    def _load_cpu_overhead_df(self, file_path: str) -> pd.DataFrame:  # 加载CPU开销数据的方法
        df = self._read_input_file(file_path)  # 读取输入文件并返回数据帧
        filtered_df = df[
            (df["model_name"] == self._model_config.get_name())
            & (
                df["tensor_parallel_degree"]
                == self._replica_config.tensor_parallel_size
            )
        ]  # 筛选符合条件的数据
        return filtered_df  # 返回筛选后的数据

    def _read_input_file(self, file_path: str) -> pd.DataFrame:  # 用于读取输入文件的方法
        df = pd.read_csv(file_path)  # 读取CSV文件
        df = df.drop_duplicates()  # 删除重复项
        return df  # 返回数据帧

    def _get_compute_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:  # 获取带有衍生特征的计算数据的方法
        df_with_derived_features = df.copy()  # 复制数据帧
        return df_with_derived_features  # 返回带有衍生特征的数据帧

    def _get_attention_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:  # 获取带有衍生特征的注意力数据的方法
        df_with_derived_features = df.copy()  # 复制数据帧
        df_with_derived_features["num_tokens"] = df_with_derived_features[
            ["prefill_chunk_size", "batch_size"]
        ].max(axis=1)  # 计算num_tokens列为prefill_chunk_size和batch_size的最大值
        df_with_derived_features["is_decode"] = (
            df_with_derived_features["prefill_chunk_size"] == 0
        )  # 计算is_decode列为prefill_chunk_size是否为0的布尔值
        df_with_derived_features["prefill_chunk_size_squared"] = (
            df_with_derived_features["prefill_chunk_size"] ** 2
        )  # 计算prefill_chunk_size_squared列为prefill_chunk_size的平方
        return df_with_derived_features  # 返回带有衍生特征的数据帧

    def _get_all_reduce_df_with_derived_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:  # 获取带有衍生特征的All Reduce数据的方法
        df_with_derived_features = df.copy()  # 复制数据帧
        # convert bytes to num tokens
        # each token is of size 2 * h bytes
        df_with_derived_features["num_tokens"] = (  # 计算num_tokens列，将字节转换为token数
            df_with_derived_features["size"] / self._model_config.embedding_dim / 2
        )
        return df_with_derived_features  # 返回带有衍生特征的数据帧

    def _get_send_recv_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:  # 获取带有衍生特征的Send/Recv数据的方法
        df_with_derived_features = df.copy()  # 复制数据帧
        df_with_derived_features["num_tokens"] = (  # 计算num_tokens列，将字节转换为token数
            df_with_derived_features["size"] / self._model_config.embedding_dim / 2
        )
        return df_with_derived_features  # 返回带有衍生特征的数据帧

    def _get_cpu_overhead_df_with_derived_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:  # 获取带有衍生特征的CPU开销数据的方法
        df_with_derived_features = df.copy()  # 复制数据帧
        return df_with_derived_features  # 返回带有衍生特征的数据帧

    @staticmethod
    def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:  # 静态方法，用于计算平均绝对百分比误差
        y_true, y_pred = np.array(y_true), np.array(y_pred)  # 将输入转为numpy数组
        # Handling the case where y_true is 0 separately to avoid division by zero
        zero_true_mask = y_true == 0  # 找出y_true中为0的元素
        non_zero_true_mask = ~zero_true_mask  # 找出y_true中非0的元素

        # For non-zero true values, calculate the absolute percentage error
        error = np.zeros_like(y_true, dtype=float)  # using float instead of np.float
        error[non_zero_true_mask] = (  # 计算非0元素的绝对百分比误差
            np.abs(
                (y_true[non_zero_true_mask] - y_pred[non_zero_true_mask])
                / y_true[non_zero_true_mask]
            )
            * 100
        )

        # For zero true values, if prediction is also 0, error is 0, else it is 100
        error[zero_true_mask] = np.where(y_pred[zero_true_mask] == 0, 0, 100)  # 计算0元素的误差

        # Return the mean of the absolute percentage errors
        return np.mean(error)  # 返回平均绝对百分比误差

    def _get_scorer(self) -> Any:  # 获取评分器的方法
        return make_scorer(
            SklearnExecutionTimePredictor.mean_absolute_percentage_error,
            greater_is_better=False,
        )  # 使用自定义评分器，指标为平均绝对百分比误差，需要最小化

    @abstractmethod
    def _get_grid_search_params(self) -> Dict[str, Any]:  # 抽象方法，获取网格搜索参数，需在子类中实现
        pass

    @abstractmethod
    def _get_estimator(self) -> BaseEstimator:  # 抽象方法，获取估计器，需在子类中实现
        pass

    def _get_model_hash(self, model_name: str, df: pd.DataFrame = None) -> str:  # 获取模型哈希的方法
        config_str = str(self.to_dict())  # 将模型配置转为字符串

        if df is None:
            combined_str = f"{config_str}_{model_name}"  # 如果没有数据帧，组合字符串只包含配置和模型名
        else:
            df_hash_str = hashlib.md5(df.to_json().encode("utf-8")).hexdigest()  # 如果有数据帧，计算数据帧的哈希
            combined_str = f"{config_str}_{model_name}_{df_hash_str}"  # 组合字符串包含配置、模型名和数据帧哈希

        return hashlib.md5(combined_str.encode("utf-8")).hexdigest()[0:8]  # 返回组合字符串的哈希值前8位

    def _load_model_from_cache(self, model_name: str, model_hash: str) -> BaseEstimator:  # 从缓存加载模型的方法
        with InterProcessReaderWriterLock(
            f"{self._cache_dir}/{model_hash}_model_lock.file"
        ).read_lock():  # 在多进程环境下加读锁
            if self._config.no_cache:
                return  # 如果配置为不使用缓存，则直接返回
            # check if model is in cache
            cache_file = f"{self._cache_dir}/{model_name}_{model_hash}.pkl"  # 定义缓存文件路径
            if not os.path.exists(cache_file):
                return  # 如果缓存文件不存在，则返回

            logger.debug(f"Found model {model_name} in cache")  # 记录日志，打印从缓存中找到的模型信息
            model = pickle.load(open(cache_file, "rb"))  # 从缓存中加载模型
            return model  # 返回模型

    def _store_model_in_cache(
        self, model_name: str, model_hash: str, model: BaseEstimator
    ) -> None:  # 将模型存储到缓存的方法
        with InterProcessReaderWriterLock(
            f"{self._cache_dir}/{model_hash}_model_lock.file"
        ).write_lock():  # 在多进程环境下加写锁
            # store model in cache
            cache_file = f"{self._cache_dir}/{model_name}_{model_hash}.pkl"  # 定义缓存文件路径
            pickle.dump(model, open(cache_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)  # 将模型存储到缓存中，使用最高协议


    def _store_training_prediction_data(
        self,
        model_name: str,
        model_hash: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        model: BaseEstimator,
    ) -> None:
        df = df.copy()  # 复制数据帧

        # 将数据帧转换为元组列表
        df["prediction"] = model.predict(df[feature_cols])  # 用模型预测并添加预测列

        # 存储预测数据
        df[feature_cols + [target_col, "prediction"]].to_csv(
            f"{self._cache_dir}/{model_name}_{model_hash}_training_predictions.csv",
            index=False,
        )  # 将数据存储为csv文件，不包含索引

    def _train_model(
        self,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> BaseEstimator:
        if len(df) == 0:
            raise Exception(f"Training data for model {model_name} is empty")  # 如果训练数据为空，抛出异常

        model_hash = self._get_model_hash(model_name, df)  # 获取模型哈希值

        cached_model = self._load_model_from_cache(model_name, model_hash)  # 从缓存加载模型
        if cached_model:
            return cached_model  # 如果缓存中已有模型，直接返回

        model = self._get_estimator()  # 获取估计器
        grid_search_params = self._get_grid_search_params()  # 获取网格搜索参数

        if len(df) < self._config.k_fold_cv_splits:
            cv = 2  # 如果数据量小于k折交叉验证的分裂数量，设置为2
        else:
            cv = self._config.k_fold_cv_splits  # 否则，使用默认的k折数量

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid_search_params,
            scoring=self._get_scorer(),
            cv=cv,
            n_jobs=self._config.num_training_job_threads,
        )  # 创建网格搜索对象

        # 我们不创建训练/测试分割，因为我们想用所有数据进行训练
        # 我们并不关心过拟合，因为我们只是想在相同领域内预测执行时间
        X, y = df[feature_cols], df[target_col]  # 特征和目标列分开

        grid_search.fit(X, y)  # 进行网格搜索训练
        score = grid_search.score(X, y)  # 计算得分

        logger.info(
            f"Trained model {model_name} and found best parameters: {grid_search.best_params_} "
            f"with mean absolute percentage error (MEAP) {-score}%"
        )  # 打印日志信息，显示最佳参数和误差

        self._store_model_in_cache(model_name, model_hash, grid_search.best_estimator_)  # 将最佳估计器存储到缓存中

        self._store_training_prediction_data(
            model_name=model_name,
            model_hash=model_hash,
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            model=grid_search.best_estimator_,
        )  # 存储训练预测数据
        return grid_search.best_estimator_  # 返回最佳估计器

    def _store_model_predication_cache(
        self, model_name: str, model_hash: str, predictions: Dict[Tuple, float]
    ) -> None:
        with InterProcessReaderWriterLock(
            f"{self._cache_dir}/{model_hash}_prediction_lock.file"
        ).write_lock():
            cache_file = f"{self._cache_dir}/{model_name}_{model_hash}_predictions.pkl"
            pickle.dump(
                predictions, open(cache_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL
            )  # 将预测结果以最高协议序列化并存储到缓存文件

    def _load_model_predication_cache(
        self, model_name: str, model_hash: str
    ) -> Dict[Tuple, float]:
        with InterProcessReaderWriterLock(
            f"{self._cache_dir}/{model_hash}_prediction_lock.file"
        ).read_lock():
            if self._config.no_cache:
                return  # 如果配置中不允许缓存，直接返回
            cache_file = f"{self._cache_dir}/{model_name}_{model_hash}_predictions.pkl"

            if not os.path.exists(cache_file):
                return  # 如果缓存文件不存在，直接返回

            logger.debug(f"Found model {model_name} predictions in cache")

            predictions = pickle.load(open(cache_file, "rb"))  # 从缓存文件加载预测结果
            return predictions

    def _get_model_prediction(
        self, model_name: str, model: BaseEstimator, X: pd.DataFrame
    ) -> Dict[Tuple, float]:
        X = X.copy()  # 复制数据帧

        model_hash = self._get_model_hash(model, df=None)  # 获取模型哈希值

        cached_predictions = self._load_model_predication_cache(model_name, model_hash)  # 从缓存加载预测结果
        if cached_predictions:
            return cached_predictions  # 如果缓存中已有预测结果，直接返回

        logger.info(f"Predicting execution time for model {model_name}")

        predictions_array = model.predict(X)  # 使用模型进行预测

        # 将结果转换为字典，以便于缓存
        # 键是每行X的元组
        predictions = dict(zip([tuple(x) for x in X.values], predictions_array))

        self._store_model_predication_cache(model_name, model_hash, predictions)  # 将预测结果存储到缓存中

        X["prediction"] = predictions_array  # 添加预测列
        X.to_csv(
            f"{self._cache_dir}/{model_name}_{model_hash}_predictions.csv",
            index=False,
        )  # 将数据存储为csv文件，不包含索引

        return predictions

    def _train_compute_models(self) -> Dict[str, BaseEstimator]:
        compute_df = self._load_compute_df(self._compute_input_file)  # 加载计算数据帧
        compute_df = self._get_compute_df_with_derived_features(compute_df)  # 获取带有派生特征的计算数据帧

        models = {}
        model_names = [
            "attn_pre_proj",
            "attn_post_proj",
            "mlp_up_proj",
            "mlp_down_proj",
            "mlp_act",
            "input_layernorm",
            "post_attention_layernorm",
            "attn_rope",
            "add",
        ]

        for model_name in model_names:
            logger.debug(
                f"Training model {model_name}, size of training data: {len(compute_df)}"
            )  # 打印日志信息，显示模型名称和训练数据大小
            models[model_name] = self._train_model(
                model_name=model_name,
                df=compute_df,
                feature_cols=["num_tokens"],
                target_col=f"time_stats.{model_name}.median",
            )  # 训练模型并添加到字典中

        attention_df = self._load_attention_df(self._attention_input_file)  # 加载注意力数据帧
        attention_df = self._get_attention_df_with_derived_features(attention_df)  # 获取带有派生特征的注意力数据帧

        model_names = [
            "attn_kv_cache_save",
        ]

        for model_name in model_names:
            models[model_name] = self._train_model(
                model_name=model_name,
                df=attention_df,
                feature_cols=["num_tokens"],
                target_col=f"time_stats.{model_name}.median",
            )  # 训练模型并添加到字典中

        if self._replica_config.num_pipeline_stages > 1:
            send_recv_df = self._load_send_recv_df(self._send_recv_input_file)  # 加载发送接收数据帧
            send_recv_df = self._get_send_recv_df_with_derived_features(send_recv_df)  # 获取带有派生特征的发送接收数据帧

            models["send_recv"] = self._train_model(
                model_name="send_recv",
                df=send_recv_df,
                feature_cols=["num_tokens"],
                target_col="time_stats.send_recv.median",
            )  # 训练发送接收模型并添加到字典中

        if self._replica_config.tensor_parallel_size > 1:
            all_reduce_df = self._load_all_reduce_df(self._all_reduce_input_file)  # 加载全归约数据帧
            all_reduce_df = self._get_all_reduce_df_with_derived_features(all_reduce_df)  # 获取带有派生特征的全归约数据帧

            models["all_reduce"] = self._train_model(
                model_name="all_reduce",
                df=all_reduce_df,
                feature_cols=["num_tokens"],
                target_col="time_stats.all_reduce.median",
            )  # 训练全归约模型并添加到字典中

        return models

    def _train_cpu_overhead_models(self) -> Dict[str, BaseEstimator]:
        if self._config.skip_cpu_overhead_modeling:
            return {}  # 如果配置中跳过CPU开销建模，直接返回空字典

        models = {}
        model_names = [
            "schedule",
            "sampler_e2e",
            "prepare_inputs_e2e",
            "process_model_outputs",
            "ray_comm_time",
        ]

        cpu_overhead_df = self._load_cpu_overhead_df(self._cpu_overhead_input_file)  # 加载CPU开销数据帧
        cpu_overhead_df = self._get_cpu_overhead_df_with_derived_features(
            cpu_overhead_df
        )  # 获取带有派生特征的CPU开销数据帧

        for model_name in model_names:
            if model_name == "ray_comm_time":
                target_col = "ray_comm_time_mean"  # 设置目标列名
            else:
                target_col = f"{model_name}_median"  # 设置目标列名

            models[model_name] = self._train_model(
                model_name=model_name,
                df=cpu_overhead_df,
                feature_cols=["batch_size"],
                target_col=target_col,
            )  # 训练模型并添加到字典中

        return models

    def _train_attention_layer_models(self) -> Dict[str, BaseEstimator]:
        attention_df = self._load_attention_df(self._attention_input_file)  # 加载注意力数据帧
        attention_df = self._get_attention_df_with_derived_features(attention_df)  # 获取带有派生特征的注意力数据帧
        prefill_df = attention_df[~attention_df["is_decode"]]  # 筛选出不是解码的数据
        decode_df = attention_df[attention_df["is_decode"]]  # 筛选出解码的数据

        models = {}

        chunked_prefill_df = prefill_df[prefill_df["kv_cache_size"] > 0].copy()  # 筛选出kv缓存大小大于0的数据并复制
        chunked_prefill_df["total_prefill_tokens"] = (
            chunked_prefill_df["kv_cache_size"]
            + chunked_prefill_df["prefill_chunk_size"]
        )  # 添加总预填充标记列

        models["attn_prefill"] = self._train_model(
            model_name="attn_prefill",
            df=prefill_df,
            feature_cols=["kv_cache_size", "prefill_chunk_size_squared"],
            target_col="time_stats.attn_prefill.median",
        )  # 训练注意力预填充模型并添加到字典中

        models["attn_decode"] = self._train_model(
            model_name="attn_decode",
            df=decode_df,
            feature_cols=["batch_size", "kv_cache_size"],
            target_col="time_stats.attn_decode.median",
        )  # 训练注意力解码模型并添加到字典中

        return models

    def _train_models(self) -> Dict[str, BaseEstimator]:
        models = self._train_compute_models()  # 训练计算模型
        models.update(self._train_cpu_overhead_models())  # 更新CPU开销模型
        models.update(self._train_attention_layer_models())  # 更新注意力层模型

        return models

    def _predict_for_compute_models(self) -> Dict[str, Any]:
        predictions = {}

        model_names = [
            "attn_pre_proj",
            "attn_post_proj",
            "mlp_up_proj",
            "mlp_down_proj",
            "mlp_act",
            "attn_rope",
            "attn_kv_cache_save",
            "input_layernorm",
            "post_attention_layernorm",
            "add",
        ]

        if self._replica_config.num_pipeline_stages > 1:
            model_names.append("send_recv")  # 如果有多个流水线阶段，添加send_recv模型

        if self._replica_config.tensor_parallel_size > 1:
            model_names.append("all_reduce")  # 如果有张量并行，添加all_reduce模型

        num_token_range = np.arange(1, self._max_tokens + 1)  # 创建标记数量范围
        X = pd.DataFrame({"num_tokens": num_token_range})  # 创建数据帧

        for model_name in model_names:
            model = self._models[model_name]  # 获取模型
            predictions[model_name] = self._get_model_prediction(model_name, model, X)  # 获取模型预测结果添加到字典中

        return predictions  # 返回预测结果字典

    def _predict_for_cpu_overhead_models(self) -> Dict[str, Any]:
        if self._config.skip_cpu_overhead_modeling:  # 如果配置中设置了跳过 CPU 开销建模
            return {}  # 那么返回一个空的字典

        predictions = {}  # 创建一个空字典用于存储预测结果

        model_names = [  # 定义需要进行预测的模型名称
            "schedule",
            "sampler_e2e",
            "prepare_inputs_e2e",
            "process_model_outputs",
            "ray_comm_time",
        ]

        batch_size_range = np.arange(1, self._config.prediction_max_batch_size + 1)  # 创建批次大小的范围
        X = pd.DataFrame({"batch_size": batch_size_range})  # 将批次大小范围转换为 DataFrame 格式

        for model_name in model_names:  # 遍历每一个模型名称
            model = self._models[model_name]  # 从模型列表中获取相应的模型
            predictions[model_name] = self._get_model_prediction(model_name, model, X)  # 获取模型预测并存储

        return predictions  # 返回所有模型的预测结果

    def _predict_for_attention_layer_models(self) -> Dict[str, Any]:
        predictions = {}  # 创建一个空字典用于存储预测结果

        decode_batch_size_range = np.arange(1, self._config.prediction_max_batch_size + 1)  # 创建解码批大小的范围
        decode_kv_cache_size_range = np.arange(  # 创建解码键值缓存大小的范围
            0,
            self._config.prediction_max_tokens_per_request + 1,
            self._config.kv_cache_prediction_granularity,
        )
        decode_prefill_chunk_size_range = [0]  # 解码预填充块大小设置为 0
        decode_batch_size, decode_kv_cache_size, decode_prefill_chunk_size = zip(  # 使用笛卡儿积将三者结合
            *product(
                decode_batch_size_range,
                decode_kv_cache_size_range,
                decode_prefill_chunk_size_range,
            )
        )

        prefill_batch_size_range = [1]  # 预填充批次大小范围仅为 1
        prefill_kv_cache_size_range = np.arange(  # 创建预填充键值缓存大小的范围
            0,
            self._config.prediction_max_tokens_per_request + 1,
            self._config.kv_cache_prediction_granularity,
        )
        prefill_prefill_chunk_size_range = np.arange(  # 创建预填充块大小的范围
            1, self._config.prediction_max_prefill_chunk_size + 1
        )
        prefill_batch_size, prefill_kv_cache_size, prefill_prefill_chunk_size = zip(  # 使用笛卡儿积将三者结合
            *product(
                prefill_batch_size_range,
                prefill_kv_cache_size_range,
                prefill_prefill_chunk_size_range,
            )
        )

        attention_df = pd.DataFrame(  # 创建一个 DataFrame 包含所有参数组合
            {
                "batch_size": decode_batch_size + prefill_batch_size,
                "kv_cache_size": decode_kv_cache_size + prefill_kv_cache_size,
                "prefill_chunk_size": decode_prefill_chunk_size + prefill_prefill_chunk_size,
            }
        )

        attention_df["is_decode"] = attention_df["prefill_chunk_size"] == 0  # 判断是否为解码过程
        attention_df["num_tokens"] = attention_df[  # 计算每行的最大 token 数
            ["prefill_chunk_size", "batch_size"]
        ].max(axis=1)
        attention_df["prefill_chunk_size_squared"] = (  # 计算预填充块大小的平方
            attention_df["prefill_chunk_size"] ** 2
        )

        prefill_df = attention_df[~attention_df["is_decode"]]  # 过滤出预填充过程的行
        decode_df = attention_df[attention_df["is_decode"]]  # 过滤出解码过程的行
        chunked_prefill_df = prefill_df[prefill_df["kv_cache_size"] > 0].copy()  # 过滤出键值缓存大小大于零的预填充行
        chunked_prefill_df["total_prefill_tokens"] = (  # 计算总的预填充 token 数
            chunked_prefill_df["kv_cache_size"] + chunked_prefill_df["prefill_chunk_size"]
        )

        predictions["attn_prefill"] = self._get_model_prediction(  # 获取并存储注意力预填充模型的预测
            "attn_prefill",
            self._models["attn_prefill"],
            prefill_df[["kv_cache_size", "prefill_chunk_size_squared"]],
        )

        predictions["attn_decode"] = self._get_model_prediction(  # 获取并存储注意力解码模型的预测
            "attn_decode",
            self._models["attn_decode"],
            decode_df[["batch_size", "kv_cache_size"]],
        )

        return predictions  # 返回注意力模型的预测结果

    def _predict_from_models(self) -> Dict[str, Any]:
        predictions = self._predict_for_compute_models()  # 获取计算模型的预测
        predictions.update(self._predict_for_cpu_overhead_models())  # 更新并添加 CPU 开销模型的预测
        predictions.update(self._predict_for_attention_layer_models())  # 更新并添加注意力层模型的预测

        return predictions  # 返回所有模型的预测结果

    def _get_batch_decode_attention_params(self, batch: Batch) -> Tuple[int, int]:
        if hasattr(batch, "_decode_params"):  # 检查批次是否已经有了解码参数
            return batch._decode_params  # 返回已存在的解码参数

        decode_kv_cache_sizes = []  # 初始化一个列表用于存储解码键值缓存大小

        for request in batch.requests:  # 遍历批次中的每个请求
            if request._is_prefill_complete:  # 如果请求已经完成预填充
                decode_kv_cache_sizes.append(request.num_processed_tokens)  # 添加处理过的 token 数到列表

        if not decode_kv_cache_sizes:  # 如果解码键值缓存大小列表为空
            batch._decode_params = (0, 0)  # 将解码参数设置为 (0, 0)
            return batch._decode_params  # 返回解码参数

        decode_batch_size = len(decode_kv_cache_sizes)  # 解码批次大小为组合数量
        decode_avg_kv_cache_size = int(np.mean(decode_kv_cache_sizes))  # 计算解码键值缓存大小的平均值
        decode_avg_kv_cache_size = (  # 向上取整到键值缓存预测粒度
            (
                decode_avg_kv_cache_size
                + self._config.kv_cache_prediction_granularity
                - 1
            )
            // self._config.kv_cache_prediction_granularity
        ) * self._config.kv_cache_prediction_granularity

        batch._decode_params = (decode_batch_size, decode_avg_kv_cache_size)  # 设置解码参数

        return batch._decode_params  # 返回解码参数

    def _get_batch_prefill_attention_params(
        self, batch: Batch
    ) -> List[Tuple[int, int]]:
        if hasattr(batch, "_prefill_params"):  # 检查批次是否已经有了预填充参数
            return batch._prefill_params  # 返回已存在的预填充参数

        prefill_params = []  # 初始化一个列表用于存储预填充参数

        for request, num_tokens_to_process in zip(batch.requests, batch.num_tokens):  # 遍历批次中的请求和相应的 token 数
            if request._is_prefill_complete:  # 如果请求已经完成预填充
                continue  # 跳过此请求

            prefill_chunk_size = num_tokens_to_process  # 设置预填充块大小为需要处理的 token 数
            kv_cache_size = (  # 计算键值缓存大小，向上取整到键值缓存预测粒度
                (
                    request.num_processed_tokens
                    + self._config.kv_cache_prediction_granularity
                    - 1
                )
                // self._config.kv_cache_prediction_granularity
            ) * self._config.kv_cache_prediction_granularity

            prefill_params.append((kv_cache_size, prefill_chunk_size))  # 添加预填充参数到列表

        batch._prefill_params = prefill_params  # 设置预填充参数

        return prefill_params  # 返回预填充参数

    def _get_attention_layer_pre_proj_execution_time(self, batch: Batch) -> float:
        return self._predictions["attn_pre_proj"][(batch._total_num_tokens_rounded,)]  # 获取注意力层前投影的执行时间

    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:
        return self._predictions["attn_post_proj"][(batch._total_num_tokens_rounded,)]  # 获取注意力层后投影的执行时间

    def _get_mlp_layer_up_proj_execution_time(self, batch: Batch) -> float:
        return self._predictions["mlp_up_proj"][(batch._total_num_tokens_rounded,)]  # 获取 MLP 层向上投影的执行时间

    def _get_mlp_layer_down_proj_execution_time(self, batch: Batch) -> float:
        return self._predictions["mlp_down_proj"][(batch._total_num_tokens_rounded,)]  # 获取 MLP 层向下投影的执行时间

    def _get_mlp_layer_act_execution_time(self, batch: Batch) -> float:
        return self._predictions["mlp_act"][(batch._total_num_tokens_rounded,)]  # 获取 MLP 层激活的执行时间

    def _get_attn_norm_layer_act_execution_time(self, batch: Batch) -> float:
        return self._predictions["input_layernorm"][(batch._total_num_tokens_rounded,)]  # 获取注意力层规范化的执行时间

    def _get_mlp_norm_layer_act_execution_time(self, batch: Batch) -> float:
        if not self._model_config.post_attn_norm:  # 如果模型配置中没有设置后注意力规范化
            return 0  # 返回 0 表示不需要执行时间

        return self._predictions["post_attention_layernorm"][
            (batch._total_num_tokens_rounded,)
        ]  # 获取 MLP 规范化层的执行时间

    def _get_add_layer_act_execution_time(self, batch: Batch) -> float:
        return self._predictions["add"][(batch._total_num_tokens_rounded,)]  # 获取加层的激活执行时间

    def _get_tensor_parallel_communication_time(self, batch: Batch) -> float:
        return (
            self._predictions["all_reduce"][(batch._total_num_tokens_rounded,)]  # 获取张量并行通信的全规约执行时间
            + self._config.nccl_cpu_launch_overhead_ms  # 添加 NCCL CPU 启动的开销
            + self._config.nccl_cpu_skew_overhead_per_device_ms  # 添加 NCCL CPU 偏斜的单设备开销
            * self._replica_config.tensor_parallel_size**1.25  # 根据张量并行大小计算额外开销
        )

    def _get_pipeline_parallel_communication_time(self, batch: Batch) -> float:
        try:
            return self._predictions["send_recv"][(batch._total_num_tokens_rounded,)]  # 获取流水并行通信的发送接收执行时间
        except KeyError as e:  # 如果出现 KeyError 异常
            logger.error(f"Failed to get send_recv prediction for batch {batch}")  # 记录错误日志
            raise e  # 重新抛出异常

    def _get_attention_rope_execution_time(self, batch: Batch) -> float:
        return self._predictions["attn_rope"][(batch._total_num_tokens_rounded,)]  # 获取注意力旋转位置编码的执行时间

    def _get_attention_kv_cache_save_execution_time(self, batch: Batch) -> float:
        # 不要向上取整到 8 的倍数，因为我们要预测的是精确的 token 数
        num_tokens = sum(batch.num_tokens)  # 计算请求内的总 token 数

        return self._predictions["attn_kv_cache_save"][(num_tokens,)]  # 获取注意力键值缓存保存的执行时间

    def _get_attention_decode_execution_time(self, batch: Batch) -> float:
        (
            decode_batch_size,
            decode_avg_kv_cache_size,
        ) = self._get_batch_decode_attention_params(batch)  # 获取解码的批次注意力参数
        if decode_batch_size == 0:  # 如果解码批次大小为零
            return 0  # 返回零表示无需执行时间

        return self._predictions["attn_decode"][
            (decode_batch_size, decode_avg_kv_cache_size)
        ] * (
            1
            + self._attention_decode_batching_overhead_fraction  # 添加 attention 解码分批的额外开销
            * int(decode_batch_size > 1)  # 如果批次大于 1，则乘以额外开销
        )

    def _get_attention_prefill_execution_time(self, batch: Batch) -> float:
        prefill_params = self._get_batch_prefill_attention_params(batch)  # 获取预填充 attention 参数

        if len(prefill_params) == 0:  # 如果没有预填充参数
            return 0  # 返回零表示无需执行时间

        kv_cache_sizes, prefill_chunk_sizes = zip(*prefill_params)  # 解包预填充参数

        agg_kv_cache_size = sum(kv_cache_sizes)  # 计算总的键值缓存大小
        agg_prefill_chunk_size = sum([x**2 for x in prefill_chunk_sizes]) ** 0.5  # 计算预填充块大小的平方和的平方根

        return self._predictions["attn_prefill"][
            (agg_kv_cache_size, round(agg_prefill_chunk_size) ** 2)
        ] * (
            1
            + self._attention_prefill_batching_overhead_fraction  # 添加 attention 预填充分批的额外开销
            * int(len(prefill_params) > 1)  # 如果预填充参数多于一个，则乘以额外开销
        )

    def _get_schedule_time(self, batch: Batch) -> float:
        if self._config.skip_cpu_overhead_modeling:  # 如果配置中设置了跳过 CPU 开销建模
            return 0  # 返回零表示不需要时间

        return self._predictions["schedule"][(batch.size,)]  # 获取调度模型的执行时间

    def _get_sampler_e2e_time(self, batch: Batch) -> float:
        if self._config.skip_cpu_overhead_modeling:  # 如果配置中设置了跳过 CPU 开销建模
            return 0  # 返回零表示不需要时间

        return self._predictions["sampler_e2e"][(batch.size,)]  # 获取采样器端到端的执行时间

    def _get_prepare_inputs_e2e_time(self, batch: Batch) -> float:
        if self._config.skip_cpu_overhead_modeling:  # 如果配置中设置了跳过 CPU 开销建模
            return 0  # 返回零表示不需要时间

        return self._predictions["prepare_inputs_e2e"][(batch.size,)]  # 获取准备输入端到端的执行时间

    def _get_process_model_outputs_time(self, batch: Batch) -> float:
        if self._config.skip_cpu_overhead_modeling:  # 如果配置中设置了跳过 CPU 开销建模
            return 0  # 返回零表示不需要时间

        return self._predictions["process_model_outputs"][(batch.size,)]  # 获取处理模型输出的执行时间

    def _get_ray_comm_time(self, batch: Batch) -> float:
        if self._config.skip_cpu_overhead_modeling:  # 如果配置中设置了跳过 CPU 开销建模
            return 0  # 返回零表示不需要时间

        return self._predictions["ray_comm_time"][(batch.size,)]  # 获取 Ray 通信时间

    def to_dict(self) -> dict:
        return {
            "model_provider": str(self._config.get_type()),  # 模型提供者的类型字符串
            "num_tensor_parallel_workers": self._replica_config.tensor_parallel_size,  # 张量并行的工作者数
            "k_fold_cv_splits": self._config.k_fold_cv_splits,  # k 折交叉验证的拆分数
            "num_q_heads": self._model_config.num_q_heads,  # 查询头的数量
            "num_kv_heads": self._model_config.num_kv_heads,  # 键值头的数量
            "embedding_dim": self._model_config.embedding_dim,  # 嵌入维度
            "mlp_hidden_dim": self._model_config.mlp_hidden_dim,  # MLP 隐层维度
            "use_gated_mlp": self._model_config.use_gated_mlp,  # 是否使用门控 MLP
            "vocab_size": self._model_config.vocab_size,  # 词汇表大小
            "block_size": self._block_size,  # 块大小
            "max_tokens": self._max_tokens,  # 最大 token 数
            "compute_input_file": self._compute_input_file,  # 计算输入文件
            "all_reduce_input_file": self._all_reduce_input_file,  # 全规约输入文件
            "send_recv_input_file": self._send_recv_input_file,  # 发送接收输入文件
            "cpu_overhead_input_file": self._cpu_overhead_input_file,  # CPU 开销输入文件
            "prediction_max_prefill_chunk_size": self._config.prediction_max_prefill_chunk_size,  # 预测的最大预填充块大小
            "max_batch_size": self._config.prediction_max_batch_size,  # 最大批次大小
        }
