from abc import ABC, abstractmethod  # 从 abc 模块导入 ABC 和 abstractmethod

from vidur.config import (  # 从 vidur.config 模块导入所需配置类
    BaseExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    ReplicaConfig,
)
from vidur.entities import Batch, ExecutionTime  # 从 vidur.entities 模块导入 Batch 和 ExecutionTime 实体类

class BaseExecutionTimePredictor(ABC):  # 定义 BaseExecutionTimePredictor 抽象基类
    def __init__(  # 定义初始化方法
        self,
        predictor_config: BaseExecutionTimePredictorConfig,  # 执行时间预测器配置
        replica_config: ReplicaConfig,  # 副本配置
        replica_scheduler_config: BaseReplicaSchedulerConfig,  # 副本调度器配置
        metrics_config: MetricsConfig,  # 度量配置
    ) -> None:
        self._config = predictor_config  # 初始化执行时间预测器配置
        self._replica_config = replica_config  # 初始化副本配置
        self._model_config = replica_config.model_config  # 初始化模型配置

        # 获取配置
        self._replica_scheduler_provider = str(replica_scheduler_config.get_type())  # 初始化副本调度器提供者类型
        self._block_size = replica_scheduler_config.block_size  # 初始化块大小
        self._cache_dir = metrics_config.cache_dir  # 初始化缓存目录
        self._num_layers_per_pipeline_stage = (  # 计算每个流水线阶段的层数
            self._model_config.num_layers // self._replica_config.num_pipeline_stages
        )

    def get_execution_time(self, batch: Batch, pipeline_stage: int) -> ExecutionTime:  # 获取执行时间的方法
        if pipeline_stage == self._replica_config.num_pipeline_stages - 1:  # 判断是否为最后一个流水线阶段
            pipeline_parallel_communication_time = 0  # 最后一个阶段没有流水线并行通信时间
        else:
            pipeline_parallel_communication_time = (
                self._get_pipeline_parallel_communication_time(batch)  # 获取流水线并行通信时间
            )

        if self._replica_config.tensor_parallel_size == 1:  # 判断张量并行大小是否为 1
            tensor_parallel_communication_time = 0  # 张量并行大小为 1 没有通信时间
        else:
            tensor_parallel_communication_time = (
                self._get_tensor_parallel_communication_time(batch)  # 获取张量并行通信时间
            )

        return ExecutionTime(  # 返回执行时间实例
            self._num_layers_per_pipeline_stage,  # 每个流水线阶段的层数
            self._get_attention_rope_execution_time(batch),  # 获取注意力绳索执行时间
            self._get_attention_kv_cache_save_execution_time(batch),  # 获取注意力键值缓存保存执行时间
            self._get_attention_decode_execution_time(batch),  # 获取注意力解码执行时间
            self._get_attention_prefill_execution_time(batch),  # 获取注意力预填充执行时间
            self._get_attention_layer_pre_proj_execution_time(batch),  # 获取注意力层前投影执行时间
            self._get_attention_layer_post_proj_execution_time(batch),  # 获取注意力层后投影执行时间
            self._get_mlp_layer_up_proj_execution_time(batch),  # 获取 MLP 层上投影执行时间
            self._get_mlp_layer_down_proj_execution_time(batch),  # 获取 MLP 层下投影执行时间
            self._get_mlp_layer_act_execution_time(batch),  # 获取 MLP 层激活执行时间
            self._get_attn_norm_layer_act_execution_time(batch),  # 获取注意力规范层激活执行时间
            self._get_mlp_norm_layer_act_execution_time(batch),  # 获取 MLP 规范层激活执行时间
            self._get_add_layer_act_execution_time(batch),  # 获取求和层激活执行时间
            tensor_parallel_communication_time,  # 张量并行通信时间
            pipeline_parallel_communication_time,  # 流水线并行通信时间
            self._get_schedule_time(batch),  # 获取调度时间
            self._get_sampler_e2e_time(batch),  # 获取采样器端到端时间
            self._get_prepare_inputs_e2e_time(batch),  # 获取准备输入端到端时间
            self._get_process_model_outputs_time(batch),  # 获取处理模型输出时间
            self._get_ray_comm_time(batch),  # 获取 Ray 通信时间
        )

    @abstractmethod
    def _get_attention_layer_pre_proj_execution_time(self, batch: Batch) -> float:  # 抽象方法：获取注意力层前投影执行时间
        pass

    @abstractmethod
    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:  # 抽象方法：获取注意力层后投影执行时间
        pass

    @abstractmethod
    def _get_attention_rope_execution_time(self, batch: Batch) -> float:  # 抽象方法：获取注意力绳索执行时间
        pass

    @abstractmethod
    def _get_attention_kv_cache_save_execution_time(self, batch: Batch) -> float:  # 抽象方法：获取注意力键值缓存保存执行时间
        pass

    @abstractmethod
    def _get_attention_decode_execution_time(self, batch: Batch) -> float:  # 抽象方法：获取注意力解码执行时间
        pass

    @abstractmethod
    def _get_attention_prefill_execution_time(self, batch: Batch) -> float:  # 抽象方法：获取注意力预填充执行时间
        pass

    @abstractmethod
    def _get_mlp_layer_up_proj_execution_time(self, batch: Batch) -> float:  # 抽象方法：获取 MLP 层上投影执行时间
        pass

    @abstractmethod
    def _get_mlp_layer_down_proj_execution_time(self, batch: Batch) -> float:  # 抽象方法：获取 MLP 层下投影执行时间
        pass

    @abstractmethod
    def _get_mlp_layer_act_execution_time(self, batch: Batch) -> float:  # 抽象方法：获取 MLP 层激活执行时间
        pass

    @abstractmethod
    def _get_tensor_parallel_communication_time(self, batch: Batch) -> float:  # 抽象方法：获取张量并行通信时间
        pass

    @abstractmethod
    def _get_pipeline_parallel_communication_time(self, batch: Batch) -> float:  # 抽象方法：获取流水线并行通信时间
        pass

    @abstractmethod
    def _get_schedule_time(self, batch: Batch) -> float:  # 抽象方法：获取调度时间
        pass

    @abstractmethod
    def _get_sampler_e2e_time(self, batch: Batch) -> float:  # 抽象方法：获取采样器端到端时间
        pass

    @abstractmethod
    def _get_prepare_inputs_e2e_time(self, batch: Batch) -> float:  # 抽象方法：获取准备输入端到端时间
        pass

    @abstractmethod
    def _get_process_model_outputs_time(self, batch: Batch) -> float:  # 抽象方法：获取处理模型输出时间
        pass

    @abstractmethod
    def _get_ray_comm_time(self, batch: Batch) -> float:  # 抽象方法：获取 Ray 通信时间
        pass

    @abstractmethod
    def _get_mlp_norm_layer_act_execution_time(self, batch: Batch) -> float:  # 抽象方法：获取 MLP 规范层激活执行时间
        pass

    @abstractmethod
    def _get_attn_norm_layer_act_execution_time(self, batch: Batch) -> float:  # 抽象方法：获取注意力规范层激活执行时间
        pass

    @abstractmethod
    def _get_add_layer_act_execution_time(self, batch: Batch) -> float:  # 抽象方法：获取求和层激活执行时间
        pass
