import hashlib  # 导入 hashlib 模块，用于生成哈希值
from dataclasses import dataclass  # 导入 dataclass 装饰器，用于简化类定义
from itertools import product  # 导入 product 函数，用于生成多个可迭代对象的笛卡尔积
from typing import List, Optional  # 导入类型提示模块中的 List 和 Optional

@dataclass  # 使用 dataclass 装饰器自动生成类的特殊方法
class ModelConfig:
    name: str  # 定义模型名称属性
    identifier: str  # 定义模型标识符属性
    exclude_tp_dims: List[int] = None  # 定义排除的张量并行维度列表，默认值为 None

    def get_key(self):
        return self.name  # 返回模型名称作为键

    def to_config_dict(self):
        return {
            "replica_config_model_name": self.identifier,  # 返回模型配置字典
        }

    def is_tensor_parallel_degree_valid(self, tp_degree: int):
        # 检查张量并行度是否有效，如果没有排除列表或并行度不在排除列表中，则返回 True
        return self.exclude_tp_dims is None or tp_degree not in self.exclude_tp_dims


@dataclass
class TraceConfig:
    name: str  # 定义跟踪名称属性
    trace_file: str  # 定义跟踪文件路径属性
    max_seq_len: int  # 定义最大序列长度属性
    num_requests: int  # 定义请求数量属性
    start_qps: float  # 定义起始QPS（每秒查询数）属性

    def get_key(self):
        # 返回跟踪的键，由名称、最大序列长度和请求数量组成
        return f"{self.name}_tk{self.max_seq_len}_rq{self.num_requests}"

    def to_config_dict(self):
        # 返回跟踪配置字典
        return {
            "request_generator_config_type": "synthetic",
            "length_generator_config_type": "trace",
            "interval_generator_config_type": "poisson",
            "synthetic_request_generator_config_max_tokens": self.max_seq_len,
            "trace_request_length_generator_config_max_tokens": self.max_seq_len,
            "zipf_request_length_generator_config_max_tokens": self.max_seq_len,
            "uniform_request_length_generator_config_max_tokens": self.max_seq_len,
            "fixed_request_length_generator_config_max_tokens": self.max_seq_len,
            "trace_request_generator_config_max_tokens": self.max_seq_len,
            "trace_request_length_generator_config_trace_file": self.trace_file,
            "trace_request_length_generator_config_prefill_scale_factor": 1,
            "trace_request_length_generator_config_decode_scale_factor": 1,
            "synthetic_request_generator_config_num_requests": self.num_requests,
            "vllm_scheduler_config_max_tokens_in_batch": self.max_seq_len,
        }


@dataclass
class ClusterConfig:
    device: str  # 定义设备类型属性
    num_gpus: int  # 定义GPU数量属性
    gpus_per_node: int  # 定义每个节点的GPU数量属性

    def get_key(self):
        return self.device  # 返回设备类型作为键

    def to_config_dict(self):
        return {
            "replica_config_device": self.device,  # 返回集群配置字典
        }


@dataclass
class SchedulerConfig:
    scheduler: str  # 定义调度器类型属性
    chunk_size: Optional[int] = None  # 定义块大小属性，默认为 None

    def get_key(self):
        scheduler = self.scheduler  # 将调度器类型赋给局部变量

        if self.chunk_size is not None:
            scheduler += f"_cs{self.chunk_size}"  # 如果块大小不为空，将其添加到调度器键中

        return scheduler  # 返回调度器键，包括块大小（如果存在）

    def to_config_dict(self):
        if self.scheduler == "vllm":
            return {
                "replica_scheduler_config_type": "vllm",  # 如果调度器类型是 vllm，返回对应配置字典
            }

        assert self.scheduler == "sarathi"  # 断言调度器类型必须是 sarathi
        assert self.chunk_size is not None  # 断言块大小不为空
        return {
            "replica_scheduler_config_type": "sarathi",
            "sarathi_scheduler_config_chunk_size": self.chunk_size,  # 返回 sarathi 调度器配置字典
        }


class JobConfig:
    def __init__(
        self,
        model_config: ModelConfig,  # 模型配置对象
        trace_config: TraceConfig,  # 跟踪配置对象
        cluster_config: ClusterConfig,  # 集群配置对象
        scheduler_config: SchedulerConfig,  # 调度器配置对象
        num_tensor_parallel_workers: int,  # 张量并行工作的数量
        num_pipeline_stages: int,  # 流水线阶段的数量
        batch_size: int,  # 批次大小
    ):
        self.model_config = model_config  # 保存模型配置对象
        self.trace_config = trace_config  # 保存跟踪配置对象
        self.cluster_config = cluster_config  # 保存集群配置对象
        self.scheduler_config = scheduler_config  # 保存调度器配置对象
        self.num_tensor_parallel_workers = num_tensor_parallel_workers  # 保存张量并行工作数量
        self.num_pipeline_stages = num_pipeline_stages  # 保存流水线阶段数量
        self.num_workers = self.num_tensor_parallel_workers * self.num_pipeline_stages  # 计算工作者数量
        self.batch_size = batch_size * num_pipeline_stages  # 计算整体批次大小
        self.num_replicas = self.cluster_config.num_gpus // self.num_workers  # 计算副本数量

        self.start_qps = self.trace_config.start_qps  # 保存起始QPS

    def is_valid(self):
        # 检查配置是否有效：副本数量大于0，张量并行度有效，并且不超过每个节点的GPU数量
        return (
            self.num_replicas > 0
            and self.model_config.is_tensor_parallel_degree_valid(
                self.num_tensor_parallel_workers
            )
            and self.num_tensor_parallel_workers <= self.cluster_config.gpus_per_node
        )

    def get_key(self):
        # 返回作业的唯一键，由多个配置属性组合而成
        return (
            f"{self.model_config.name}_{self.trace_config.get_key()}_{self.cluster_config.get_key()}_{self.scheduler_config.get_key()}"
            f"_tp{self.num_tensor_parallel_workers}_pp{self.num_pipeline_stages}_bsz{self.batch_size}"
        )

    def get_human_readable_name(self):
        # 返回易读的作业名称，包含多种配置信息及哈希值
        return (
            f"Model: {self.model_config.name}, Trace: {self.trace_config.name}, Cluster: {self.cluster_config.device}, "
            f"Scheduler: {self.scheduler_config.scheduler}, TP: {self.num_tensor_parallel_workers}, "
            f"PP: {self.num_pipeline_stages}, BSZ: {self.batch_size}, CS: {self.scheduler_config.chunk_size}, Hash: {self.get_hash()}"
        )

    def get_hash(self):
        # 生成作业配置的哈希值
        return hashlib.sha1(self.get_key().encode("utf-8")).hexdigest()[:8]


    def to_config_dict(self):
        # 返回作业配置的所有组合字典
        return {
            **self.model_config.to_config_dict(),  # 追加模型配置字典
            **self.trace_config.to_config_dict(),  # 追加trace配置字典
            **self.cluster_config.to_config_dict(),  # 追加集群配置字典
            **self.scheduler_config.to_config_dict(),  # 追加调度器配置字典
            "replica_config_tensor_parallel_size": self.num_tensor_parallel_workers,  # 设置张量并行大小
            "replica_config_num_pipeline_stages": self.num_pipeline_stages,  # 设置流水线阶段数量
            "vllm_scheduler_config_batch_size_cap": self.batch_size,  # 设置 vllm 调度器批量大小上限
            "lightllm_scheduler_config_batch_size_cap": self.batch_size,  # 设置 lightllm 调度器批量大小上限
            "orca_scheduler_config_batch_size_cap": self.batch_size,  # 设置 orca 调度器批量大小上限
            "faster_transformer_scheduler_config_batch_size_cap": self.batch_size,  # 设置 faster transformer 调度器批量大小上限
            "sarathi_scheduler_config_batch_size_cap": self.batch_size,  # 设置 sarathi 调度器批量大小上限
            "cluster_config_num_replicas": self.num_replicas,  # 设置集群副本数量
        }


    @classmethod
    def generate_job_configs(cls, config: dict):
        job_configs = []  # 创建空列表用于存储作业配置
        for (
            model_config,
            trace_config,
            cluster_config,
            scheduler_config,
            tp_dimension,
            pp_dimension,
            batch_size,
        ) in product(
            config["models"],
            config["traces"],
            config["clusters"],
            config["schedulers"],
            config["tp_dimensions"],
            config["pp_dimensions"],
            config["batch_sizes"],
        ):
            job_config = cls(
                ModelConfig(**model_config),
                TraceConfig(**trace_config),
                ClusterConfig(**cluster_config),
                SchedulerConfig(**scheduler_config),
                tp_dimension,
                pp_dimension,
                batch_size,
            )
            if not job_config.is_valid():
                continue  # 如果作业配置无效，跳过此次循环

            job_configs.append(job_config)  # 将有效的作业配置添加到列表中

        return job_configs  # 返回生成的作业配置列表

    @classmethod
    def generate_unique_model_job_configs(cls, config: dict, num_requests: int = 32):
        job_configs = []  # 创建空列表用于存储唯一模型作业配置

        trace_config = TraceConfig(**config["traces"][0])
        trace_config.num_requests = num_requests  # 更改请求数量并保存
        scheduler_config = SchedulerConfig(**config["schedulers"][0])
        batch_size = config["batch_sizes"][0]  # 批次大小
        pp_dimensions = [2]  # 设置 pp_dimensions 为 2，因为它覆盖所有选项

        for model_config, cluster_config, tp_dimension, pp_dimension in product(
            config["models"],
            config["clusters"],
            config["tp_dimensions"],
            pp_dimensions,
        ):
            job_config = cls(
                ModelConfig(**model_config),
                trace_config,
                ClusterConfig(**cluster_config),
                scheduler_config,
                tp_dimension,
                pp_dimension,
                batch_size,
            )
            if not job_config.is_valid():
                continue  # 如果作业配置无效，跳过此次循环

            job_configs.append(job_config)  # 将有效的作业配置添加到列表中

        return job_configs  # 返回生成的唯一模型作业配置列表


@dataclass
class SimulationConfig:
    output_dir: str  # 定义输出目录属性
    cache_dir: str  # 定义缓存目录属性
    qps: float  # 定义QPS（每秒查询数）属性
    time_limit: int  # 定义时间限制属性（以分钟为单位）
    job_config: JobConfig  # 定义作业配置对象

    def to_config_dict(self):
        # 返回模拟配置的所有组合字典
        return {
            **self.job_config.to_config_dict(),
            "metrics_config_output_dir": self.get_run_dir(),
            "metrics_config_cache_dir": self.cache_dir,
            "poisson_request_interval_generator_config_qps": self.qps,
            "gamma_request_interval_generator_config_qps": self.qps,
            "time_limit": self.time_limit * 60,  # 将时间限制转换为秒
            "no-metrics_config_save_table_to_wandb": None,
            "no-metrics_config_store_plots": None,
            "no-metrics_config_store_operation_metrics": None,
            "no-metrics_config_store_token_completion_metrics": None,
            "no-metrics_config_enable_chrome_trace": None,
            "linear_regression_execution_time_predictor_config_skip_cpu_overhead_modeling": None,
            "random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling": None,
        }

    def to_args(self):
        args = []  # 创建空列表用于存储参数

        for key, value in self.to_config_dict().items():
            if value is not None:
                args.append(f"--{key} {value}")  # 如果值不为空，格式化为参数并添加到列表中
            else:
                args.append(f"--{key}")  # 否则，只添加参数键

        return " ".join(args)  # 返回参数列表，以空格分隔成字符串

    def to_human_readable_name(self):
        # 返回易读的模拟名称，包括作业名称和QPS
        return f"{self.job_config.get_human_readable_name()}, QPS: {self.qps}"

    def get_run_dir(self):
        # 返回运行目录，包含输出目录、作业哈希值和QPS
        return f"{self.output_dir}/runs/{self.job_config.get_hash()}/{self.qps}"
