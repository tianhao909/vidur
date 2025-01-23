import hashlib  # 导入hashlib模块，用于生成哈希值
from dataclasses import dataclass, field  # 导入dataclass和field，用于创建数据类
from itertools import product  # 导入product，用于生成排列组合
from typing import List, Optional  # 导入List和Optional，用于类型注解


def _get_hash(key):
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]  # 生成给定字符串的SHA1哈希值并截取前8位


@dataclass
class ModelConfig:
    name: str  # 模型配置的名称
    identifier: str  # 模型配置的标识符
    parallel_specs: List[str] = field(default_factory=list)  # 并行规格列表，默认为空列表
    scheduler_specs: List[str] = field(default_factory=list)  # 调度器规格列表，默认为空列表
    traces: List[str] = field(default_factory=list)  # 轨迹列表，默认为空列表

    def get_key(self):
        return self.name  # 返回模型配置的名称

    def get_human_readable_name(self):
        return f"Model: {self.name}"  # 返回人类可读的模型配置名称

    def to_config_dict(self):
        return {
            "model_config_model": self.identifier,  # 返回包含标识符的配置字典
        }

    def is_parallel_spec_valid(self, spec_name: str):
        return not self.parallel_specs or spec_name in self.parallel_specs  # 检查并行规格是否有效

    def is_scheduler_spec_valid(self, spec_name: str):
        return not self.scheduler_specs or spec_name in self.scheduler_specs  # 检查调度器规格是否有效

    def is_traces_valid(self, trace_name: str):
        return not self.traces or trace_name in self.traces  # 检查轨迹名称是否有效


@dataclass
class TraceConfig:
    name: str  # 轨迹配置的名称
    trace_file: str  # 轨迹文件路径
    max_seq_len: int  # 最大序列长度
    num_requests: int  # 请求数量
    start_qps: float  # 起始QPS（每秒查询率）

    def get_key(self):
        return f"{self.name}_tk{self.max_seq_len}_rq{self.num_requests}"  # 返回轨迹配置的键

    def get_human_readable_name(self):
        return f"Trace: {self.name}, Max Seq Len: {self.max_seq_len}, Num Requests: {self.num_requests}, Start QPS: {self.start_qps}"  # 返回人类可读的轨迹名称

    def to_config_dict(self):
        return {
            "request_generator_config_type": "SYNTHETIC",  # 请求生成器配置类型
            "length_generator_config_type": "TRACE",  # 长度生成器配置类型
            "interval_generator_config_type": "POISSON",  # 间隔生成器配置类型
            "trace_request_length_generator_config_max_tokens": self.max_seq_len,  # 轨迹请求长度生成器配置最大令牌数
            "model_config_max_model_len": self.max_seq_len,  # 模型配置最大模型长度
            "trace_request_length_generator_config_trace_file": self.trace_file,  # 轨迹请求长度生成器配置轨迹文件
            "trace_request_length_generator_config_prefill_scale_factor": 1,  # 轨迹请求长度生成器配置预填比例因子
            "trace_request_length_generator_config_decode_scale_factor": 1,  # 轨迹请求长度生成器配置解码比例因子
            "synthetic_request_generator_config_num_requests": self.num_requests,  # 合成请求生成器配置请求数量
            "vllm_scheduler_config_max_batched_tokens": self.max_seq_len,  # vllm调度器配置最大批处理令牌数
        }


@dataclass
class SchedulerConfig:
    name: str  # 调度器配置的名称
    scheduler: str  # 调度器类型
    batch_size: int  # 批处理大小
    chunk_size: Optional[int] = None  # 块大小，可选

    def get_key(self):
        key = f"{self.scheduler}_bs{self.batch_size}"  # 初始化键值

        if self.chunk_size is not None:
            key += f"_cs{self.chunk_size}"  # 如果块大小不为None，将其添加到键值中

        return key  # 返回键值

    def get_human_readable_name(self):
        return f"Scheduler: {self.scheduler}, Batch Size: {self.batch_size}, Chunk Size: {self.chunk_size}"  # 返回人类可读的调度器名称

    def to_config_dict(self):
        if self.scheduler == "vllm":
            return {
                "scheduler_config_type": "VLLM",  # 调度器配置类型为VLLM
                "vllm_scheduler_config_max_num_seqs": self.batch_size,  # VLLM调度器配置最大批次序列数
            }
        elif self.scheduler == "orca":
            return {
                "scheduler_config_type": "ORCA",  # 调度器配置类型为ORCA
                "orca_scheduler_config_max_num_seqs": self.batch_size,  # ORCA调度器配置最大批次序列数
            }
        elif self.scheduler == "sarathi":
            assert self.chunk_size is not None
            return {
                "scheduler_config_type": "SARATHI",  # 调度器配置类型为SARATHI
                "sarathi_scheduler_config_max_num_seqs": self.batch_size,  # SARATHI调度器配置最大批次序列数
                "sarathi_scheduler_config_chunk_size": self.chunk_size,  # SARATHI调度器配置块大小
            }
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")  # 抛出异常，未知的调度器类型


@dataclass
class ParallelConfig:
    name: str  # 并行配置的名称
    tp_dimension: int  # 张量并行维度
    pp_dimension: int  # 流水线并行维度

    def get_key(self):
        return f"tp{self.tp_dimension}_pp{self.pp_dimension}"  # 返回并行配置的键

    def get_human_readable_name(self):
        return f"TP: {self.tp_dimension}, PP: {self.pp_dimension}"  # 返回人类可读的并行配置名称

    def get_num_gpus(self):
        return self.tp_dimension * self.pp_dimension  # 计算并返回所需GPU数量

    def to_config_dict(self):
        return {
            "parallel_config_tensor_parallel_size": self.tp_dimension,  # 并行配置的张量并行大小
            "parallel_config_pipeline_parallel_size": self.pp_dimension,  # 并行配置的流水线并行大小
        }
class JobConfig:  # 定义一个名为 JobConfig 的类

    def __init__(  # 初始化方法，当创建类的实例时被调用
        self,
        model_config: ModelConfig,  # 传入模型配置的参数
        trace_config: TraceConfig,  # 传入跟踪配置的参数
        scheduler_config: SchedulerConfig,  # 传入调度器配置的参数
        parallel_config: ParallelConfig,  # 传入并行处理配置的参数
    ):
        self.model_config = model_config  # 将模型配置参数赋值给实例变量
        self.trace_config = trace_config  # 将跟踪配置参数赋值给实例变量
        self.scheduler_config = scheduler_config  # 将调度器配置参数赋值给实例变量
        self.parallel_config = parallel_config  # 将并行配置参数赋值给实例变量

        self.start_qps = self.trace_config.start_qps  # 从跟踪配置中获取初始QPS（query per second）

    def get_key(self):  # 定义一个方法用于生成配置的键
        config_keys = [  # 创建一个列表，用于存储每个配置的键
            self.model_config.get_key(),  # 获取模型配置的键
            self.trace_config.get_key(),  # 获取跟踪配置的键
            self.scheduler_config.get_key(),  # 获取调度器配置的键
            self.parallel_config.get_key(),  # 获取并行配置的键
        ]

        return "_".join(config_keys)  # 将列表中的键通过下划线连接成一个字符串返回

    def get_wandb_run_name(self):  # 定义一个方法用于生成 WandB 的运行名称
        substrings = [  # 创建一个列表，用于存储每个配置的 WandB 运行名
            self.model_config.get_wandb_run_name(),  # 获取模型配置的 WandB 运行名
            self.trace_config.get_wandb_run_name(),  # 获取跟踪配置的 WandB 运行名
            self.scheduler_config.get_wandb_run_name(),  # 获取调度器配置的 WandB 运行名
            self.parallel_config.get_wandb_run_name(),  # 获取并行配置的 WandB 运行名
        ]
        return "_".join(substrings)  # 将列表中的名称通过下划线连接成一个字符串返回

    def get_human_readable_name(self):  # 定义一个方法用于生成可读名称
        substrings = [  # 创建一个列表，用于存储每个配置的人类可读名
            self.model_config.get_human_readable_name(),  # 获取模型配置的人类可读名
            self.trace_config.get_human_readable_name(),  # 获取跟踪配置的人类可读名
            self.scheduler_config.get_human_readable_name(),  # 获取调度器配置的人类可读名
            self.parallel_config.get_human_readable_name(),  # 获取并行配置的人类可读名
            f"Hash: {_get_hash(self.get_key())}",  # 获取当前配置的散列值并将其格式化为字符串
        ]
        return ", ".join(substrings)  # 将列表中的名称通过逗号连接成一个字符串返回

    def get_num_gpus(self):  # 定义一个方法用于获取使用的 GPU 数量
        return self.parallel_config.get_num_gpus()  # 返回并行配置中定义的 GPU 数量

    def to_config_dict(self):  # 定义一个方法用于将配置转换为字典
        return {  # 返回一个合并了所有配置字典的字典
            **self.model_config.to_config_dict(),  # 添加模型配置的字典
            **self.trace_config.to_config_dict(),  # 添加跟踪配置的字典
            **self.parallel_config.to_config_dict(),  # 添加并行配置的字典
            **self.scheduler_config.to_config_dict(),  # 添加调度器配置的字典
        }

    @classmethod  # 类方法，使用类本身作为第一个参数
    def generate_job_configs(cls, config: dict):  # 定义一个方法用于生成一组作业配置
        job_configs = []  # 创建一个空列表用于存储生成的作业配置
        for (  # 使用嵌套循环遍历所有可能的组合
            model_config,
            trace_config,
            scheduler_config,
            parallel_config,
        ) in product(
            config["models"],  # 遍历所有模型配置
            config["traces"],  # 遍历所有跟踪配置
            config["schedulers"],  # 遍历所有调度器配置
            config["parallel_spec"],  # 遍历所有并行配置
        ):
            model_config = ModelConfig(**model_config)  # 创建模型配置实例
            trace_config = TraceConfig(**trace_config)  # 创建跟踪配置实例
            scheduler_config = SchedulerConfig(**scheduler_config)  # 创建调度器配置实例
            parallel_config = ParallelConfig(**parallel_config)  # 创建并行配置实例

            if (  # 如果配置组合无效则跳过
                not model_config.is_parallel_spec_valid(parallel_config.name)
                or not model_config.is_scheduler_spec_valid(scheduler_config.name)
                or not model_config.is_traces_valid(trace_config.name)
            ):
                continue

            job_config = cls(  # 创建一个新的作业配置实例
                model_config,
                trace_config,
                scheduler_config,
                parallel_config,
            )
            job_configs.append(job_config)  # 将新的作业配置添加到列表中

        return job_configs  # 返回生成的作业配置列表

    def __str__(self) -> str:  # 重载字符串表示方法
        return self.get_human_readable_name()  # 返回可读名称


@dataclass  # 使用 dataclass 装饰器自动生成初始化方法
class BenchmarkConfig:
    output_dir: str  # 定义一个名为 output_dir 的字符串变量
    wandb_project: str  # 定义一个名为 wandb_project 的字符串变量
    wandb_group: str  # 定义一个名为 wandb_group 的字符串变量
    wandb_sweep_id: str  # 定义一个名为 wandb_sweep_id 的字符串变量
    qps: float  # 定义一个名为 qps 的浮点数变量
    time_limit: int  # 定义一个名为 time_limit 的整数变量
    job_config: JobConfig  # 定义一个名为 job_config 的 JobConfig 类型变量

    def to_config_dict(self):  # 定义一个方法用于将配置转换为字典
        if self.wandb_project:  # 如果 WandB 项目名称不为空
            wandb_args = {  # 创建 WandB 参数字典
                "metrics_config_wandb_project": self.wandb_project,
                "metrics_config_wandb_group": self.job_config.get_key(),
                "metrics_config_wandb_sweep_id": self.wandb_sweep_id,
                "metrics_config_wandb_run_id": self.get_run_id(),
                "metrics_config_wandb_run_name": f"qps_{self.qps}",
            }
        else:
            wandb_args = {}  # 如果项目名称为空则创建一个空字典

        return {  # 返回合并了所有配置的字典
            **self.job_config.to_config_dict(),  # 添加作业配置的字典
            "output_dir": self.get_run_dir(),
            "poisson_request_interval_generator_config_qps": self.qps,
            "time_limit": self.time_limit * 60,  # 转换为秒
            "metrics_config_enable_op_level_metrics": False,
            "metrics_config_enable_cpu_op_level_metrics": False,
            "metrics_config_keep_individual_batch_metrics": False,
            "metrics_config_enable_chrome_trace": False,
            **wandb_args,  # 添加 WandB 参数字典
        }

    def get_run_id(self):  # 定义获取运行 ID 的方法
        return _get_hash(self.get_key())  # 返回配置键的散列值

    def get_key(self):  # 定义获取键的方法
        return f"{self.job_config.get_key()}_qps{self.qps}"  # 返回作业配置键和 QPS 的组合字符串

    def to_args(self):  # 定义将配置转换为命令行参数的方法
        args = []  # 创建一个空列表用于存储参数

        for key, value in self.to_config_dict().items():  # 遍历配置字典中的键和值
            if value is not None:  # 如果值不是 None
                args.append(f"--{key} {value}")  # 添加参数和值到列表中
            else:
                args.append(f"--{key}")  # 仅添加参数

        return " ".join(args)  # 将列表中的参数通过空格连接成一个字符串返回

    def to_human_readable_name(self):  # 定义生成可读名称的方法
        return f"{self.job_config.get_human_readable_name()}, QPS: {self.qps}, Run id: {self.get_run_id()}"  # 返回可读名称字符串

    def get_run_dir(self):  # 定义获取运行目录的方法
        return (  # 返回格式化的运行目录字符串
            f"{self.output_dir}/runs/{_get_hash(self.job_config.get_key())}/{self.qps}"
        )

