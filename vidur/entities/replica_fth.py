from math import ceil  # 导入`ceil`函数，用于向上取整

from vidur.config import BaseRequestGeneratorConfig, ReplicaConfig  # 导入配置相关的模块
from vidur.entities.base_entity import BaseEntity  # 导入基本实体类
from vidur.logger import init_logger  # 导入日志初始化函数

logger = init_logger(__name__)  # 初始化日志

class Replica(BaseEntity):  # 定义Replica类，继承自BaseEntity
    def __init__(
        self,
        replica_config: ReplicaConfig,  # `replica_config`的类型为`ReplicaConfig`
        generator_config: BaseRequestGeneratorConfig,  # `generator_config`的类型为`BaseRequestGeneratorConfig`
    ) -> None:
        self._id = Replica.generate_id()  # 对象的唯一标识符

        self._replica_config = replica_config  # 存储副本配置
        self._model_config = replica_config.model_config  # 存储模型配置
        self._device_config = replica_config.device_config  # 存储设备配置
        self._generator_config = generator_config  # 存储生成器配置

        assert (
            self._model_config.num_layers % self._replica_config.num_pipeline_stages
            == 0
        )  # 确保层数能被流水线阶段数整除
        assert (
            self._model_config.embedding_dim % self._replica_config.tensor_parallel_size
            == 0
        )  # 确保嵌入维度能被张量并行大小整除

    @property
    def id(self) -> int:  # 对象唯一标识符的属性
        return self._id

    @property
    def num_layers(self) -> int:  # 模型层数的属性
        return self._model_config.num_layers

    @property
    def num_q_heads(self) -> int:  # 查询头数的属性
        return self._model_config.num_q_heads

    @property
    def num_kv_heads(self) -> int:  # 键值头数的属性
        return self._model_config.num_kv_heads

    @property
    def embedding_dim(self) -> int:  # 嵌入维度的属性
        return self._model_config.embedding_dim

    @property
    def mlp_hidden_dim(self) -> int:  # MLP隐藏层维度的属性
        return self._model_config.mlp_hidden_dim

    @property
    def use_gated_mlp(self) -> int:  # 是否使用Gated MLP的属性
        return self._model_config.use_gated_mlp

    @property
    def vocab_size(self) -> int:  # 词汇表大小的属性
        return self._model_config.vocab_size

    @property
    def num_pipeline_stages(self) -> int:  # 流水线阶段数的属性
        return self._replica_config.num_pipeline_stages

    @property
    def num_layers_per_pipeline_stage(self) -> int:  # 每个流水线阶段的层数
        return self._model_config.num_layers // self._replica_config.num_pipeline_stages

    @property
    def attention_head_dim(self) -> int:  # 注意力头的维度
        return self._model_config.embedding_dim // self._model_config.num_q_heads

    @property
    def q_heads_per_tensor_parallel_worker(self) -> int:  # 每个张量并行工作器的查询头数
        return (
            self._model_config.num_q_heads // self._replica_config.tensor_parallel_size
        )

    @property
    def kv_heads_per_tensor_parallel_worker(self) -> int:  # 每个张量并行工作器的键值头数
        return ceil(
            self._model_config.num_kv_heads / self._replica_config.tensor_parallel_size
        )

    @property
    def num_tensor_parallel_workers(self) -> int:  # 张量并行工作器的数量
        return self._replica_config.tensor_parallel_size

    @property
    def total_memory_gb(self) -> int:  # 总内存的属性
        return self._device_config.total_memory_gb

    @property
    def memory_margin_fraction(self) -> float:  # 内存余量百分比的属性
        return self._replica_config.memory_margin_fraction

    @property
    def max_request_tokens(self) -> int:  # 最大请求token数的属性
        return self._generator_config.max_tokens

    @property
    def per_device_flops(self) -> float:  # 每个设备的flops
        return self._device_config.fp16_tflops * 2**40

    def to_dict(self) -> dict:  # 将对象转换为字典的函数
        return {
            "id": self.id,
            "num_layers": self.num_layers,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "embedding_dim": self.embedding_dim,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "use_gated_mlp": self.use_gated_mlp,
            "vocab_size": self.vocab_size,
            "num_pipeline_stages": self.num_pipeline_stages,
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,
        }