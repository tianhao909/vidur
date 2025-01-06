from vidur.scheduler.replica_scheduler.faster_transformer_replica_scheduler import (  # 从 faster_transformer_replica_scheduler 模块导入 FasterTransformerReplicaScheduler
    FasterTransformerReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.lightllm_replica_scheduler import (  # 从 lightllm_replica_scheduler 模块导入 LightLLMReplicaScheduler
    LightLLMReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.orca_replica_scheduler import (  # 从 orca_replica_scheduler 模块导入 OrcaReplicaScheduler
    OrcaReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.sarathi_replica_scheduler import (  # 从 sarathi_replica_scheduler 模块导入 SarathiReplicaScheduler
    SarathiReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.vllm_replica_scheduler import (  # 从 vllm_replica_scheduler 模块导入 VLLMReplicaScheduler
    VLLMReplicaScheduler,
)
from vidur.types import ReplicaSchedulerType  # 从 vidur.types 模块导入 ReplicaSchedulerType
from vidur.utils.base_registry import BaseRegistry  # 从 vidur.utils.base_registry 模块导入 BaseRegistry

class ReplicaSchedulerRegistry(BaseRegistry):  # 定义一个继承自 BaseRegistry 的类 ReplicaSchedulerRegistry
    pass  # 空语句，表示什么都不做

ReplicaSchedulerRegistry.register(  # 在 ReplicaSchedulerRegistry 中注册 FasterTransformerReplicaScheduler 与其类型
    ReplicaSchedulerType.FASTER_TRANSFORMER, FasterTransformerReplicaScheduler
)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.ORCA, OrcaReplicaScheduler)  # 在 ReplicaSchedulerRegistry 中注册 OrcaReplicaScheduler 与其类型
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.SARATHI, SarathiReplicaScheduler)  # 在 ReplicaSchedulerRegistry 中注册 SarathiReplicaScheduler 与其类型
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.VLLM, VLLMReplicaScheduler)  # 在 ReplicaSchedulerRegistry 中注册 VLLMReplicaScheduler 与其类型
ReplicaSchedulerRegistry.register(  # 在 ReplicaSchedulerRegistry 中注册 LightLLMReplicaScheduler 与其类型
    ReplicaSchedulerType.LIGHTLLM, LightLLMReplicaScheduler
)