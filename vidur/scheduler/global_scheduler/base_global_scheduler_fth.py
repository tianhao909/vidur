from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

# 从vidur模块中引入所需的类
from vidur.config import SimulationConfig
from vidur.entities import Replica, Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)

class BaseGlobalScheduler(ABC):  # 定义一个抽象基类(ABC) BaseGlobalScheduler
    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]):  # 初始化方法，接收配置和副本字典
        self._config = config  # 保存仿真配置
        self._replicas = replicas  # 保存副本信息

        self._num_replicas = len(self._replicas)  # 计算副本数量

        execution_time_predictor = ExecutionTimePredictorRegistry.get(  # 获取执行时间预测器实例
            config.execution_time_predictor_config.get_type(),  # 获取预测器类型
            predictor_config=config.execution_time_predictor_config,
            replica_config=config.cluster_config.replica_config,
            replica_scheduler_config=config.cluster_config.replica_scheduler_config,
            metrics_config=config.metrics_config,
        )
        self._replica_schedulers = {  # 创建副本调度器字典
            replica_id: ReplicaSchedulerRegistry.get(  # 获取每个副本的调度器实例
                config.cluster_config.replica_scheduler_config.get_type(),
                replica_config=config.cluster_config.replica_config,
                replica_scheduler_config=config.cluster_config.replica_scheduler_config,
                request_generator_config=config.request_generator_config,
                replica=replica,
                num_stages=replica.num_pipeline_stages,  # 获取副本的流水线阶段数
                execution_time_predictor=execution_time_predictor,
            )
            for replica_id, replica in replicas.items()  # 对每个副本进行迭代
        }
        self._request_queue = []  # 初始化请求队列为空

    def sort_requests(self) -> None:  # 定义方法用于对请求进行排序
        self._request_queue.sort(key=lambda request: request._arrived_at)  # 按请求到达时间排序

    def add_request(self, request: Request) -> None:  # 定义方法用于添加请求到队列
        self._request_queue.append(request)  # 向请求队列添加新的请求

    def get_replica_scheduler(self, replica_id: int):  # 获取指定副本的调度器
        return self._replica_schedulers[replica_id]  # 返回特定ID的副本调度器

    def get_replica_stage_scheduler(self, replica_id: int, stage_id: int):  # 获取特定阶段的调度器
        return self._replica_schedulers[replica_id].get_replica_stage_scheduler(stage_id)  # 返回特定阶段的调度器

    def is_empty(self) -> bool:  # 检查调度器和请求队列是否为空
        return len(self._request_queue) == 0 and all(  # 如果请求队列为空且所有副本调度器也为空
            replica_scheduler.is_empty()
            for replica_scheduler in self._replica_schedulers.values()
        )

    @abstractmethod
    def schedule(self) -> List[Tuple[int, Request]]:  # 抽象方法，需要子类实现具体的调度逻辑
        pass