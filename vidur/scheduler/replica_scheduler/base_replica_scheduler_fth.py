from abc import ABC, abstractmethod
from typing import List

from vidur.config import (
    BaseReplicaSchedulerConfig,
    BaseRequestGeneratorConfig,
    ReplicaConfig,
)
from vidur.entities import Batch, Replica, Request
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.logger import init_logger
from vidur.scheduler.replica_stage_scheduler import ReplicaStageScheduler
from vidur.scheduler.utils.memory_planner import MemoryPlanner

logger = init_logger(__name__)  # 初始化日志记录器

class BaseReplicaScheduler(ABC):  # 定义抽象基类BaseReplicaScheduler
    def __init__(
        self,
        replica_config: ReplicaConfig,  # 初始化参数: 副本配置
        replica_scheduler_config: BaseReplicaSchedulerConfig,  # 初始化参数: 副本调度器配置
        request_generator_config: BaseRequestGeneratorConfig,  # 初始化参数: 请求生成器配置
        replica: Replica,  # 初始化参数: 副本实体
        num_stages: int,  # 初始化参数: 阶段数量
        execution_time_predictor: BaseExecutionTimePredictor,  # 初始化参数: 执行时间预测器
    ) -> None:
        self._config = replica_scheduler_config  # 保存副本调度器配置
        self._replica_config = replica_config  # 保存副本配置
        self._request_generator_config = request_generator_config  # 保存请求生成器配置
        self._replica_id = replica.id  # 保存副本ID
        self._num_stages = num_stages  # 保存阶段数量

        self._max_blocks_per_sequence = (  # 计算每个序列的最大块数
            self._request_generator_config.max_tokens // self._config.block_size
        )

        memory_planner = MemoryPlanner(self._replica_config, replica)  # 创建内存规划器实例

        if not self._config.num_blocks:  # 如果配置中没有设定块数
            self._config.num_blocks = (  # 计算并设定可用的总块数
                self._max_blocks_per_sequence * memory_planner.get_max_request_slots()
            )
        self._max_batch_size = min(  # 计算最大批处理大小
            memory_planner.get_max_batch_size(),
            self._config.batch_size_cap,
        )

        logger.debug(  # 记录最大批处理大小
            f"Obtained max batch size of {self._max_batch_size} for replica {self._replica_id}"
        )

        self._request_queue = []  # 初始化请求队列
        self._num_allocated_blocks = 0  # 初始化已分配块数
        self._allocation_map = {}  # 初始化分配映射

        self._replica_stage_schedulers = {  # 初始化副本阶段调度器
            stage_id: ReplicaStageScheduler(
                replica.id,
                stage_id,
                stage_id == num_stages - 1,
                execution_time_predictor,
            )
            for stage_id in range(num_stages)
        }

    @property
    def num_pending_requests(self) -> int:  # 返回挂起请求的数量
        return len(self._request_queue)

    @property
    def replica_id(self) -> int:  # 返回副本ID
        return self._replica_id

    @property
    def num_allocated_blocks(self) -> int:  # 返回已分配块数
        return self._num_allocated_blocks

    @property
    def memory_usage_percent(self) -> int:  # 返回内存使用百分比
        return (self._num_allocated_blocks * 100) / self._config.num_blocks

    def is_empty(self) -> bool:  # 检查调度器是否为空
        return (
            self.num_pending_requests == 0
            and len(self._allocation_map) == 0
            and all(
                stage_scheduler.is_empty()
                for stage_scheduler in self._replica_stage_schedulers.values()
            )
        )

    def _get_request_next_num_tokens(self, request: Request) -> int:  # 获取该request的下一个阶段的token处理数量； 获取请求的下一个token数量
        assert not request.completed

        if request.is_prefill_complete:  # 如果请求的预填充完成
            return 1  # 返回1

        return request.num_prefill_tokens  # 否则返回请求的预填充token数

    def add_request(self, request: Request) -> None:  # 添加请求到队列
        self._request_queue.append(request)

    def get_replica_stage_scheduler(self, stage_id: int):  # 获取特定阶段的调度器
        return self._replica_stage_schedulers[stage_id]

    def can_allocate(self, num_blocks: int) -> bool:  # 检查是否可以分配指定块数
        return self._config.num_blocks - self._num_allocated_blocks >= num_blocks

    def allocate(self, request_id: int, num_blocks: int) -> None:  # 分配指定块数给某请求
        self._num_allocated_blocks += num_blocks
        if request_id not in self._allocation_map:
            self._allocation_map[request_id] = num_blocks
        else:
            self._allocation_map[request_id] += num_blocks

        assert self._num_allocated_blocks <= self._config.num_blocks  # 检查不超过总块数

    def free(self, *request_ids: List[int]) -> None:  # 释放指定请求的块
        for request_id in request_ids:
            num_blocks = self._allocation_map.pop(request_id)
            self._num_allocated_blocks -= num_blocks

        assert self._num_allocated_blocks >= 0  # 检查不出现负值

    def free_batch(self, batch: Batch) -> None:  # 释放批处理中所有请求的块
        self.free(*batch.request_ids)

    @abstractmethod
    def on_batch_end(self, batch: Batch) -> None:  # 抽象方法: 在批处理结束时调用
        pass

    @abstractmethod
    def _get_next_batch(self) -> Batch:  # 抽象方法: 获取下一个批处理
        pass

    def on_schedule(self) -> List[Batch]:  # 调度批处理
        scheduled_batches = []
        while self._num_running_batches < self._num_stages:  # 当运行中的批处理少于阶段数时
            batch = self._get_next_batch()  # 获取下一个批处理
            if not batch:  # 如果没有批处理
                break  # 跳出循环
            scheduled_batches.append(batch)
            self._num_running_batches += 1  # 增加运行中批处理计数
        return scheduled_batches  # 返回已调度批处理列表