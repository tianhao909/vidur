from typing import List

from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)

class ReplicaScheduleEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int):
        super().__init__(time, EventType.REPLICA_SCHEDULE)  # 初始化父类

        self._replica_id = replica_id  # 保存副本ID

        self._batches = []  # 初始化批处理列表


    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.batch_stage_arrival_event import BatchStageArrivalEvent  # 导入BatchStageArrivalEvent类

        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)  # 获取副本调度器 return self._replica_schedulers[replica_id]

        self._batches = replica_scheduler.on_schedule()  # 调度批处理任务


        if not self._batches:
            return []  # 如果没有批处理任务，返回空列表

        memory_usage_percent = replica_scheduler.memory_usage_percent  # 获取内存使用百分比
        metrics_store.on_replica_schedule(
            self.time, self._replica_id, memory_usage_percent
        )  # 储存调度事件的度量数据

        for batch in self._batches:
            # print('>>fth 进入调度每个批处理任务')
            batch.on_schedule(self.time)  # 调度每个批处理任务
            # print('>>fth 进入调度每个批处理任务')

        return [
            BatchStageArrivalEvent(
                self.time,
                self._replica_id,
                0,  # stage_id
                batch,
            )
            for batch in self._batches  # 为每个批处理任务返回BatchStageArrivalEvent
        ]

    def to_dict(self):
        return {
            "time": self.time,  # 事件时间
            "event_type": self.event_type,  # 事件类型
            "replica_id": self._replica_id,  # 副本ID
            "batch_ids": [batch.id for batch in self._batches],  # 批处理ID列表
        }