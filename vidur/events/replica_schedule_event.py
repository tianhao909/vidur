from typing import List

from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class ReplicaScheduleEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int):
        super().__init__(time, EventType.REPLICA_SCHEDULE)

        self._replica_id = replica_id

        self._batches = []
        # print('fth生成Replica调度事件') 

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.batch_stage_arrival_event import BatchStageArrivalEvent

        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)
        # print('fth进入函数片段，replica_scheduler.on_schedule() /disk1/futianhao/software1/vidur/vidur/events/replica_schedule_event.py')   
        # >>fth def on_schedule() vidur/vidur/scheduler/replica_scheduler/base_replica_scheduler.py
        self._batches = replica_scheduler.on_schedule()
        # print('fth离开函数片段，replica_scheduler.on_schedule() /disk1/futianhao/software1/vidur/vidur/events/replica_schedule_event.py')   

        if not self._batches:
            return []

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_replica_schedule(
            self.time, self._replica_id, memory_usage_percent
        )

        for batch in self._batches:
            # print('>>fth 进入调度每个批处理任务 for batch in self._batches /disk1/futianhao/software1/vidur/vidur/events/replica_schedule_event.py')
            # >>fth def on_schedule() vidur/vidur/entities/batch.py
            batch.on_schedule(self.time)
            # print('>>fth 出去调度每个批处理任务 for batch in self._batches /disk1/futianhao/software1/vidur/vidur/events/replica_schedule_event.py')

        return [
            BatchStageArrivalEvent(
                self.time,
                self._replica_id,
                0,  # stage_id
                batch,
            )
            for batch in self._batches
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "batch_ids": [batch.id for batch in self._batches],
        }
