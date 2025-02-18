from typing import List  # 导入List类型

from vidur.entities.batch import Batch  # 从vidur.entities.batch模块导入Batch类
from vidur.events import BaseEvent  # 从vidur.events模块导入BaseEvent类
from vidur.logger import init_logger  # 从vidur.logger模块导入init_logger函数
from vidur.metrics import MetricsStore  # 从vidur.metrics模块导入MetricsStore类
from vidur.scheduler import BaseGlobalScheduler  # 从vidur.scheduler模块导入BaseGlobalScheduler类
from vidur.types import EventType  # 从vidur.types模块导入EventType类

logger = init_logger(__name__)  # 初始化日志记录器，并使用当前模块名称

class BatchStageArrivalEvent(BaseEvent):  # 定义BatchStageArrivalEvent类，继承自BaseEvent
    def __init__(self, time: float, replica_id: int, stage_id: int, batch: Batch):  # 构造函数，初始化事件
        super().__init__(time, EventType.BATCH_STAGE_ARRIVAL)  # 调用父类构造函数，设置时间和事件类型

        self._replica_id = replica_id  # 保存副本ID
        self._stage_id = stage_id  # 保存阶段ID
        self._batch = batch  # 保存批次对象

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:  # 处理事件的方法
        from vidur.events.replica_stage_schedule_event import ReplicaStageScheduleEvent  # 延迟导入ReplicaStageScheduleEvent类

        scheduler.get_replica_stage_scheduler(
            self._replica_id, self._stage_id
        ).add_batch(self._batch)  # 获取调度器并添加批次

        return [
            ReplicaStageScheduleEvent(
                self.time,
                self._replica_id,
                self._stage_id,
            )
        ]  # 返回新的事件列表，包括ReplicaStageScheduleEvent

    def to_dict(self):  # 将事件对象转换为字典的方法
        return {
            "time": self.time,  # 事件时间
            "event_type": self.event_type,  # 事件类型
            "replica_id": self._replica_id,  # 副本ID
            "stage_id": self._stage_id,  # 阶段ID
            "batch_id": self._batch.id,  # 批次ID
        }