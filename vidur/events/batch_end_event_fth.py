from typing import List  # 导入 List 类型

from vidur.entities import Batch  # 从 vidur.entities 模块导入 Batch 类
from vidur.events import BaseEvent  # 从 vidur.events 模块导入 BaseEvent 类
from vidur.logger import init_logger  # 从 vidur.logger 模块导入 init_logger 函数
from vidur.metrics import MetricsStore  # 从 vidur.metrics 模块导入 MetricsStore 类
from vidur.scheduler import BaseGlobalScheduler  # 从 vidur.scheduler 模块导入 BaseGlobalScheduler 类
from vidur.types import EventType  # 从 vidur.types 模块导入 EventType 类

logger = init_logger(__name__)  # 初始化日志记录器

class BatchEndEvent(BaseEvent):  # 定义 BatchEndEvent 类，继承自 BaseEvent
    def __init__(self, time: float, replica_id: int, batch: Batch):  # 定义构造方法
        super().__init__(time, EventType.BATCH_END)  # 调用父类构造方法，传递时间和事件类型

        self._replica_id = replica_id  # 初始化 _replica_id 属性
        self._batch = batch  # 初始化 _batch 属性

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore  # 定义 handle_event 方法，并传递调度器和指标存储对象
    ) -> List[BaseEvent]:  
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent  # 从 vidur.events.replica_schedule_event 模块导入 ReplicaScheduleEvent 类

        self._batch.on_batch_end(self.time)  # 调用Batch对象的 on_batch_end 方法
        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)  # 获取副本调度器
        replica_scheduler.on_batch_end(self._batch)  # 调用副本调度器的 on_batch_end 方法

        memory_usage_percent = replica_scheduler.memory_usage_percent  # 获取内存使用百分比
        metrics_store.on_batch_end(
            self.time, self._batch, self._replica_id, memory_usage_percent  # 保存批次结束的指标
        )

        return [ReplicaScheduleEvent(self.time, self._replica_id)]  # 返回新的 ReplicaScheduleEvent 事件列表

    def to_dict(self):  # 定义 to_dict 方法，将对象转换为字典
        return {
            "time": self.time,  # 时间
            "event_type": self.event_type,  # 事件类型
            "batch_id": self._batch.id,  # 批次ID
        }