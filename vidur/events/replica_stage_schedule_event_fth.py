from typing import List  # 从 typing 模块中导入 List 类型

from vidur.events import BaseEvent  # 从 vidur.events 模块中导入 BaseEvent 类
from vidur.logger import init_logger  # 从 vidur.logger 模块中导入 init_logger 函数
from vidur.metrics import MetricsStore  # 从 vidur.metrics 模块中导入 MetricsStore 类
from vidur.scheduler import BaseGlobalScheduler  # 从 vidur.scheduler 模块中导入 BaseGlobalScheduler 类
from vidur.types import EventType  # 从 vidur.types 模块中导入 EventType 类

logger = init_logger(__name__)  # 使用模块名称初始化日志记录器

class ReplicaStageScheduleEvent(BaseEvent):  # 定义一个继承自 BaseEvent 的类 ReplicaStageScheduleEvent
    def __init__(self, time: float, replica_id: int, stage_id: int):  # 初始化方法，接收时间、replica_id 和 stage_id
        super().__init__(time, EventType.REPLICA_STAGE_SCHEDULE)  # 调用父类的初始化方法，传入时间和事件类型

        self._replica_id = replica_id  # 设置 _replica_id 属性
        self._stage_id = stage_id  # 设置 _stage_id 属性

        self._batch = None  # 初始化 _batch 属性为 None
        self._batch_stage = None  # 初始化 _batch_stage 属性为 None
        self._is_last_stage = None  # 初始化 _is_last_stage 属性为 None

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore  # 定义 handle_event 方法，接收调度器和指标存储
    ) -> List[BaseEvent]:  # 返回值类型为一个包含 BaseEvent 的列表
        from vidur.events.batch_stage_end_event import BatchStageEndEvent  # 在方法内部导入 BatchStageEndEvent 类

        stage_scheduler = scheduler._replica_schedulers[  # 获取当前副本的阶段调度器
            self._replica_id
        ]._replica_stage_schedulers[self._stage_id]

        self._batch, self._batch_stage, execution_time = stage_scheduler.on_schedule()  # 获取批次、批次阶段和执行时间

        if not (self._batch and self._batch_stage):  # 如果批次或批次阶段不存在
            return []  # 返回一个空列表

        self._batch_stage.on_schedule(self.time)  # 调用批次阶段的 on_schedule 方法，传入当前时间
        metrics_store.on_replica_stage_schedule(  # 将调度信息记录到指标存储中
            self.time,
            self._replica_id,
            self._stage_id,
            self._batch_stage,
            execution_time,
        )

        self._is_last_stage = stage_scheduler.is_last_stage  # 设置 _is_last_stage 属性

        return [  # 返回一个包含 BatchStageEndEvent 的列表
            BatchStageEndEvent(
                self.time + self._batch_stage.execution_time,  # 事件结束时间为当前时间加上阶段执行时间
                self._replica_id,
                self._stage_id,
                self._is_last_stage,
                self._batch,
                self._batch_stage,
            ),
        ]

    def to_dict(self):  # 定义 to_dict 方法
        return {  # 返回一个字典，包含当前对象的属性
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "batch_id": self._batch.id if self._batch else None,  # 如果有批次就返回其 id，否则返回 None
            "batch_stage_id": self._batch_stage.id if self._batch_stage else None,  # 如果有批次阶段就返回其 id，否则返回 None
            "is_last_stage": self._is_last_stage,
        }