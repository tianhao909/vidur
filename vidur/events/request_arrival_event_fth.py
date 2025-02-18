from typing import List  # 导入List类型，用于类型注解。

from vidur.entities import Request  # 从vidur.entities模块导入Request类。
from vidur.events.base_event import BaseEvent  # 从vidur.events.base_event模块导入BaseEvent类。
from vidur.logger import init_logger  # 从vidur.logger模块导入init_logger函数。
from vidur.metrics import MetricsStore  # 从vidur.metrics模块导入MetricsStore类。
from vidur.scheduler import BaseGlobalScheduler  # 从vidur.scheduler模块导入BaseGlobalScheduler类。
from vidur.types import EventType  # 从vidur.types模块导入EventType类。

logger = init_logger(__name__)  # 初始化日志记录器，使用当前模块的名称。

class RequestArrivalEvent(BaseEvent):  # 定义RequestArrivalEvent类，继承自BaseEvent。
    def __init__(self, time: float, request: Request) -> None:  # 初始化函数，接收时间和请求参数。
        super().__init__(time, EventType.REQUEST_ARRIVAL)  # 调用父类的初始化函数，设置时间和事件类型。

        self._request = request  # 将请求参数保存为实例变量。

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:  # 定义处理事件方法，接收调度器和指标存储，并返回BaseEvent列表。
        from vidur.events.global_schedule_event import GlobalScheduleEvent  # 在函数内部导入GlobalScheduleEvent类。

        logger.debug(f"Request: {self._request.id} arrived at {self.time}")  # 记录请求到达的调试信息。
        scheduler.add_request(self._request)  # 向调度器添加请求。
        metrics_store.on_request_arrival(self.time, self._request)  # 更新指标存储，记录请求到达时间和请求信息。
        return [GlobalScheduleEvent(self.time)]  # 返回包含一个GlobalScheduleEvent的列表。

    def to_dict(self) -> dict:  # 定义方法，将对象转换为字典。
        return {
            "time": self.time,  # 将时间转换为字典中的一个项。
            "event_type": self.event_type,  # 将事件类型转换为字典中的一个项。
            "request": self._request.id,  # 将请求ID转换为字典中的一个项。
        }