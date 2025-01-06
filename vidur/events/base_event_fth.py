from abc import ABC, abstractmethod
from typing import List

from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

class BaseEvent(ABC):  # 定义一个抽象基类 BaseEvent
    _id = 0  # 类变量 id，用于唯一标识事件

    def __init__(self, time: float, event_type: EventType):  # 初始化方法，接受时间和事件类型作为参数
        self._time = time  # 初始化事件发生时间
        self._id = BaseEvent.generate_id()  # 生成并分配唯一事件 ID
        self._event_type = event_type  # 初始化事件类型
        self._priority_number = self._get_priority_number()  # 初始化事件优先级

    @classmethod
    def generate_id(cls):  # 类方法用于生成唯一 ID
        cls._id += 1  # 递增类变量 id
        return cls._id  # 返回新的唯一 ID

    @property
    def id(self) -> int:  # 定义 id 属性
        return self._id  # 返回事件的唯一 ID

    @property
    def time(self):  # 定义 time 属性
        return self._time  # 返回事件发生时间

    @property
    def event_type(self):  # 定义 event_type 属性
        pass  # 抽象属性，在子类中实现

    @abstractmethod
    def handle_event(  # 定义抽象方法 handle_event
        self,
        current_time: float,
        scheduler: BaseGlobalScheduler,
        metrics_store: MetricsStore,
    ) -> List["BaseEvent"]:  # 返回事件列表，方法返回值类型为 BaseEvent 列表
        pass  # 留给子类实现特定的事件处理逻辑

    def _get_priority_number(self):  # 私有方法用于获取事件优先级
        return (self._time, self._id, self.event_type)  # 返回基于时间、ID 和事件类型的元组

    def __lt__(self, other):  # 重载小于运算符，用于比较事件
        if self._time == other._time:  # 如果时间相同
            if self._event_type == other._event_type:  # 如果事件类型也相同
                return self._id < other._id  # 比较事件 ID
            return self._event_type < other._event_type  # 否则比较事件类型
        else:
            return self._time < other._time  # 如果时间不同，比较事件时间

    def __eq__(self, other):  # 重载等于运算符，用于比较事件
        return (
            self._time == other._time  # 比较事件时间
            and self._event_type == other._event_type  # 比较事件类型
            and self._id == other._id  # 比较事件 ID
        )

    def __str__(self) -> str:  # 重载字符串转换方法
        class_name = self.__class__.__name__  # 获取类名
        return f"{class_name}({str(self.to_dict())})"  # 返回类名和事件信息字符串

    def to_dict(self):  # 转换事件信息为字典
        return {"time": self.time, "event_type": self.event_type}  # 返回包含时间和事件类型的字典

    def to_chrome_trace(self) -> dict:  # 用于转换为 Chrome trace 格式
        return None  # 默认返回 None，留给子类实现