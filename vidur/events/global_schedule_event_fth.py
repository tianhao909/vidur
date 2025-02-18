from typing import List  # 导入List类型提示

from vidur.events import BaseEvent  # 从vidur.events中导入BaseEvent类
from vidur.logger import init_logger  # 从vidur.logger中导入init_logger函数
from vidur.metrics import MetricsStore  # 从vidur.metrics中导入MetricsStore类
from vidur.scheduler import BaseGlobalScheduler  # 从vidur.scheduler中导入BaseGlobalScheduler类
from vidur.types import EventType  # 从vidur.types中导入EventType类

logger = init_logger(__name__)  # 使用模块名初始化日志记录器

class GlobalScheduleEvent(BaseEvent):  # 定义GlobalScheduleEvent类，继承自BaseEvent
    def __init__(self, time: float):  # 定义初始化方法，接受一个时间参数
        super().__init__(time, EventType.GLOBAL_SCHEDULE)  # 调用父类初始化方法，设置时间和事件类型

        self._replica_set = []  # 初始化副本集合为空列表
        self._request_mapping = []  # 初始化请求映射为空列表

    def handle_event(  # 定义处理事件的方法
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore  # 方法接受调度器和指标存储作为参数
    ) -> List[BaseEvent]:  # 返回值是BaseEvent类型的列表
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent  # 导入ReplicaScheduleEvent类

        self._replica_set = set()  # 将副本集合重置为一个空集合
        self._request_mapping = scheduler.schedule()  # 调用调度器的schedule方法获取请求映射

        for replica_id, request in self._request_mapping:  # 遍历请求映射
            self._replica_set.add(replica_id)  # 将副本ID添加到副本集合中
            scheduler.get_replica_scheduler(replica_id).add_request(request)  # 将请求添加到对应的副本调度器

        return [  # 返回一个列表
            ReplicaScheduleEvent(self.time, replica_id)  # 列表包含创建的ReplicaScheduleEvent事件
            for replica_id in self._replica_set  # 对于每个在副本集合中的副本ID
        ]

    def to_dict(self):  # 定义将对象转换为字典的方法
        return {  # 返回一个字典
            "time": self.time,  # 包含时间
            "event_type": self.event_type,  # 事件类型
            "replica_set": self._replica_set,  # 副本集合
            "request_mapping": [  # 请求映射
                (replica_id, request.id)  # 每个映射中包含副本ID和请求ID
                for replica_id, request in self._request_mapping  # 遍历请求映射
            ],
        }