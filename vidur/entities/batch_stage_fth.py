from typing import List  # 导入 List 类型

from vidur.entities.base_entity import BaseEntity  # 从 vidur.entities.base_entity 中导入 BaseEntity
from vidur.entities.request import Request  # 从 vidur.entities.request 中导入 Request
from vidur.logger import init_logger  # 从 vidur.logger 中导入 init_logger

logger = init_logger(__name__)  # 初始化 logger 对象

# 一个装饰器，用于检查请求是否已经被调度
def check_scheduled(func):  # 定义 check_scheduled 装饰器
    def wrapper(self, *args, **kwargs):  # 定义内部函数 wrapper
        if not self._scheduled:  # 如果批次未被调度
            raise ValueError("Batch has not been scheduled yet")  # 抛出异常
        return func(self, *args, **kwargs)  # 否则调用被装饰的函数

    return wrapper  # 返回 wrapper 函数

class BatchStage(BaseEntity):  # 定义 BatchStage 类，继承自 BaseEntity
    def __init__(  # 定义构造函数
        self,
        batch_id: int,
        replica_id: int,
        pipeline_stage: int,
        execution_time: float,
        model_execution_time: float,
        requests: List[Request],
        num_tokens: List[Request],
    ) -> None:  # 参数列表中的类型注解

        self._id = BatchStage.generate_id()  # 生成并分配一个新 id

        self._requests = requests  # 保存请求列表
        self._num_tokens = num_tokens  # 保存 token 数量列表
        self._batch_id = batch_id  # 保存批次 ID
        self._replica_id = replica_id  # 保存副本 ID
        self._pipeline_stage = pipeline_stage  # 保存流水线阶段
        self._execution_time = execution_time  # 保存执行时间
        self._model_execution_time = model_execution_time  # 保存模型执行时间

        self._scheduled_at = None  # 初始化调度时间
        self._completed_at = None  # 初始化完成时间
        self._scheduled = False  # 初始化调度状态为未调度

    @property
    def num_tokens(self) -> List[int]:  # 定义 num_tokens 属性
        return self._num_tokens  # 返回 token 数量列表

    @property
    @check_scheduled  # 应用检查调度装饰器
    def scheduled_at(self) -> float:  # 定义 scheduled_at 属性
        return self._scheduled_at  # 返回调度时间

    @property
    @check_scheduled  # 应用检查调度装饰器
    def completed_at(self) -> float:  # 定义 completed_at 属性
        return self._completed_at  # 返回完成时间

    @property
    def execution_time(self) -> float:  # 定义 execution_time 属性
        return self._execution_time  # 返回执行时间

    @property
    def model_execution_time(self) -> float:  # 定义 model_execution_time 属性
        return self._model_execution_time  # 返回模型执行时间

    @property
    def pipeline_stage(self) -> int:  # 定义 pipeline_stage 属性
        return self._pipeline_stage  # 返回流水线阶段

    @property
    def request_ids(self) -> List[int]:  # 定义 request_ids 属性
        return [request.id for request in self._requests]  # 返回请求 ID 列表

    @property
    def requests(self) -> List[Request]:  # 定义 requests 属性
        return self._requests  # 返回请求列表

    @property
    def size(self) -> int:  # 定义 size 属性
        return len(self._requests)  # 返回请求列表的长度

    def on_schedule(  # 定义 on_schedule 方法
        self,
        time: float,
    ) -> None:  # 类型注解：无返回值
        self._scheduled_at = time  # 设置调度时间
        self._scheduled = True  # 设置调度状态为已调度

        for request in self._requests:  # 遍历请求列表
            request.on_batch_stage_schedule(time)  # 调用每个请求对象的 on_batch_stage_schedule 方法

    def on_stage_end(  # 定义 on_stage_end 方法
        self,
        time: float,
    ) -> None:  # 类型注解：无返回值
        assert (  # 断言执行时间是否匹配
            time == self._scheduled_at + self._execution_time
        ), f"{time} != {self._scheduled_at} + {self._execution_time}"  # 如果不匹配，抛出异常

        self._completed_at = time  # 设置完成时间

        for request in self._requests:  # 遍历请求列表
            request.on_batch_stage_end(  # 调用每个请求对象的 on_batch_stage_end 方法
                time, self._execution_time, self._model_execution_time
            )

    def to_dict(self) -> dict:  # 定义 to_dict 方法
        return {  # 返回一个字典，包含对象的各个属性
            "id": self._id,
            "size": self.size,
            "execution_time": self._execution_time,
            "model_execution_time": self._model_execution_time,
            "scheduled_at": self._scheduled_at,
            "completed_at": self._completed_at,
            "replica_id": self._replica_id,
            "batch_id": self._batch_id,
            "pipeline_stage": self._pipeline_stage,
            "scheduled": self._scheduled,
            "request_ids": self.request_ids,
            "num_tokens": self._num_tokens,
        }

    def to_chrome_trace(self, time: int) -> dict:  # 定义 to_chrome_trace 方法
        return {  # 返回一个字典，用于生成 Chrome Trace
            "name": f"{self.request_ids}",
            "ph": "X",
            "ts": (time - self._execution_time) * 1e6,
            "dur": self._execution_time * 1e6,
            "pid": self._replica_id,
            "tid": self._pipeline_stage,
            "args": {
                "batch_id": self._batch_id,
                "batch_size": self.size,
                "request_ids": self.request_ids,
                "num_tokens": self._num_tokens,
                # "requests": [request.to_dict() for request in self._requests], 该行注释掉的原因可能是为了减少数据量或敏感信息的泄露
            },
        }