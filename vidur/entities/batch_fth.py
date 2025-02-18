from typing import List

from vidur.entities.base_entity import BaseEntity
from vidur.entities.request import Request
from vidur.logger import init_logger

logger = init_logger(__name__)  # 初始化日志记录器


# a decorator which checks if the request has been scheduled
# 一个装饰器，检查请求是否已被调度
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:  # 如果请求没有被调度
            raise ValueError("Batch has not been scheduled yet")  # 抛出一个错误
        return func(self, *args, **kwargs)

    return wrapper


def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:  # 如果请求没有完成
            raise ValueError("Batch has not been scheduled yet")  # 抛出一个错误
        return func(self, *args, **kwargs)

    return wrapper


class Batch(BaseEntity):
    def __init__(
        self,
        replica_id: int,  # 副本ID
        requests: List[Request],  # 请求列表
        num_tokens: List[int],  # 令牌数量列表
    ) -> None:
        self._id = Batch.generate_id()  # 生成批处理ID
        self._replica_id = replica_id  # 设置副本ID

        self._requests = requests  # 设置请求列表
        self._num_tokens = num_tokens  # 设置令牌数量列表
        self._total_num_tokens = sum(num_tokens)  # 计算总令牌数量
        self._num_prefill_tokens = sum(
            [
                (t if not r.is_prefill_complete else 0)  # 检查预填充是否完成
                for r, t in zip(self.requests, self._num_tokens)
            ]
        )

        # 将总令牌数量四舍五入到最近的8的倍数
        self._total_num_tokens_rounded = (self._total_num_tokens + 7) // 8 * 8

        self._scheduled_at = None  # 初始化计划时间为空
        self._completed_at = None  # 初始化完成时间为空
        self._scheduled = False  # 初始化为未调度状态
        self._completed = False  # 初始化为未完成状态

    @property
    def replica_id(self) -> int:
        return self._replica_id  # 返回副本ID

    @property
    def creation_time(self) -> float:
        return self._creation_time  # 返回创建时间

    @property
    def num_tokens(self) -> List[int]:
        return self._num_tokens  # 返回令牌数量列表

    @property
    def total_num_tokens(self) -> int:
        return self._total_num_tokens  # 返回总令牌数量

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens  # 返回预填充令牌数量

    @property
    def num_decode_tokens(self) -> int:
        return self.total_num_tokens - self.num_prefill_tokens  # 返回解码令牌数量

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at  # 返回计划时间

    @property
    @check_completed
    def completed_at(self) -> float:
        return self._completed_at  # 返回完成时间

    @property
    def completed(self) -> bool:
        return self._completed  # 返回是否完成

    @property
    def scheduled(self) -> bool:
        return self._scheduled  # 返回是否已调度

    @property
    def size(self) -> int:
        return len(self._requests)  # 返回请求列表的长度

    @property
    def requests(self) -> List[Request]:
        return self._requests  # 返回请求列表

    @property
    def request_ids(self) -> List[int]:
        return [request.id for request in self._requests]  # 返回请求ID列表

    @property
    def all_requests_completed(self) -> bool:
        return all([request.completed for request in self._requests])  # 返回所有请求是否完成

    def on_schedule(
        self,
        time: float,  # 当前时间
    ) -> None:
        self._scheduled_at = time  # 设置调度时间
        self._scheduled = True  # 将状态更新为已调度

        for request in self._requests:
            request.on_batch_schedule(time)  # 通知每个请求已调度

    def on_batch_end(self, time: float):
        self._completed = True  # 更新状态为已完成
        self._completed_at = time  # 设置完成时间

        for request, num_tokens in zip(self._requests, self._num_tokens):
            request.on_batch_end(time, num_tokens)  # 通知每个请求已完成

    @property
    def preempted_requests(self) -> List[Request]:
        return [request for request in self._requests if request.preempted]  # 返回被抢占的请求列表

    @property
    def completed_requests(self) -> List[Request]:
        return [request for request in self._requests if request.completed]  # 返回已完成的请求列表

    def to_dict(self) -> dict:
        return {
            "id": self._id,  # 批处理ID
            "size": self.size,  # 批处理大小
            "replica_id": self._replica_id,  # 副本ID
            "scheduled_at": self._scheduled_at,  # 调度时间
            "completed_at": self._completed_at,  # 完成时间
            "scheduled": self._scheduled,  # 是否已调度
            "request_ids": self.request_ids,  # 请求ID列表
            "num_tokens": self._num_tokens,  # 令牌数量列表
            "num_prefill_tokens": self.num_prefill_tokens,  # 预填充令牌数量
            "num_decode_tokens": self.num_decode_tokens,  # 解码令牌数量
        }