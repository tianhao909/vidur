from typing import Tuple

from vidur.entities.base_entity import BaseEntity  # 从vidur.entities.base_entity模块导入BaseEntity
from vidur.logger import init_logger  # 从vidur.logger模块导入init_logger函数

logger = init_logger(__name__)  # 初始化日志记录器

# 一个装饰器，用于检查请求是否已经被调度
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:  # 如果请求未被调度
            raise ValueError("Request has not been scheduled yet")  # 抛出异常
        return func(self, *args, **kwargs)  # 调用被装饰的函数

    return wrapper

# 一个装饰器，用于检查请求是否已经完成
def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:  # 如果请求未完成
            raise ValueError("Request has not been completed yet")  # 抛出异常
        return func(self, *args, **kwargs)  # 调用被装饰的函数

    return wrapper

class Request(BaseEntity):  # 继承自BaseEntity的Request类
    def __init__(
        self,
        arrived_at: float,  # 请求到达时间
        num_prefill_tokens: int,  # 预填充的标记数量
        num_decode_tokens: int,  # 需要解码的标记数量
        num_processed_tokens: int = 0,  # 已处理的标记数量，默认为0
    ):
        self._id = Request.generate_id()  # 生成请求的唯一标识符
        self._arrived_at = arrived_at  # 设置请求到达时间
        self._num_prefill_tokens = num_prefill_tokens  # 设置预填充的标记数量
        self._num_decode_tokens = num_decode_tokens  # 设置需要解码的标记数量
        self._num_processed_tokens = num_processed_tokens  # 设置已处理的标记数量

        # 初始化请求的各个时间属性
        self._scheduled_at = 0
        self._execution_time = 0
        self._model_execution_time = 0
        self._scheduling_delay = 0
        self._preempted_time = 0
        self._completed_at = 0
        self._prefill_completed_at = 0
        self._latest_stage_scheduled_at = 0
        self._latest_stage_completed_at = 0
        self._latest_iteration_scheduled_at = 0
        self._latest_iteration_completed_at = 0
        self._latest_iteration_scheduling_delay = 0

        # 初始化请求的状态标志
        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False

        self._num_restarts = 0  # 重启次数初始化为0

    @property
    def size(self) -> Tuple[int, int]:
        return (self._num_prefill_tokens, self._num_decode_tokens)  # 返回预填充和解码标记的数量

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at  # 返回调度时间

    @property
    @check_scheduled
    def latest_stage_scheduled_at(self) -> float:
        return self._latest_stage_scheduled_at  # 返回最近阶段调度时间

    @property
    @check_scheduled
    def latest_stage_completed_at(self) -> float:
        return self._latest_stage_completed_at  # 返回最近阶段完成时间

    @property
    @check_scheduled
    def latest_iteration_scheduled_at(self) -> float:
        return self._latest_iteration_scheduled_at  # 返回最近迭代调度时间

    @property
    @check_scheduled
    def latest_iteration_completed_at(self) -> float:
        return self._latest_iteration_completed_at  # 返回最近迭代完成时间

    @property
    @check_scheduled
    def latest_iteration_scheduling_delay(self) -> float:
        return self._latest_iteration_scheduling_delay  # 返回最近迭代调度延迟时间

    @property
    @check_scheduled
    def prefill_completed_at(self) -> float:
        return self._prefill_completed_at  # 返回预填充完成时间

    @property
    @check_scheduled
    def scheduling_delay(self) -> float:
        return self._scheduling_delay  # 返回调度延迟时间

    @property
    @check_scheduled
    def preempted_time(self) -> float:
        return self._preempted_time  # 返回抢占时间

    @property
    @check_completed
    def completed_at(self) -> float:
        return self._completed_at  # 返回完成时间

    @property
    @check_scheduled
    def e2e_time(self) -> float:
        return self._completed_at - self._arrived_at  # 返回端到端时间（完成时间减去到达时间）

    @property
    @check_scheduled
    def e2e_time_normalized(self) -> float:
        return self.e2e_time / self.num_decode_tokens  # 返回归一化的端到端时间

    @property
    @check_scheduled
    def execution_time(self) -> float:
        return self._execution_time  # 返回执行时间

    @property
    @check_scheduled
    def execution_time_normalized(self) -> float:
        return self._execution_time / self.num_decode_tokens  # 返回归一化的执行时间

    @property
    @check_scheduled
    def model_execution_time(self) -> float:
        return self._model_execution_time  # 返回模型执行时间

    @property
    @check_scheduled
    def model_execution_time_normalized(self) -> float:
        return self._model_execution_time / self.num_decode_tokens  # 返回归一化的模型执行时间

    @property
    def arrived_at(self) -> float:
        return self._arrived_at  # 返回到达时间

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens  # 返回预填充标记数量

    @property
    def num_decode_tokens(self) -> int:
        return self._num_decode_tokens  # 返回解码标记数量

    @property
    def pd_ratio(self) -> float:
        return self._num_prefill_tokens / self._num_decode_tokens  # 返回预填充与解码标记数量的比率

    @property
    def num_processed_tokens(self) -> int:
        return self._num_processed_tokens  # 返回已处理的标记数量

    @property
    def total_tokens(self) -> int:
        return self._num_prefill_tokens + self._num_decode_tokens  # 返回标记总数

    @property
    def num_processed_prefill_tokens(self) -> int:
        return min(self._num_processed_tokens, self._num_prefill_tokens)  # 返回已处理的预填充标记数量

    @property
    def num_processed_decode_tokens(self) -> int:
        return max(self._num_processed_tokens - self._num_prefill_tokens, 0)  # 返回已处理的解码标记数量

    @property
    def scheduled(self) -> bool:
        return self._scheduled  # 返回是否已调度

    @property
    def preempted(self) -> bool:
        return self._preempted and not self._completed  # 返回是否被抢占且未完成

    @property
    def completed(self) -> bool:
        return self._completed  # 返回是否已完成

    @property
    def num_restarts(self) -> int:
        return self._num_restarts  # 返回重启次数

    @property
    def is_prefill_complete(self) -> bool:
        return self._is_prefill_complete  # 返回预填充是否完成

    @property
    def has_started_decode(self) -> bool:
        return self._num_processed_tokens > self._num_prefill_tokens + 1  # 返回是否已开始解码

    def on_batch_schedule(
        self,
        time: float,  # 当前时间
    ) -> None:
        self._latest_iteration_scheduled_at = time  # 设置最近迭代调度时间
        self._latest_iteration_scheduling_delay = (
            time - self._latest_iteration_completed_at  # 计算最近迭代调度延迟时间
        )

        if self._scheduled:  # 如果已调度，则返回
            return

        if self._num_restarts > 0:  # 如果已重启，则设置调度标志并返回
            self._scheduled = True
            return

        # 设置调度时间和调度延迟
        self._scheduled_at = time
        self._scheduling_delay = time - self._arrived_at
        self._scheduled = True  # 设置为已调度

    def on_batch_end(
        self,
        time: float,  # 当前时间
        num_tokens_processed: int,  # 本批处理的标记数量
    ) -> None:
        self._num_processed_tokens += num_tokens_processed  # 更新已处理标记数量
        self._latest_iteration_completed_at = time  # 设置最近迭代完成时间

        assert self._num_processed_tokens <= self.total_tokens  # 确保已处理标记不超过总标记

        if self._num_processed_tokens == self._num_prefill_tokens:  # 如果预填充已完成
            self._is_prefill_complete = True  # 设置预填充完成标志
            self._num_processed_tokens += 1  # 处理一个解码标记

            if self._prefill_completed_at == 0:  # 如果预填充完成时间未设置
                self._prefill_completed_at = time  # 设置预填充完成时间

        if self._num_processed_tokens == self.total_tokens:  # 如果所有标记已处理完成
            self._completed_at = time  # 设置完成时间
            self._completed = True  # 设置为已完成
            logger.debug(f"Request {self._id} completed at {self._completed_at}")  # 记录调试信息

    def on_batch_stage_schedule(
        self,
        time: float,  # 当前时间
    ) -> None:
        self._latest_stage_scheduled_at = time  # 设置最近阶段调度时间
        if self._latest_stage_completed_at == 0:  # 如果没完成上一个阶段
            self._preempted_time = 0  # 设置抢占时间为0
        else:
            self._preempted_time += time - self._latest_stage_completed_at  # 更新抢占时间
        self._preempted = False  # 设置抢占标志为False

    def on_batch_stage_end(
        self,
        time: float,  # 当前时间
        execution_time: float,  # 本次执行时间
        model_execution_time: float,  # 本次模型执行时间
    ) -> None:
        self._execution_time += execution_time  # 更新总执行时间
        self._model_execution_time += model_execution_time  # 更新总模型执行时间
        self._latest_stage_completed_at = time  # 设置最近阶段完成时间
        self._preempted = True  # 设置为已抢占

    def to_dict(self) -> dict:
        return {
            "id": self._id,  # 请求标识符
            "arrived_at": self._arrived_at,  # 请求到达时间
            "execution_time": self._execution_time,  # 总执行时间
            "model_execution_time": self._model_execution_time,  # 总模型执行时间
            "scheduled_at": self._scheduled_at,  # 调度时间
            "scheduling_delay": self._scheduling_delay,  # 调度延迟
            "preempted_time": self._preempted_time,  # 抢占时间
            "completed_at": self._completed_at,  # 完成时间
            "num_prefill_tokens": self._num_prefill_tokens,  # 预填充标记数量
            "num_decode_tokens": self._num_decode_tokens,  # 解码标记数量
            "num_processed_tokens": self._num_processed_tokens,  # 已处理标记数量
            "scheduled": self._scheduled,  # 是否调度
            "preempted": self._preempted,  # 是否抢占
            "completed": self._completed,  # 是否完成
            "latest_stage_scheduled_at": self._latest_stage_scheduled_at,  # 最近阶段调度时间
            "latest_stage_completed_at": self._latest_stage_completed_at,  # 最近阶段完成时间
            "latest_iteration_scheduled_at": self._latest_iteration_scheduled_at,  # 最近迭代调度时间
            "latest_iteration_completed_at": self._latest_iteration_completed_at,  # 最近迭代完成时间
            "num_restarts": self._num_restarts,  # 重启次数
        }

    def restart(self):
        logger.debug(f"Restarting request {self._id}")  # 记录重启请求的调试信息

        # 重启时，可以一次性处理所有先前解码的标记（即可以预填充所有标记）
        total_tokens = self._num_prefill_tokens + self._num_decode_tokens  # 总标记数量
        self._num_prefill_tokens = self._num_processed_tokens  # 更新预填充标记数量
        self._num_decode_tokens = total_tokens - self._num_prefill_tokens  # 更新解码标记数量

        self._num_processed_tokens = 0  # 重置已处理标记数量
        self._scheduled = False  # 重置调度标志
        self._preempted = False  # 重置抢占标志
        self._completed = False  # 重置完成标志
        self._is_prefill_complete = False  # 重置预填充完成标志

        self._num_restarts += 1  # 增加重启次数