from vidur.entities.batch import Batch
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class OrcaReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类的初始化方法

        self._preempted_requests = []  # 初始化被抢占的请求列表
        self._num_running_batches = 0  # 初始化正在运行的批次计数

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1  # 批次结束，减少正在运行的批次计数

        for request in batch.requests:  # 遍历批次中的所有请求
            if request.completed:  # 如果请求已完成
                self.free(request.id)  # 释放资源
            else:
                self._preempted_requests.append(request)  # 否则，将请求加入被抢占的请求列表

    def _get_next_batch(self) -> Batch:
        requests = []  # 用于存储下一个批次的请求
        num_tokens = []  # 用于存储每个请求的令牌数量

        # 所有被抢占的请求已完成预填充
        while self._preempted_requests:
            if len(requests) == self._max_batch_size:  # 如果请求数量达到最大批次大小
                break  # 退出循环

            request = self._preempted_requests.pop(0)  # 取第一个被抢占的请求
            next_num_tokens = self._get_request_next_num_tokens(request)  # 获取请求的下一个令牌数量
            requests.append(request)  # 将请求加入请求列表
            num_tokens.append(next_num_tokens)  # 将令牌数量加入列表

        while self._request_queue:
            if len(requests) == self._max_batch_size:  # 如果请求数量达到最大批次大小
                break  # 退出循环

            if not self.can_allocate(self._max_blocks_per_sequence):  # 如果不能分配最大块数
                break  # 退出循环

            request = self._request_queue.pop(0)  # 取出请求队列的第一个请求

            self.allocate(request.id, self._max_blocks_per_sequence)  # 为请求分配资源
            next_num_tokens = self._get_request_next_num_tokens(request)  # 获取请求的下一个令牌数量
            requests.append(request)  # 将请求加入请求列表
            num_tokens.append(next_num_tokens)  # 将令牌数量加入列表

        if not requests:  # 如果请求列表为空
            return  # 返回 None

        return Batch(self._replica_id, requests, num_tokens)  # 返回新的批次对象