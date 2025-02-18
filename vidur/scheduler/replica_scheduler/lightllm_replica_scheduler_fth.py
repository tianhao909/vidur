from typing import List, Tuple

import numpy as np

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class LightLLMReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._preempted_requests: List[Request] = []  # 被抢占的请求列表
        self._num_running_batches = 0  # 当前正在运行的批次数
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages  # 最大微批量大小
        assert (
            self._config.block_size == 1
        ), "LightLLM 调度程序仅支持块大小为 1。"  # 确保块大小为1
        assert (
            self._num_stages == 1
        ), "LightLLM 调度程序不支持流水线并行。"  # 确保没有使用流水线并行

        self._cache_len_list = []  # 缓存长度列表
        self._num_waiting_iters = 0  # 等待迭代次数

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1  # 批处理结束，减少正在运行的批次数

        for request in batch.requests:
            if request.completed:
                self.free(request.id)  # 如果请求已完成，释放其资源
            else:
                self._preempted_requests.append(request)  # 否则将其加入抢占列表

    def _get_tuple_tokens(self, request: Request) -> Tuple[int, int]:
        if request.scheduled:
            num_processed_tokens = request.num_processed_tokens  # 获取已处理的令牌数
            remaining_tokens = (
                request.num_decode_tokens - request.num_processed_decode_tokens - 1
            )  # 计算剩余未处理的解码令牌数
        else:
            num_processed_tokens = request.num_prefill_tokens + 1  # 获取预填充的令牌数
            remaining_tokens = request.num_decode_tokens - 1 - 1  # 计算剩余未处理的解码令牌数

        remaining_tokens = max(0, remaining_tokens)  # 保证剩余令牌数不为负数

        return (num_processed_tokens, remaining_tokens)  # 返回处理和剩余令牌数的元组

    def _can_allocate_request(self, request: Request) -> bool:
        # 我不知道这段代码具体做什么
        self.cache_len_list.append(self._get_tuple_tokens(request))  # 将请求的令牌信息加入缓存列表
        self.cache_len_list.sort(key=lambda x: -x[1])  # 按照剩余令牌数排序

        left_out_len_array = np.array([e[1] for e in self.cache_len_list])  # 创建剩余令牌数的数组
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])  # 创建已处理令牌数的数组
        cum_run_len_array = np.cumsum(has_run_len_array)  # 创建累积已处理令牌数的数组
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)  # 创建大小数组

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()  # 计算需要的最大令牌数量

        return need_max_token_num < self._config.num_blocks  # 判断是否可以分配请求

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            self.allocate(request.id, request.num_prefill_tokens)  # 分配预填充阶段的请求
            return

        self.allocate(request.id, 1)  # 分配解码阶段的请求

    def _get_prefill_batch(self) -> Batch:
        requests = []  # 请求列表
        num_tokens = []  # 令牌数列表
        num_batch_tokens = 0  # 批处理令牌数

        self.cache_len_list = [
            self._get_tuple_tokens(request) for request in self._preempted_requests
        ]  # 更新缓存列表为已抢占请求的令牌信息

        while self._request_queue:
            request = self._request_queue[0]  # 获取队列中的第一个请求

            next_num_tokens = self._get_request_next_num_tokens(request)  # 获取请求的下一个令牌数

            if num_batch_tokens + next_num_tokens > self._config.max_tokens_in_batch:
                break  # 如果当前批处理令牌超过最大值，终止循环

            if len(self._allocation_map) == self._config.batch_size_cap:
                break  # 如果达到批量大小上限，终止循环

            if len(requests) == self._max_micro_batch_size:
                break  # 如果达到微批量大小上限，终止循环

            if not self._can_allocate_request(request):
                break  # 如果无法分配请求，终止循环

            request = self._request_queue.pop(0)  # 从队列中移除请求

            self._allocate_request(request)  # 分配请求
            requests.append(request)  # 将请求加入请求列表
            num_tokens.append(next_num_tokens)  # 将令牌数加入令牌数列表
            num_batch_tokens += next_num_tokens  # 增加批处理令牌数

        if requests:
            return Batch(self._replica_id, requests, num_tokens)  # 如果有请求，创建批处理并返回

        return

    def _get_decode_batch(self) -> Batch:
        requests = []  # 请求列表
        num_tokens = []  # 令牌数列表

        # 所有 preempted_requests 都完成了预填充
        while self._preempted_requests:
            assert len(requests) < self._max_micro_batch_size  # 确保未超过微批量大小

            request = self._preempted_requests.pop(0)  # 从抢占列表中移除请求

            assert self.can_allocate(1)  # 确保可以分配一个令牌
            self._allocate_request(request)  # 分配请求

            next_num_tokens = self._get_request_next_num_tokens(request)  # 获取请求的下一个令牌数
            requests.append(request)  # 将请求加入请求列表
            num_tokens.append(next_num_tokens)  # 将令牌数加入令牌数列表

        if not requests:
            return

        return Batch(self._replica_id, requests, num_tokens)  # 返回解码批处理

    def _can_decode(self):
        return self.can_allocate(len(self._preempted_requests))  # 判断是否能解码

    def _get_next_batch(self) -> Batch:
        if not self._preempted_requests:
            batch = self._get_prefill_batch()  # 获取预填充批处理
            if batch:
                self._num_waiting_iters = 0  # 重置等待迭代次数
            return batch

        if self._num_waiting_iters >= self._config.max_waiting_iters:
            self._num_waiting_iters = 0  # 重置等待迭代次数
            batch = self._get_prefill_batch()  # 获取预填充批处理
            if batch:
                return batch

        if self._can_decode():
            self._num_waiting_iters += 1  # 增加等待迭代次数
            return self._get_decode_batch()  # 获取解码批处理
        else:
            raise RuntimeError("OOM 处理尚未实现")  # 引发运行时错误，提示 OOM 处理未实现