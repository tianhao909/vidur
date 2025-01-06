from math import ceil

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)

class SarathiReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # sarathi 配置
        self._num_running_batches = 0  # 当前运行的批次数
        self._preempted_requests = []  # 被抢占的请求列表
        # 对于 vLLM 及其派生应用，只需设置一个宽松的最大批次大小
        # 内存要求由调度器明确处理
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages  # 最大微批次大小
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks  # 水印块数
        )

    def _can_allocate_request(self, request: Request) -> bool:
        if request.id not in self._allocation_map:
            # 新请求
            num_required_blocks = ceil(
                request.num_prefill_tokens / self._config.block_size  # 计算所需的块数
            )
            return (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks  # 确认是否可以分配
            )

        # vllm 至少需要一个可用的块
        return self._config.num_blocks - self._num_allocated_blocks >= 1

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            # 新请求
            num_required_blocks = ceil(
                request.num_prefill_tokens / self._config.block_size  # 计算所需的块数
            )
            self.allocate(request.id, num_required_blocks)  # 分配请求
            return

        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size  # 预留的token数
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)  # 所需的token数

        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}"  # 确保分配的token合理

        if num_tokens_required == 0:
            return

        self.allocate(request.id, 1)  # 分配一个token

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1  # 减少当前运行的批次数

        for request in batch.requests:
            if request.completed:
                self.free(request.id)  # 释放已完成请求的资源
            else:
                self._preempted_requests.append(request)  # 将未完成的请求添加到被抢占列表

    def _get_request_next_num_tokens(
        self, request: Request, batch_contains_prefill: bool, num_batch_tokens: int
    ) -> int:
        assert not request.completed  # 确保请求未完成

        if request.is_prefill_complete:
            return 1  # 如果预填充完成，则返回1个token

        next_num_tokens = min(
            request.num_prefill_tokens - request.num_processed_tokens,
            self._config.chunk_size - num_batch_tokens,  # 计算下一个token数
        )

        next_num_tokens = max(0, next_num_tokens)  # 确保token数不小于0

        return next_num_tokens

    def _get_next_batch(self) -> Batch:
        requests = []
        num_tokens = []
        skipped_requests = []
        running_prefills = []
        contains_prefill = False
        num_batch_tokens = 0

        # 被抢占的请求可能包含多个部分预填充完成的请求，所以需要小心处理
        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                break

            request = self._preempted_requests.pop(0)

            if not request.is_prefill_complete:
                running_prefills.append(request)
                continue

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                skipped_requests.append(request)
                continue

            while not self._can_allocate_request(request):
                if self._preempted_requests:
                    victim_request = self._preempted_requests.pop(-1)
                    victim_request.restart()
                    self.free(victim_request.id)
                    self._request_queue = [victim_request] + self._request_queue
                else:
                    request.restart()
                    self.free(request.id)
                    self._request_queue = [request] + self._request_queue
                    break
            else:
                self._allocate_request(request)
                assert request.is_prefill_complete
                num_batch_tokens += next_num_tokens
                requests.append(request)
                num_tokens.append(next_num_tokens)

        for request in running_prefills:
            assert not request.is_prefill_complete

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                skipped_requests.append(request)
                continue

            contains_prefill = True
            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)

        # 重新添加跳过的请求，但确保将它们添加到队列的前面，以便首先调度并维护FIFO排序
        self._preempted_requests = skipped_requests + self._preempted_requests
        self._preempted_requests = sorted(
            self._preempted_requests, key=lambda req: req.arrived_at
        )
        skipped_requests = []

        while self._request_queue:
            if len(self._allocation_map) == self._config.batch_size_cap:
                break

            if len(requests) == self._max_micro_batch_size:
                break

            if not self._can_allocate_request(self._request_queue[0]):
                break

            next_num_tokens = self._get_request_next_num_tokens(
                self._request_queue[0], contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                break

            request = self._request_queue.pop(0)

            self._allocate_request(request)

            # 所有新请求都将有一次预填充
            contains_prefill = True
            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)

        if not requests:
            return

        return Batch(self._replica_id, requests, num_tokens)