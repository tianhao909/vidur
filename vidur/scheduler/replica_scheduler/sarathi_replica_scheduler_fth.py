from math import ceil

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class SarathiReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # sarathi配置
        self._num_running_batches = 0  # 初始化正在运行的批次数量为0
        self._preempted_requests = []  # 被抢占的请求列表
        # 对于vLLM及其衍生品，我们只需设置一个宽松的最大批处理大小
        # 内存需求由调度器显式处理
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages  # 计算最大微批大小
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )  # 计算水印块数

    def _can_allocate_request(self, request: Request) -> bool:
        if request.id not in self._allocation_map:
            # 新请求
            num_required_blocks = ceil(
                request.num_prefill_tokens / self._config.block_size
            )  # 计算所需的块数
            return (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks  # 确认剩余块数是否大于等于水印块数
            )

        # vLLM至少需要一个可用块
        return self._config.num_blocks - self._num_allocated_blocks >= 1  # 检查是否有至少一个可用块

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            # 新请求
            num_required_blocks = ceil(
                request.num_prefill_tokens / self._config.block_size
            )  # 计算所需块数
            self.allocate(request.id, num_required_blocks)  # 分配所需块数
            return

        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size  # 计算已分配的代币数
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)  # 计算需要的代币数

        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}"  # 断言只需要0或1个代币

        if num_tokens_required == 0:
            return

        self.allocate(request.id, 1)  # 分配一个额外的块

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1  # 批处理结束，减少计数

        for request in batch.requests:
            if request.completed:
                self.free(request.id)  # 如果请求已完成，释放分配的资源
            else:
                self._preempted_requests.append(request)  # 否则，将请求添加到被抢占的请求列表中

    def _get_request_next_num_tokens(
        self, request: Request, batch_contains_prefill: bool, num_batch_tokens: int
    ) -> int:
        assert not request.completed  # 确保请求未完成

        if request.is_prefill_complete:
            return 1  # 如果预填充完成，则返回1
        
        # 计算当前请求（request）在下一次处理时需要分配的 token 数量（next_num_tokens）。
        # 它通过两者的最小值来限制处理的 token 数量，以确保既不会超过请求的剩余需求，也不会超出系统允许的 chunk 大小限制。
        next_num_tokens = min(
            request.num_prefill_tokens - request.num_processed_tokens,  # 请求中剩余未处理的 token 数
            self._config.chunk_size - num_batch_tokens                 # 当前 batch 中允许的剩余 token 数
        ) # 计算下一个块所需的token数

        next_num_tokens = max(0, next_num_tokens)  # 确保token数不为负

        return next_num_tokens

    def _get_next_batch(self) -> Batch:
        requests = []  # 存储请求的列表
        num_tokens = []  # 存储代币数量的列表
        skipped_requests = []  # 存储跳过的请求
        running_prefills = []  # 存储正在运行的预填充请求
        contains_prefill = False  # 标记批���理是否包含预填充
        num_batch_tokens = 0  # 批处理中的总代币数

        # 搶占的请求可能包含多个请求只完成了部分预填充，因此需要小心处理
        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                break  # 达到最大微批大小，停止

            request = self._preempted_requests.pop(0)  # 从抢占的请求列表中取出一个请求

            if not request.is_prefill_complete:
                running_prefills.append(request)  # 如果预填充未完成，将请求添加到正在运行的预填充列表中
                continue

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )  # 计算下一个请求所需的代币数

            if next_num_tokens == 0:
                skipped_requests.append(request)  # 如果不需要代币，将请求添加到跳过的请求列表中
                continue

            while not self._can_allocate_request(request):
                if self._preempted_requests:
                    victim_request = self._preempted_requests.pop(-1)  # 取出一个抢占的请求进行释放
                    victim_request.restart()
                    self.free(victim_request.id)
                    self._request_queue = [victim_request] + self._request_queue  # 重新放入请求队列
                else:
                    request.restart()
                    self.free(request.id)
                    self._request_queue = [request] + self._request_queue
                    break
            else:
                self._allocate_request(request)  # 分配请求
                assert request.is_prefill_complete
                num_batch_tokens += next_num_tokens  # 增加批处理中代币数量
                requests.append(request)  # 将请求添加到请求列表中
                num_tokens.append(next_num_tokens)  # 将代币数量添加到代币列表中

        for request in running_prefills:
            assert not request.is_prefill_complete

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )  # 计算下一个请求所需的代币数

            if next_num_tokens == 0:
                skipped_requests.append(request)  # 如果不需要代币，将请求添加到跳过的请求列表中
                continue

            contains_prefill = True  # 批处理包含预填充请求
            num_batch_tokens += next_num_tokens  # 增加批处理中代币数量
            requests.append(request)  # 将请求添加到请求列表中
            num_tokens.append(next_num_tokens)  # 将代币数量添加到代币列表中

        # 重新添加跳过的请求，但要确保将它们添加到队列的前面，以便保持FIFO顺序
        self._preempted_requests = skipped_requests + self._preempted_requests
        self._preempted_requests = sorted(
            self._preempted_requests, key=lambda req: req.arrived_at
        )  # 按到达时间对请求排序
        skipped_requests = []

        while self._request_queue:
            if len(self._allocation_map) == self._config.batch_size_cap:
                break  # 达到批处理大小上限，停止

            if len(requests) == self._max_micro_batch_size:
                break  # 达到最大微批大小，停止

            if not self._can_allocate_request(self._request_queue[0]):
                break  # 如果不能分配第一个请求，停止

            next_num_tokens = self._get_request_next_num_tokens(
                self._request_queue[0], contains_prefill, num_batch_tokens
            )  # 计算下一个请求所需的代币数

            if next_num_tokens == 0:
                break  # 如果不需要代币，停止

            request = self._request_queue.pop(0)  # 从请求队列中取出请求

            self._allocate_request(request)  # 分配请求

            # 所有新请求都将有预填充
            contains_prefill = True  # 批处理包含预填充请求
            num_batch_tokens += next_num_tokens  # 增加批处理中代币数量
            requests.append(request)  # 将请求添加到请求列表中
            num_tokens.append(next_num_tokens)  # 将代币数量添加到代币列表中

        if not requests:
            return  # 如果没有请求，返回空

        return Batch(self._replica_id, requests, num_tokens)  # 返回批次对象
