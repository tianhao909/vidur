from math import ceil  # 导入ceil函数用于向上取整
from typing import List  # 导入List类型用于类型注解

from vidur.entities.batch import Batch, Request  # 从vidur.entities.batch导入Batch和Request类
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (  # 从vidur.scheduler.replica_scheduler.base_replica_scheduler导入BaseReplicaScheduler类
    BaseReplicaScheduler,
)


class VLLMReplicaScheduler(BaseReplicaScheduler):  # 定义VLLMReplicaScheduler类，继承自BaseReplicaScheduler
    def __init__(self, *args, **kwargs):  # 初始化方法，接收多个参数
        super().__init__(*args, **kwargs)  # 调用父类的初始化方法

        self._preempted_requests: List[Request] = []  # 定义一个列表用于存储被中断的请求
        self._num_running_batches = 0  # 记录当前正在运行的批次数量
        # 对于vLLM和其衍生版本, 我们只需要设置一个宽松的最大批大小
        # 内存需求由调度器显式处理
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages  # 计算最大微批大小
        self._watermark_blocks = int(  # 计算水位块数量
            self._config.watermark_blocks_fraction * self._config.num_blocks  # 依据配置参数水位块分数和总块数量计算
        )

    def on_batch_end(self, batch: Batch) -> None:  # 批处理结束时的回调方法
        self._num_running_batches -= 1  # 减少正在运行的批次数量

        for request in batch.requests:  # 遍历批次中的每个请求
            if request.completed:  # 如果请求已完成
                self.free(request.id)  # 释放该请求所占用的资源
            else:
                self._preempted_requests.append(request)  # 将未完成的请求添加到被中断请求列表中

    def _can_allocate_request(self, request: Request) -> bool:  # 检查是否可以分配请求
        if request.id not in self._allocation_map:  # 如果请求ID不在分配映射中
            # 新请求
            num_required_blocks = ceil(  # 计算所需的块数量（用ceil向上取整）
                (request.num_prefill_tokens) / self._config.block_size  # 根据请求的预填充token数量和块大小计算
            )
            return (  # 判断是否有足够的块可用
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks  # 可用块需大于等于水位块数量
            )

        # vllm需要至少有一个块可用
        return self._config.num_blocks - self._num_allocated_blocks >= 1  # 如果已经在分配中，需要至少一个可用块

    def _allocate_request(self, request: Request) -> None:  # 分配请求所需资源
        if request.id not in self._allocation_map:  # 如果请求ID不在分配映射中
            # 新请求
            num_required_blocks = ceil(  # 计算所需的块数量
                (request.num_prefill_tokens) / self._config.block_size  # 根据预填充token数量和块大小
            )
            self.allocate(request.id, num_required_blocks)  # 分配所需数量的块
            return

        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size  # 已保留的tokens数量
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)  # 计算需要的tokens数量
        assert (
            num_tokens_required == 0 or num_tokens_required == 1  # 确保需要的tokens数量为0或1
        ), f"num_tokens_required: {num_tokens_required}"

        if num_tokens_required == 0:  # 如果不需要额外的tokens
            return

        self.allocate(request.id, 1)  # 分配一个块

    def _get_next_batch(self) -> Batch:  # 获取下一个批次
        requests = []  # 初始化请求列表
        num_tokens = []  # 初始化tokens数量列表
        num_batch_tokens = 0  # 初始化批处理tokens数量

        while self._request_queue:  # 当请求队列不为空时
            request = self._request_queue[0]  # 获取队列中的第一个请求

            next_num_tokens = self._get_request_next_num_tokens(request)  # 获取请求的下一个tokens数量

            if not self._can_allocate_request(request):  # 如果不能分配请求
                break

            new_num_tokens = num_tokens + [next_num_tokens]  # 计算新的tokens数量
            new_num_batch_tokens = len(new_num_tokens) * max(new_num_tokens)  # 计算新的批处理tokens数量
            if new_num_batch_tokens > self._config.max_tokens_in_batch:  # 如果超出最大批处理tokens数量
                break

            if len(self._allocation_map) == self._config.batch_size_cap:  # 如果达到最大批大小
                break

            if len(requests) == self._max_micro_batch_size:  # 如果达到最大微批大小
                break

            request = self._request_queue.pop(0)  # 从队列中移除该请求

            self._allocate_request(request)  # 分配请求所需资源
            requests.append(request)  # 将请求添加到请求列表
            num_tokens.append(next_num_tokens)  # 将tokens数量添加到列表
            num_batch_tokens += next_num_tokens  # 更新批处理tokens数量

        if requests:  # 如果请求列表不为空
            return Batch(self._replica_id, requests, num_tokens)  # 返回新的批次

        # 更安全地对preempted_requests排序以保持FIFO顺序
        self._preempted_requests.sort(key=lambda r: r.arrived_at)  # 根据到达时间排序被中断的请求列表
        # 所有preempted_requests将完成预填充
        while self._preempted_requests:  # 当被中断的请求列表不为空时
            if len(requests) == self._max_micro_batch_size:  # 如果达到最大微批大小
                break

            request = self._preempted_requests.pop(0)  # 从列表中移除第一个请求

            while not self._can_allocate_request(request):  # 当不能分配请求时
                if self._preempted_requests:  # 如果还有被中断的请求
                    victim_request = self._preempted_requests.pop(-1)  # 移除最后一个请求作为受害者
                    victim_request.restart()  # 重新启动受害者请求
                    self.free(victim_request.id)  # 释放受害者请求占用的资源
                    self._request_queue = [victim_request] + self._request_queue  # 将受害者请求放回请求队列
                else:
                    request.restart()  # 重新启动请求
                    self.free(request.id)  # 释放请求占用的资源
                    self._request_queue = [request] + self._request_queue  # 将请求放回请求队列
                    break
            else:
                self._allocate_request(request)  # 分配请求所需资源
                next_num_tokens = self._get_request_next_num_tokens(request)  # 获取下一个tokens数量
                requests.append(request)  # 将请求添加到请求列表
                num_tokens.append(next_num_tokens)  # 将tokens数量添加到列表

        if not requests:  # 如果请求列表为空
            return

        return Batch(self._replica_id, requests, num_tokens)  # 返回新的批次