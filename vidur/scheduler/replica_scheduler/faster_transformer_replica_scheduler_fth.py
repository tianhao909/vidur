from vidur.entities.batch import Batch  # 从vidur.entities.batch导入Batch类
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (  # 从vidur.scheduler.replica_scheduler.base_replica_scheduler导入BaseReplicaScheduler
    BaseReplicaScheduler,  # 导入BaseReplicaScheduler类
)


class FasterTransformerReplicaScheduler(BaseReplicaScheduler):  # 定义从BaseReplicaScheduler继承的FasterTransformerReplicaScheduler类
    def __init__(self, *args, **kwargs):  # 初始化方法，接收任意数量的位置参数和关键字参数
        super().__init__(*args, **kwargs)  # 调用父类的初始化方法

        self._preempted_batches = []  # 初始化被中断的批处理列表
        self._num_running_batches = 0  # 初始化当前正在运行的批处理数量
        self._pending_free_map = {}  # 初始化待释放的内存映射

    def on_batch_end(self, batch: Batch) -> None:  # 定义在批处理结束时调用的方法
        self._num_running_batches -= 1  # 减少运行中的批处理数量

        if batch.all_requests_completed:  # 如果所有请求都已完成
            # 释放所有请求使用的内存
            self.free_batch(batch)  # 释放整个批处理的内存
            self.free(*self._pending_free_map.pop(batch.id, []))  # 释放与批处理关联的所有内存
        else:  # 如果有未完成的请求
            self._preempted_batches.append(batch)  # 将批处理添加到被中断的批处理中

    def _generate_next_batch_from_preempted(self, preempted_batch: Batch) -> Batch:  # 从被中断的批处理中生成下一个批处理
        requests = []  # 初始化请求列表
        num_tokens = []  # 初始化请求的令牌数量列表

        for request in preempted_batch.requests:  # 遍历被中断批处理中的所有请求
            if request.completed:  # 如果请求已完成
                continue  # 跳过已完成的请求
            next_num_tokens = self._get_request_next_num_tokens(request)  # 获取请求的下一个令牌数量
            requests.append(request)  # 将请求添加到请求列表中
            num_tokens.append(next_num_tokens)  # 将令牌数量添加到令牌数量列表中

        if not requests:  # 如果没有请求
            return  # 返回None

        return Batch(self._replica_id, requests, num_tokens)  # 返回新的Batch对象

    def _get_next_batch(self) -> Batch:  # 获取下一个批处理
        if self._preempted_batches:  # 如果有被中断的批处理
            preempted_batch = self._preempted_batches.pop(0)  # 弹出第一个被中断的批处理
            return self._generate_next_batch_from_preempted(preempted_batch)  # 从被中断的批处理中生成并返回下一个批处理

        requests = []  # 初始化请求列表
        num_tokens = []  # 初始化请求的令牌数量列表

        while self._request_queue:  # 当请求队列不为空时
            if len(requests) == self._max_batch_size:  # 如果请求数量达到最大批处理大小
                break  # 退出循环

            if not self.can_allocate(self._max_blocks_per_sequence):  # 如果不能分配最大序列块
                break  # 退出循环

            request = self._request_queue.pop(0)  # 从请求队列中弹出第一个请求
            self.allocate(request.id, self._max_blocks_per_sequence)  # 为请求分配内存
            next_num_tokens = self._get_request_next_num_tokens(request)  # 获取请求的下一个令牌数量
            requests.append(request)  # 将请求添加到请求列表中
            num_tokens.append(next_num_tokens)  # 将令牌数量添加到令牌数量列表中

        if not requests:  # 如果没有请求
            return  # 返回None

        return Batch(self._replica_id, requests, num_tokens)  # 返回新的Batch对象