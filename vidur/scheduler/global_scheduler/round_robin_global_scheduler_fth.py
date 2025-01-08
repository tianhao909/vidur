from typing import List, Tuple  # 导入 List 和 Tuple 类型，用于类型注解

from vidur.entities import Request  # 从 vidur.entities 模块导入 Request 类
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler  # 导入 BaseGlobalScheduler 基类

class RoundRobinGlobalScheduler(BaseGlobalScheduler):  # 定义类 RoundRobinGlobalScheduler 继承自 BaseGlobalScheduler
    def __init__(self, *args, **kwargs):  # 初始化方法，接受可变数量的参数
        super().__init__(*args, **kwargs)  # 调用父类的初始化方法
        self._request_counter = 0  # 初始化请求计数器为 0

    def schedule(self) -> List[Tuple[int, Request]]:  # 定义调度方法，返回值类型为包含(int, Request) 元组的列表
        self.sort_requests()  # 调用自定义的请求排序方法

        request_mapping = []  # 初始化一个空列表用于存储请求映射
        while self._request_queue:  # 当请求队列不为空时循环
            request = self._request_queue.pop(0)  # 获取并删除队列的第一个请求
            replica_id = self._request_counter % self._num_replicas  # 计算副本 ID，以实现轮询调度
            self._request_counter += 1  # 请求计数器加 1
            request_mapping.append((replica_id, request))  # 将副本 ID 和请求组成的元组添加到请求映射中

        return request_mapping  # 返回请求映射列表