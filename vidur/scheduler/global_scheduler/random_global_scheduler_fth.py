from random import randint  # 导入 randint 函数
from typing import List, Tuple  # 从 typing 模块导入 List 和 Tuple

from vidur.entities import Request  # 从 vidur.entities 模块导入 Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler  # 从 vidur.scheduler.global_scheduler.base_global_scheduler 模块导入 BaseGlobalScheduler

class RandomGlobalScheduler(BaseGlobalScheduler):  # 定义 RandomGlobalScheduler 类，继承自 BaseGlobalScheduler
    def schedule(self) -> List[Tuple[int, Request]]:  # 定义 schedule 方法，返回类型为 List[Tuple[int, Request]]
        self.sort_requests()  # 调用父类的 sort_requests 方法对请求进行排序

        request_mapping = []  # 初始化一个空列表用于存储请求映射
        while self._request_queue:  # 当请求队列不为空时循环
            request = self._request_queue.pop(0)  # 从请求队列中弹出第一个请求
            replica_id = randint(1, self._num_replicas) - 1  # 生成一个随机的副本 ID
            request_mapping.append((replica_id, request))  # 将副本 ID 和请求元组添加到请求映射列表中
        return request_mapping  # 返回请求映射列表