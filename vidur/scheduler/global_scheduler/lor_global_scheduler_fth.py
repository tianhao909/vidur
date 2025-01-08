from typing import List, Tuple  # 导入类型 List 和 Tuple

from vidur.entities import Request  # 从 vidur.entities 模块导入 Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler  # 从 vidur.scheduler.global_scheduler.base_global_scheduler 模块导入 BaseGlobalScheduler


class LORGlobalScheduler(BaseGlobalScheduler):  # 定义类 LORGlobalScheduler，继承自 BaseGlobalScheduler
    """
    Least outstanding requests (LOR) global scheduler.  # 最少未完成请求 (LOR) 全局调度器。
    """

    def schedule(self) -> List[Tuple[int, Request]]:  # 定义 schedule 方法，返回类型为 List[Tuple[int, Request]]
        self.sort_requests()  # 调用 sort_requests 方法

        request_mapping = []  # 初始化 request_mapping 列表
        # keep a map of replica_id -> replica_scheduler  # 保持一个 replica_id -> replica_scheduler 的映射
        # this is used to find the replica with the least outstanding requests  # 这用于找到未完成请求最少的副本
        pending_requests_map = {  # 初始化 pending_requests_map 字典
            replica_scheduler.replica_id: replica_scheduler.num_pending_requests  # replica_scheduler.replica_id 映射到 replica_scheduler.num_pending_requests
            for replica_scheduler in self._replica_schedulers.values()  # 遍历 _replica_schedulers 的所有值
        }

        # using a very simple implementation here, to keep wiring simple  # 这里使用一个非常简单的实现，以保持连接的简单
        while self._request_queue:  # 当 _request_queue 不为空时
            request = self._request_queue.pop(0)  # 弹出 _request_queue 的第一个元素
            replica_id = min(pending_requests_map.items(), key=lambda x: x[1])[0]  # 找到 pending_requests_map 中值最小的项，并获取其键（replica_id）
            pending_requests_map[replica_id] += 1  # 递增该 replica_id 在 pending_requests_map 中的值
            request_mapping.append((replica_id, request))  # 将 (replica_id, request) 元组添加到 request_mapping 列表中

        return request_mapping  # 返回 request_mapping