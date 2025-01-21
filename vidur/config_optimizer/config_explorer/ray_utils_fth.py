import os
import platform
import socket
import time
from typing import Optional

import ray


def get_ip() -> str:
    # special handling for macos
    # 针对 MacOS 的特殊处理
    if platform.system() == "Darwin":
        return "127.0.0.1"

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建一个 UDP socket
    s.settimeout(0)  # 设置超时时间为 0
    try:
        s.connect(("10.254.254.254", 1))  # 尝试连接到一个 IP
        ip = s.getsockname()[0]  # 获取 socket 的绑定地址，即当前 IP
    except Exception:
        ip = "127.0.0.1"  # 出现异常则返回本地回环地址
    finally:
        s.close()  # 关闭 socket
    return ip


def get_nodes() -> list[str]:
    cluster_resources_keys = list(ray.available_resources().keys())  # 获取所有可用资源的键
    ip_addresses = [
        x
        for x in cluster_resources_keys
        if x.startswith("node:") and x != "node:__internal_head__"  # 筛选出节点 IP 地址
    ]

    # special handling for macos, ensure that we only have one node
    # 对于 MacOS 的特殊处理，确保只有一个节点
    if platform.system() == "Darwin":
        assert len(ip_addresses) == 1

    return ip_addresses


def run_on_each_node(func, *args, **kwargs):
    ip_addresses = get_nodes()  # 获取节点 IP 地址
    remote_func = ray.remote(func)  # 将函数包装为远程可调用函数
    return ray.get(
        [
            remote_func.options(resources={ip_address: 0.1}).remote(*args, **kwargs)  # 在每个节点上调用函数
            for ip_address in ip_addresses
        ]
    )


@ray.remote
class CpuAssignmentManager:
    def __init__(self):
        self._nodes = get_nodes()  # 获取节点列表
        # remove "node:" prefix
        # 移除 "node:" 前缀
        self._nodes = [node[5:] for node in self._nodes]
        self._num_cores = os.cpu_count() - 2  # 获取 CPU 核心数并保留两个核心为系统使用
        self._core_mapping = {node: [False] * self._num_cores for node in self._nodes}  # 为每个节点创建核心映射

    def get_cpu_core_id(self) -> Optional[int]:
        for node in self._nodes:
            for i, is_core_assigned in enumerate(self._core_mapping[node]):  # 遍历核心映射
                if not is_core_assigned:
                    self._core_mapping[node][i] = True  # 将核心标记为已分配
                    return node, i  # 返回节点和核心 ID
        return None, None  # 如果没有可用核心，则返回 None

    def release_cpu_core_id(self, node: str, cpu_core_id: int) -> None:
        self._core_mapping[node][cpu_core_id] = False  # 释放 CPU 核心，将其标记为未分配


class RayParallelRunner:
    def __init__(self):
        self._cpu_assignment_manager = CpuAssignmentManager.remote()  # 创建 CPU 分配管理器的远程实例

    def map(self, func, collection):
        # try to assign a core to each task
        # 尝试为每个任务分配一个核心
        promises = []

        remote_func = ray.remote(func)  # 将函数包装为远程可调用函数

        for item in collection:
            node = None
            cpu_core_id = None
            while node is None:
                node, cpu_core_id = ray.get(
                    self._cpu_assignment_manager.get_cpu_core_id.remote()  # 获取一个可用的 CPU 核心
                )
                time.sleep(0.1)  # 等待一段时间，避免过于频繁的请求
            # launch the task
            # 启动任务
            promise = remote_func.options(resources={f"node:{node}": 0.001}).remote(
                self._cpu_assignment_manager, cpu_core_id, item  # 在指定的节点和核心上执行任务
            )
            promises.append(promise)

        return ray.get(promises)  # 获取所有任务结果
