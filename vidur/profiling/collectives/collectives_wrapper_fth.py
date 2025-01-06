import numpy as np  # 导入 numpy 库
import torch  # 导入 torch 库

from vidur.profiling.collectives.collectives_impl import GraphedCollective  # 从 vidur.profiling.collectives.collectives_impl 导入 GraphedCollective 类
from vidur.profiling.common.cuda_timer import CudaTimer  # 从 vidur.profiling.common.cuda_timer 导入 CudaTimer 类
from vidur.profiling.common.timer_stats_store import TimerStatsStore  # 从 vidur.profiling.common.timer_stats_store 导入 TimerStatsStore 类

WARMUP_STEPS = 1  # 设置预热步骤数为 1
ACTIVE_STEPS = 3  # 设置活动步骤数为 3
GRAPH_DISABLED_STEPS = 10  # 设置禁用图形的步骤数为 10
DISABLE_GRAPH = True  # 设置是否禁用图形为 True


class CollectiveWrapper:  # 定义 CollectiveWrapper 类
    def __init__(
        self,
        rank: int,  # 定义 rank 参数
        num_workers: int,  # 定义 num_workers 参数
        comm_id: int,  # 定义 comm_id 参数
        size: int,  # 定义 size 参数
        collective: str,  # 定义 collective 参数
        devices_per_node: int,  # 定义 devices_per_node 参数
        max_devices_per_node: int,  # 定义 max_devices_per_node 参数
    ) -> None:  # 定义 __init__ 方法的返回类型
        self._rank = rank  # 初始化 self._rank
        self._num_workers = num_workers  # 初始化 self._num_workers
        self._size = size  # 初始化 self._size
        self._comm_id = comm_id  # 初始化 self._comm_id
        self._collective = collective  # 初始化 self._collective
        self._devices_per_node = devices_per_node  # 初始化 self._devices_per_node
        self._max_devices_per_node = max_devices_per_node  # 初始化 self._max_devices_per_node

        self._graphed_collective = GraphedCollective(  # 初始化 self._graphed_collective
            num_workers, size, collective=collective, disable_graph=DISABLE_GRAPH
        )

        self.timer_stats_store = TimerStatsStore(profile_method="kineto")  # 初始化 self.timer_stats_store
        self._cuda_timer = CudaTimer(  # 初始化 self._cuda_timer
            collective, aggregation_fn=np.median, filter_str="nccl"
        )

    def _run_collective(self):  # 定义 _run_collective 方法
        torch.cuda.synchronize()  # 同步 CUDA
        torch.distributed.barrier()  # 在所有进程间同步

        with self._cuda_timer:  # 使用 self._cuda_timer
            if DISABLE_GRAPH:  # 如果禁用图形
                for _ in range(GRAPH_DISABLED_STEPS):  # 迭代 GRAPH_DISABLED_STEPS 次
                    self._graphed_collective.launch()  # 启动 collectives

            self._graphed_collective.launch()  # 启动 collectives

        torch.cuda.synchronize()  # 同步 CUDA

    def profile(self):  # 定义 profile 方法
        self.timer_stats_store.clear_stats()  # 清空统计数据
        for _ in range(ACTIVE_STEPS):  # 迭代 ACTIVE_STEPS 次
            self._run_collective()  # 运行集体操作

        return {  # 返回统计数据
            "time_stats": self.timer_stats_store.get_stats(),  # 时间统计
            "rank": self._rank,  # 排名
            "num_workers": self._num_workers,  # 工作者数量
            "size": self._size * 2,  # 大小 (字节)
            "collective": self._collective,  # 集体操作类型
            "devices_per_node": self._devices_per_node,  # 每个节点上的设备数量
            "max_devices_per_node": self._max_devices_per_node,  # 每个节点上的最大设备数量
        } 