from typing import Callable

import torch

WARMUP_STEPS = 5  # 热身步骤数量
GRAPH_STEPS = 3  # 图执行步骤数量


class GraphedCollective:
    def __init__(
        self,
        num_workers: int,
        size: int,
        collective: str = "all_reduce",
        disable_graph: bool = False,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self._size = size  # 数据大小
        self._disable_graph = disable_graph  # 是否禁用图执行
        self._collective_fn = self._get_collective_fn(collective)  # 获取对应的集合通信函数

        self._buffer = torch.empty(
            size=(size,),
            dtype=dtype,
            device="cuda",
        )  # 初始化空的 CUDA 张量缓冲区

        self._gather_buffer = None  # 初始化收集缓冲区为 None
        if collective == "all_gather":
            self._gather_tensor = torch.empty(
                size=(size * num_workers,),
                dtype=dtype,
                device="cuda",
            )  # 如果集合操作为 all_gather，初始化收集张量
        elif collective == "reduce_scatter":
            self._reduce_buffer = torch.empty(
                size=(size * num_workers,),
                dtype=dtype,
                device="cuda",
            )  # 如果集合操作为 reduce_scatter，初始化归约缓冲区

        if not self._disable_graph:
            self._graph = self._build_graph()  # 如果不禁用图执行，构建 CUDA 图

    def _run_all_reduce(self):
        torch.distributed.all_reduce(self._buffer)  # 执行 all_reduce 操作

    def _run_all_gather(self):
        torch.distributed.all_gather_into_tensor(self._gather_tensor, self._buffer)  # 执行 all_gather 操作

    def _run_broadcast(self):
        torch.distributed.broadcast(self._buffer, 0)  # 执行广播操作

    def _run_send_recv(self):
        if torch.distributed.get_rank() == 0:
            torch.distributed.send(self._buffer, 1)  # 如果是进程 0，发送数据
        else:
            torch.distributed.recv(self._buffer, 0)  # 如果是其他进程，接收数据

    def _run_reduce_scatter(self):
        torch.distributed.reduce_scatter_tensor(self._buffer, self._reduce_buffer)  # 执行 reduce_scatter 操作

    def _get_collective_fn(self, collective: str) -> Callable:
        if collective == "all_reduce":
            return self._run_all_reduce  # 返回 all_reduce 函数
        elif collective == "all_gather":
            return self._run_all_gather  # 返回 all_gather 函数
        elif collective == "broadcast":
            return self._run_broadcast  # 返回广播函数
        elif collective == "send_recv":
            return self._run_send_recv  # 返回发送接收函数
        elif collective == "reduce_scatter":
            return self._run_reduce_scatter  # 返回 reduce_scatter 函数
        else:
            raise ValueError(f"Unknown collective: {collective}")  # 如果集合操作未知，抛出异常

    def _build_graph(self) -> torch.cuda.CUDAGraph:
        # Warm up.
        for _ in range(WARMUP_STEPS):
            self._collective_fn()  # 进行热身步骤，运行集合函数

        torch.cuda.synchronize()  # 同步 CUDA 操作

        # Build graph.
        graph = torch.cuda.CUDAGraph()  # 创建 CUDA 图对象

        mempool = torch.cuda.graph_pool_handle()  # 获取图的内存池句柄

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
        ):
            with torch.cuda.graph(graph, mempool):
                for _ in range(GRAPH_STEPS):
                    self._collective_fn()  # 在图上下文中运行集合函数

        torch.cuda.synchronize()  # 同步 CUDA 操作
        return graph  # 返回构建的图

    def launch(self) -> torch.Tensor:
        # NOTE: x must be a slice of self._buffer.
        if self._disable_graph:
            self._collective_fn()  # 如果禁用图执行，直接运行集合函数
        else:
            self._graph.replay()  # 如果启用图执行，重放图