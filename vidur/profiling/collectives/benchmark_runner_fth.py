import gc  # 导入垃圾回收模块，用于手动管理内存
import os  # 导入操作系统接口模块，用于操作环境变量等
from typing import Optional  # 导入Optional类型注解，表示可选值

import ray  # 导入Ray分布式计算框架
import torch  # 导入PyTorch深度学习框架

from vidur.logger import init_logger  # 导入日志初始化函数
from vidur.profiling.collectives.collectives_input import CollectivesInput  # 导入CollectivesInput类，用于定义输入参数
from vidur.profiling.collectives.collectives_wrapper import CollectiveWrapper  # 导入CollectiveWrapper类，用于包装和执行集合通信操作

logger = init_logger(__name__)  # 初始化日志记录器


@ray.remote(num_gpus=1)  # 使用Ray的远程装饰器，指定每个任务使用1个GPU
class BenchmarkRunner:
    def __init__(self, gpu_id: int, max_gpus_per_node: int, head_ip: str) -> None:  # 初始化BenchmarkRunner类
        self._gpu_id = gpu_id  # 当前GPU的ID
        self._max_devices_per_node = max_gpus_per_node  # 每个节点的最大设备数（GPU）
        self._set_cuda_visible_devices()  # 设置CUDA_VISIBLE_DEVICES环境变量，限制可见的GPU
        self._last_num_workers_per_node = None  # 上一次每个节点的工作节点数
        self._last_num_workers = None  # 上一次总工作节点数
        self._head_ip = head_ip  # 主节点的IP地址

    def _set_cuda_visible_devices(self) -> None:  # 设置CUDA_VISIBLE_DEVICES环境变量
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            self._gpu_id % self._max_devices_per_node
        )  # 根据GPU ID和最大设备数设置可见的GPU
        # 移除可能导致图构建异常的NCCL环境变量
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)  # 移除NCCL异步错误处理环境变量
        # 设置NCCL相关环境变量以支持正确的NCCL操作捕获
        os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"  # 禁用NCCL图混合支持
        os.environ["KINETO_LOG_LEVEL"] = "5"  # 设置Kineto日志级别为5
        os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"  # 忽略禁用的P2P通信

    def run_collective(
        self,
        collectives_input: CollectivesInput,
    ) -> Optional[dict]:  # 执行集合通信操作
        if (
            collectives_input.num_workers != self._last_num_workers
            or collectives_input.num_workers_per_node != self._last_num_workers_per_node
        ) and torch.distributed.is_initialized():  # 如果工作节点数或每节点工作节点数发生变化且分布式已初始化
            torch.distributed.destroy_process_group()  # 销毁现有的分布式进程组

        rank = self._get_rank(
            collectives_input.num_workers, collectives_input.num_workers_per_node
        )  # 获取当前进程的Rank
        if rank is None:  # 如果Rank为None，说明当前GPU不参与计算
            return None  # 返回None

        if (
            collectives_input.num_workers != self._last_num_workers
            or collectives_input.num_workers_per_node != self._last_num_workers_per_node
        ):  # 如果工作节点数或每节点工作节点数发生变化
            self._init_communication(
                collectives_input.comm_id,
                rank,
                collectives_input.num_workers,
                collectives_input.num_workers_per_node,
            )  # 初始化分布式通信
            self._last_num_workers = collectives_input.num_workers  # 更新总工作节点数
            self._last_num_workers_per_node = collectives_input.num_workers_per_node  # 更新每节点工作节点数

        wrapper = CollectiveWrapper(
            rank,
            collectives_input.num_workers,
            collectives_input.comm_id,
            collectives_input.collective_size,
            collectives_input.collective,
            collectives_input.num_workers_per_node,
            self._max_devices_per_node,
        )  # 创建CollectiveWrapper对象
        stats = wrapper.profile()  # 调用profile方法进行性能分析
        del wrapper  # 删除wrapper对象以释放资源
        gc.collect()  # 手动触发垃圾回收
        return stats  # 返回性能分析结果

    def _init_communication(
        self, comm_id: int, rank: int, num_workers: int, devices_per_node: int
    ):  # 初始化分布式通信
        logger.info(
            f"Initializing gpu id: {self._gpu_id}, Rank: {rank}, num_workers: {num_workers}, comm_id: {comm_id}, "
            f"devices_per_node: {devices_per_node}, max_devices_per_node: {self._max_devices_per_node}, "
            f"ip_addr: {ray.util.get_node_ip_address()}, CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}"
        )  # 记录初始化日志信息

        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=num_workers,
            init_method=f"tcp://{self._head_ip}:{comm_id}",
        )  # 初始化分布式进程组，使用NCCL后端和TCP通信方式

    def _get_rank(self, num_workers: int, devices_per_node: int):  # 获取当前进程的Rank
        assert self._max_devices_per_node >= devices_per_node  # 确保最大设备数大于等于每节点设备数
        assert self._max_devices_per_node % devices_per_node == 0  # 确保最大设备数能被每节点设备数整除
        assert num_workers % devices_per_node == 0 or num_workers < devices_per_node  # 确保总工作节点数能被每节点设备数整除或小于每节点设备数

        num_nodes = num_workers // devices_per_node  # 计算节点总数
        current_node = self._gpu_id // self._max_devices_per_node  # 计算当前节点编号

        if current_node >= num_nodes:  # 如果当前节点编号超出节点总数
            return None  # 返回None

        local_gpu_id = self._gpu_id % self._max_devices_per_node  # 计算当前节点内的本地GPU ID

        # # 在节点内均匀分布设备
        # node_devices = list(range(self._max_devices_per_node))  # 获取节点内所有设备ID
        # device_offset = self._max_devices_per_node // devices_per_node  # 计算设备偏移量
        # selected_devices = node_devices[::device_offset]  # 选择当前工作节点使用的设备

        # 按顺序打包设备
        selected_devices = list(range(devices_per_node))  # 按顺序选择设备

        if local_gpu_id not in selected_devices:  # 如果本地GPU ID不在选中的设备列表中
            return None  # 返回None

        # 计算当前进程的Rank
        rank = current_node * devices_per_node + selected_devices.index(local_gpu_id)

        return rank  # 返回Rank
