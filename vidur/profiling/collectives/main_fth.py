import argparse  # 导入argparse模块，用于解析命令行参数
import datetime  # 导入datetime模块，用于生成时间戳
import os  # 导入os模块，用于操作文件和目录

import pandas as pd  # 导入pandas模块，用于处理数据框
import ray  # 导入ray模块，用于分布式计算
from tqdm import tqdm  # 导入tqdm模块，用于显示进度条

from vidur.logger import init_logger  # 导入自定义日志初始化函数
from vidur.profiling.collectives.benchmark_runner import BenchmarkRunner  # 导入BenchmarkRunner类，用于运行基准测试
from vidur.profiling.utils import get_collectives_inputs  # 导入get_collectives_inputs函数，用于生成集体通信输入

logger = init_logger(__name__)  # 初始化日志记录器


def parse_args():  # 定义解析命令行参数的函数
    parser = argparse.ArgumentParser(description="MLP Profiling")  # 创建ArgumentParser对象，描述为"MLP Profiling"
    parser.add_argument(  # 添加参数：每个节点的工作线程数组合
        "--num_workers_per_node_combinations",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )
    parser.add_argument(  # 添加参数：输出目录，默认为"profiling_outputs"
        "--output_dir",
        type=str,
        default="profiling_outputs",
        help="Output directory for profiling results",
    )
    parser.add_argument(  # 添加参数：集体通信的最大元素数量，默认为4096*8192
        "--max_collective_size",
        type=int,
        default=4096 * 8192,
        help="Maximum number of elements involved in the collective",
    )
    parser.add_argument(  # 添加参数：要分析的集体通信类型，默认为"all_reduce"
        "--collective",
        default="all_reduce",
        choices=["all_reduce", "send_recv"],
        help="Collective to profile",
    )
    args = parser.parse_args()  # 解析命令行参数

    # 根据当前时间生成唯一的输出目录路径
    args.output_dir = f"{args.output_dir}/collective/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(args.output_dir, exist_ok=True)  # 创建输出目录，如果目录已存在则不报错

    return args  # 返回解析后的参数


def create_runner_pool():  # 定义创建运行池的函数
    total_gpus_available = int(ray.cluster_resources()["GPU"])  # 获取Ray集群中可用的GPU总数
    logger.info(f"Total GPUs available: {total_gpus_available}")  # 记录可用GPU数量到日志

    assert total_gpus_available > 0, "No GPUs available"  # 确保至少有一个GPU可用，否则抛出异常

    all_node_ips = [x["NodeName"] for x in ray.nodes()]  # 获取Ray集群中所有节点的IP地址
    logger.info(f"All node IPs: {all_node_ips}")  # 记录所有节点IP到日志

    assert len(all_node_ips) > 0, "No nodes available"  # 确保至少有一个节点可用，否则抛出异常

    num_nodes = len(all_node_ips)  # 获取节点总数
    gpus_per_node = total_gpus_available // len(all_node_ips)  # 计算每个节点的GPU数量

    runner_pool = []  # 初始化运行池列表
    for gpu_id in range(total_gpus_available):  # 遍历所有GPU
        node_ip = all_node_ips[gpu_id // gpus_per_node]  # 根据GPU ID计算所属节点的IP
        runner_pool.append(  # 将BenchmarkRunner实例添加到运行池
            BenchmarkRunner.options(  # 设置资源分配选项
                resources={
                    f"node:{node_ip}": 0.01,  # 为每个节点分配少量资源
                }
            ).remote(gpu_id, gpus_per_node, all_node_ips[0])  # 远程初始化BenchmarkRunner
        )
    return total_gpus_available, num_nodes, runner_pool  # 返回GPU总数、节点总数和运���池


def main():  # 定义主函数
    args = parse_args()  # 解析命令行参数

    ray.init()  # 初始化Ray集群

    total_gpus_available, num_nodes, runner_pool = create_runner_pool()  # 创建运行池

    all_results = []  # 初始化结果列表

    collectives_inputs = get_collectives_inputs(  # 获取集体通信的输入参数
        num_nodes,
        args.num_workers_per_node_combinations,
        args.max_collective_size,
        args.collective,
        total_gpus_available,
    )

    for collectives_input in tqdm(collectives_inputs):  # 遍历所有集体通信输入，显示进度条
        promises = []  # 初始化Promise列表
        for gpu_id in range(total_gpus_available):  # 遍历所有GPU
            promise = runner_pool[gpu_id].run_collective.remote(collectives_input)  # 异步运行集体通信
            promises.append(promise)  # 将Promise添加到列表

        for gpu_id in range(int(total_gpus_available)):  # 遍历所有GPU
            result = ray.get([promises[gpu_id]])[0]  # 获取GPU的结果
            if result and gpu_id == 0:  # 如果结果非空且是第一个GPU
                all_results.append(result)  # 将结果添加到结果列表

        ray.get(promises)  # 等待所有Promise完成

    # 过滤掉空结果
    all_results = [x for x in all_results if x is not None]

    df = pd.DataFrame(all_results)  # 将结果转换为DataFrame
    # 将time_stats列展开为多个列，并添加前缀"time_stats."
    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )

    # 将结果写入CSV文件
    df.to_csv(f"{args.output_dir}/{args.collective}.csv")


if __name__ == "__main__":  # 确保脚本作为主程序运行时执行main函数
    main()
