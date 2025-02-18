"""
    Automated search for capacity for different systems via latency vs qps data.
    A system is characterised by:
    1. trace
    2. model
    3. sku
    4. scheduler
"""
import argparse  # 导入解析命令行参数的模块
import json  # 导入处理 JSON 数据的模块
import os  # 导入操作系统相关的功能模块
import time  # 导入处理时间相关功能的模块

import yaml  # 导入处理 YAML 格式数据的模块

from vidur.config_optimizer.config_explorer.config_explorer import ConfigExplorer  # 从 vidur.config_optimizer.config_explorer 导入 ConfigExplorer 类
from vidur.logger import init_logger  # 从 vidur.logger 模块中导入 init_logger 函数

logger = init_logger(__name__)  # 初始化日志记录器

def get_args():
    parser = argparse.ArgumentParser()  # 创建一个解析器对象
    parser.add_argument("--num-threads", type=int, default=None)  # 添加名为 "num-threads" 的命令行参数
    parser.add_argument(
        "--min-search-granularity",
        type=float,
        default=2.5,
        help="Minimum search granularity for capacity (%)",
    )  # 添加名为 "min-search-granularity" 的命令行参数，默认值为 2.5，说明其用途的帮助信息
    parser.add_argument("--output-dir", type=str, required=True)  # 添加名为 "output-dir" 的必需命令行参数
    parser.add_argument("--cache-dir", type=str, default="./cache_tmpfs")  # 添加名为 "cache-dir" 的命令行参数，默认值为 "./cache_tmpfs"
    parser.add_argument("--config-path", type=str, required=True)  # 添加名为 "config-path" 的必需命令行参数
    parser.add_argument("--scheduling-delay-slo-value", type=float, default=5.0)  # 添加名为 "scheduling-delay-slo-value" 的命令行参数，默认值为 5.0
    parser.add_argument("--scheduling-delay-slo-quantile", type=float, default=0.99)  # 添加名为 "scheduling-delay-slo-quantile" 的命令行参数，默认值为 0.99
    parser.add_argument("--max-iterations", type=int, default=20)  # 添加名为 "max-iterations" 的命令行参数，默认值为 20
    parser.add_argument(
        "--time-limit", type=int, default=30, help="Time limit in minutes"
    )  # 添加名为 "time-limit" 的命令行参数，默认值为 30，说明其用途的帮助信息
    parser.add_argument("--debug", action="store_true")  # 添加名为 "debug" 的布尔命令行参数
    parser.add_argument("--skip-cache-warmup", action="store_true")  # 添加名为 "skip-cache-warmup" 的布尔命令行参数

    args = parser.parse_args()  # 解析命令行参数

    default_num_threads = os.cpu_count() - 2  # 获取默认的线程数，等于 CPU 核心数减 2
    if args.num_threads is not None:
        args.num_threads = min(args.num_threads, default_num_threads)  # 如果指定了线程数，取指定值和默认值的最小值
    else:
        args.num_threads = default_num_threads  # 如果没有指定线程数，使用默认值

    return args  # 返回解析后的命令行参数


if __name__ == "__main__":
    args = get_args()  # 获取命令行参数

    config = yaml.safe_load(open(args.config_path))  # 读取配置文件

    assert (
        args.scheduling_delay_slo_quantile >= 0
        and args.scheduling_delay_slo_quantile <= 1
    )  # 确保 scheduling_delay_slo_quantile 在 0 到 1 之间

    os.makedirs(args.output_dir, exist_ok=True)  # 创建输出目录，如果不存在则创建

    logger.info("Starting config optimizer")  # 记录日志信息
    logger.info(f"Args: {args}")  # 记录命令行参数信息
    logger.info(f"Config: {config}")  # 记录配置文件信息

    # 存储配置文件和参数
    json.dump(vars(args), open(f"{args.output_dir}/args.json", "w"))  # 将命令行参数保存为 JSON 文件
    json.dump(config, open(f"{args.output_dir}/config.json", "w"))  # 将配置文件保存为 JSON 文件

    multiple_capacity_search = ConfigExplorer(args, config)  # 创建 ConfigExplorer 对象

    start_time = time.time()  # 记录开始时间

    all_results = multiple_capacity_search.run()  # 运行配置探索

    end_time = time.time()  # 记录结束时间

    logger.info(f"Simulation took time: {end_time - start_time}")  # 记录仿真运行时间