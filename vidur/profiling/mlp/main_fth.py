import argparse  # 导入argparse模块，用于命令行参数解析
import datetime  # 导入datetime模块，用于处理日期和时间
import itertools  # 导入itertools模块，用于迭代操作
import os  # 导入os模块，用于与操作系统交互
from typing import Any, List  # 从typing模块导入Any和List类型提示

import pandas as pd  # 导入pandas库，并简写为pd，用于数据处理
import ray  # 导入ray库，用于并行和分布式处理
import yaml  # 导入yaml库，用于读取和写入YAML文件
from tqdm import tqdm  # 从tqdm库导入tqdm，用于显示进度条

# 导入自定义模块中的类和函数
from vidur.profiling.common.model_config import ModelConfig  # 从common.model_config模块中导入ModelConfig类
from vidur.profiling.mlp.mlp_wrapper import MlpWrapper  # 从mlp.mlp_wrapper模块中导入MlpWrapper类
from vidur.profiling.utils import ProfileMethod, get_num_tokens_to_profile  # 从utils模块中导入ProfileMethod类和get_num_tokens_to_profile函数

def parse_args():  # 定义parse_args函数，用于解析命令行参数
    parser = argparse.ArgumentParser(description="MLP Profiling")  # 创建ArgumentParser对象，并设置描述
    parser.add_argument(  # 添加一个命令行参数选项，用于禁用Ray
        "--disable_ray",
        action="store_true",
        help="Disable Ray",
    )
    parser.add_argument(  # 添加一个命令行参数选项，用于设置使用的GPU数量
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for profiling",
    )
    parser.add_argument(  # 添加一个命令行参数选项，用于设置输出目录
        "--output_dir",
        type=str,
        default="profiling_outputs",
        help="Output directory for profiling results",
    )
    parser.add_argument(  # 添加一个命令行参数选项，用于设置要分析的模型列表
        "--models",
        type=str,
        nargs="+",
        default=[
            "microsoft/phi-2",
            "internlm/internlm-20b",
            "Qwen/Qwen-72B",
            "meta-llama/Llama-2-7b-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-70B",
        ],
        help="Models to profile",
    )
    parser.add_argument(  # 添加一个命令行参数选项，用于设置张量并行工作者的数量
        "--num_tensor_parallel_workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Number of tensor parallel workers to profile",
    )
    parser.add_argument(  # 添加一个命令行参数选项，用于设置最大token数量
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to profile",
    )
    parser.add_argument(  # 添加一个命令行参数选项，用于选择分析方法
        "--profile_method",
        default="record_function",
        choices=[e.value for e in ProfileMethod],
        help="Method to use for measuring time taken by operations (default: %(default)s)",
    )
    args = parser.parse_args()  # 解析命令行参数

    # 根据当前日期和时间更新输出目录，并创建目录
    args.output_dir = (
        f"{args.output_dir}/mlp/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    return args  # 返回解析后的参数

def profile_model(  # 定义profile_model函数，用于分析给定模型
    args: argparse.Namespace, model: str, num_tokens_to_profile: List[int], pbar: Any
):
    model_config = ModelConfig.from_model_name(model)  # 根据模型名称创建模型配置对象

    promises = []  # 创建空列表用于保存异步任务
    all_results = []  # 创建空列表用于保存所有结果

    # 创建Ray的远程类实例，用于模型包裹器
    # 调用 ray.remote() 作为一个函数，它可以动态地将某个类或函数转换为远程对象。
    # 等价于装饰器写法：
    # 如果你熟悉装饰器写法，这相当于
    # 但在这里，我们使用的是函数式写法，而不是装饰器写法。
    # @ray.remote(num_cpus=1, num_gpus=1)
    # class MlpWrapper:
    
    # 在分布式计算框架（如 Ray）中，远程对象（Remote Object） 是一个核心概念。它指的是通过分布式系统生成的、存储在分布式内存中的数据对象。
    # 远程对象通常由远程任务（Remote Task）或远程 Actor 的方法调用返回，并通过一个唯一的引用（ObjectRef 或类似的机制）来访问。
    # 以下是对远程对象的详细解释：
    # 1. 什么是远程对象？

    # 定义：
    # 远程对象是分布式系统中存储在共享内存（通常是 Ray 的对象存储）中的数据。
    # 它们是由远程任务或 Actor 方法执行后生成的结果。
    # 远程对象本身并不直接存储在本地，而是通过一个唯一的标识符（ObjectRef）来引用。
    # 特点：
    # 不可变性：一旦远程对象被创建，它的内容不能被修改。
    # 分布式存储：远程对象可能存储在集群中的任何节点上。
    # 延迟获取：远程对象的内容不会立即返回给调用者，而是需要显式地通过 ray.get() 获取。
    model_wrapper_actor = ray.remote(
        num_cpus=1,
        num_gpus=1,
    )(
        MlpWrapper,
    ).options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})

    for num_tensor_parallel_workers in args.num_tensor_parallel_workers:  # 遍历张量并行工作者数量
        if model_config.no_tensor_parallel and num_tensor_parallel_workers > 1:  # 判断是否需要张量并行
            pbar.update(len(num_tokens_to_profile))  # 更新进度条
            continue

        # 创建多个模型包裹器实例
        model_wrappers = [
            model_wrapper_actor.remote(
                model_config,
                num_tensor_parallel_workers,
                args.profile_method,
                rank,
                args.output_dir,
            )
            for rank in range(args.num_gpus)
        ]
        for num_tokens in num_tokens_to_profile:  # 遍历要分析的token数量
            worker_id = len(promises)  # 获取工作者ID
            promise = model_wrappers[worker_id].profile.remote(
                num_tokens,
            )  # 调用模型的profile方法
            promises.append(promise)  # 添加到异步任务列表

            if len(promises) >= args.num_gpus:  # 如果达到GPU限制
                results = ray.get(promises)  # 获取异步任务结果
                all_results.extend(results)  # 添加到所有结果列表
                promises = []  # 清空异步任务列表

            pbar.update(1)  # 更新进度条

    results = ray.get(promises)  # 获取剩余异步任务结果
    all_results.extend(results)  # 添加到所有结果列表

    df = pd.DataFrame(all_results)  # 将结果转换为DataFrame
    # 将时间统计数据展开为多个列，并添加前缀
    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )

    return df  # 返回结果DataFrame

def main():  # 定义main函数，程序的主入口
    args = parse_args()  # 解析命令行参数 # /disk1/futianhao/software1/vidur/data/model_configs/meta-llama/Llama-2-70b-hf.yml
    yaml.dump(vars(args), open(f"{args.output_dir}/config.yaml", "w"))  # 将参数保存到YAML文件

    num_tokens_to_profile = get_num_tokens_to_profile(args.max_tokens)  # 获取分析的token数量

    total_combos = itertools.product(  # 生成要分析的所有组合
        args.models,
        num_tokens_to_profile,
        args.num_tensor_parallel_workers,
    )

    pbar = tqdm(total=len(list(total_combos)))  # 创建进度条

    for model in args.models:  # 遍历每个模型
        result_df = profile_model(
            args,
            model,
            num_tokens_to_profile,
            pbar,
        )  # 分析模型并获取结果
        # 根据模型名称创建目录
        os.makedirs(f"{args.output_dir}/{model}", exist_ok=True)
        result_df.to_csv(f"{args.output_dir}/{model}/mlp.csv", index=False)  # 保存分析结果为CSV文件

if __name__ == "__main__":  # 判断是否在主模块中执行
    main()  # 调用main函数