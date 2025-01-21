import argparse  # 导入 argparse 模块，用于命令行参数解析
import copy  # 导入 copy 模块，用于对象的深拷贝

import ray  # 导入 Ray 分布式计算框架

from vidur.config_optimizer.config_explorer.capacity_search import CapacitySearch  # 从指定路径导入 CapacitySearch 类
from vidur.config_optimizer.config_explorer.config import JobConfig  # 从指定路径导入 JobConfig 类
from vidur.config_optimizer.config_explorer.ray_utils import (  # 从指定路径导入相关工具模块
    CpuAssignmentManager,  # 导入 CpuAssignmentManager 类
    RayParallelRunner,  # 导入 RayParallelRunner 类
    run_on_each_node,  # 导入 run_on_each_node 函数
)


def run_search(
    job_config: JobConfig,  # 定义 run_search 函数，接收一个 JobConfig 对象
    args: argparse.Namespace,  # 接收命令行参数解析结果
    cpu_core_assignment_manager: CpuAssignmentManager = None,  # CPU 分配管理器，默认为 None
    cpu_core_id: int = None,  # 指定的 CPU 核心 ID
):
    capacity_search = CapacitySearch(  # 创建 CapacitySearch 对象
        job_config,  # 传入作业配置
        args,  # 传入命令行参数
        cpu_core_assignment_manager,  # 传入 CPU 分配管理器
        cpu_core_id,  # 传入 CPU 核心 ID
    )
    return capacity_search.search()  # 执行搜索并返回结果


class ConfigExplorer:  # 定义 ConfigExplorer 类
    def __init__(  # 初始化方法
        self,
        args: argparse.Namespace,  # 接收命令行参数解析结果
        config: dict,  # 接收配置字典
    ):
        self.args = args  # 保存命令行参数
        self.config = config  # 保存配置字典

        ray.init(ignore_reinit_error=True)  # 初始化 Ray，忽略重复初始化错误

    def _warmup_cache(self):  # 缓存预热内部方法
        job_configs = JobConfig.generate_unique_model_job_configs(self.config)  # 生成独特的作业配置列表

        args_for_warmup = copy.deepcopy(self.args)  # 深拷贝命令行参数
        args_for_warmup.max_iterations = 1  # 设置最大迭代次数为 1

        for job_config in job_configs:  # 遍历每个作业配置
            all_node_results = run_on_each_node(  # 在每个节点上运行
                run_search,  # 指定运行的函数
                job_config,  # 传入作业配置
                args_for_warmup,  # 传入预热参数
            )
            assert all(all_node_results) or not any(  # 断言所有节点结果一致
                all_node_results
            ), "All nodes should have the same result"  # 保证所有节点具有相同的结果

    def run(self):  # 运行方法
        if not self.args.skip_cache_warmup:  # 如果未跳过缓存预热
            self._warmup_cache()  # 执行缓存预热

        job_configs = JobConfig.generate_job_configs(self.config)  # 生成作业配置列表

        ray_parallel_runner = RayParallelRunner()  # 创建 Ray 并行执行器对象

        # 1 Lambda表达式：
        # lambda 是Python中用来创建匿名函数的关键字。匿名函数是指没有显式命名的函数。常用于需要一个简单函数而不想正式定义一个函数的场景。
        # 该表达式后的参数是函数的输入参数，在这里是cpu_core_assignment_manager, cpu_core_id, 和 job_config。

        # 2 参数：
        # cpu_core_assignment_manager：这是传给Lambda函数的第一个参数，通常用于管理CPU核分配。类似一个资源管理器。
        # cpu_core_id：这是传给Lambda函数的第二个参数，表示CPU核心的ID。从名字上看，用于指定哪个CPU核心被分配执行任务。
        # job_config：这是传给Lambda函数的第三个参数，表示作业配置。用来定义作业的具体执行要求和环境。

        # 3 Lambda函数体：
        # 调用run_search函数，该函数通常定义了具体的搜索操作逻辑。
        # run_search函数使用的参数：
        # job_config：如上所述，作业的具体配置。
        # self.args：类实例的成员，可能包含命令行参数或其他相关参数，决定程序流程。
        # cpu_core_assignment_manager：同Lambda参数一，用于管理和分配资源。
        # cpu_core_id：同Lambda参数二，用于指定CPU核心。

        remote_func = (  # 定义远程执行函数
            lambda cpu_core_assignment_manager, cpu_core_id, job_config: run_search(  # 使用 Lambda 表达式
                job_config,  # 传入作业配置
                self.args,  # 传入命令行参数
                cpu_core_assignment_manager,  # 传入 CPU 分配管理器
                cpu_core_id,  # 传入 CPU 核心 ID
            )
        )
        all_results = ray_parallel_runner.map(  # 执行并行映射
            remote_func,  # 传入远程执行函数
            job_configs,  # 传入作业配置列表
        )
        return all_results  # 返回所有结果