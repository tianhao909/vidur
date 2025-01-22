import argparse  # 导入argparse模块，用于命令行参数解析
import glob  # 导入glob模块，提供文件通配符支持
import os  # 导入os模块，提供操作系统接口功能
import platform  # 导入platform模块，获取操作系统及其属性
import shlex  # 导入shlex模块，按照shell语法拆分字符串
from subprocess import Popen  # 从subprocess模块导入Popen类，用于子进程创建和管理

import pandas as pd  # 导入pandas模块，提供数据结构和数据分析工具
import ray  # 导入ray模块，用于并行和分布式计算

from vidur.config_optimizer.config_explorer.config import JobConfig, SimulationConfig  # 从vidur模块导入JobConfig和SimulationConfig类
from vidur.config_optimizer.config_explorer.ray_utils import (  # 从vidur模块导入相关函数和类
    CpuAssignmentManager,
    get_ip,
)
from vidur.logger import init_logger  # 从vidur模块导入init_logger函数，初始化日志记录

logger = init_logger(__name__)  # 初始化日志记录器

class CapacitySearch:  # 定义一个CapacitySearch类
    def __init__(
        self,
        job_config: JobConfig,
        args: argparse.Namespace,
        cpu_core_assignment_manager: CpuAssignmentManager = None,
        cpu_core_id: int = None,
    ):
        self.node_ip = get_ip()  # 获取局域网内节点的IP地址
        self.cpu_core_id = None
        self.job_config = job_config  # 任务配置
        self.args = args  # 命令行参数
        self.cpu_core_assignment_manager = cpu_core_assignment_manager  # CPU核心分配管理
        self.cpu_core_id = cpu_core_id  # 被分配的CPU核心ID

    def release_cpu_core_id(self):  # 释放CPU核心ID
        if self.cpu_core_id is None:  # 如果没有分配任何核心，直接返回
            return

        # 通过Ray远程调用释放当前分配的核心ID
        ray.get(
            self.cpu_core_assignment_manager.release_cpu_core_id.remote(
                self.node_ip,
                self.cpu_core_id,
            )
        )

    def _generate_run_command(
        self,
        scheduler_config: SimulationConfig,  # 调度器配置
    ):
        # 初始化一个字符串变量 cpu_affinity_command，默认为空字符串。
        # 该变量用于存储将 CPU 亲和性设置的命令。CPU 亲和性指的是在多核处理器上，让进程（任务）只在指定的 CPU 核心上运行
        cpu_affinity_command = ""  # 初始化CPU亲和性命令为空

        # 检查 self.cpu_core_id 是否不为空，即检查是否指定了 CPU 核心。
        # 使用 platform.system() 方法检查当前操作系统，如果操作系统不是 macOS（Darwin内核），则条件成立。
        # 如果前面的条件成立，使用 taskset 命令构建 CPU 亲和性命令， --cpu-list 参数指定了允许进程运行的 CPU 核心。
        # self.cpu_core_id 是指定的 CPU 核心ID，如核心0或核心1。
        if self.cpu_core_id is not None and platform.system() != "Darwin":  # 如果分配了CPU核心，并且不是macOS系统
            cpu_affinity_command = f"taskset --cpu-list {self.cpu_core_id}"  # 构建任务集的CPU分配命令

        # 构建完整的运行命令，包含nice命令和CPU亲和性命令
        # 构建完整的运行命令，nice -n 1 用于设置进程的调度优先级，这里的数值 1 代表设置为较低优先级。
        # {cpu_affinity_command} 是插入前面构建好的 CPU 亲和性命令，如果没有指定核心或是 macOS 系统，这部分将是空字符串。
        # python -m vidur.main 调用 Python 作为解释器，运行模块 vidur.main。
        # {scheduler_config.to_args()} 调用 scheduler_config 对象的 to_args() 方法将配置转换为命令行参数字符串。

        # 这行代码是一条用 Python 构造并执行 Shell 命令的示例。我们来逐步解析这条代码的语法和作用：

        # f"" 字符串：
        # 这是 Python 3.6 引入的格式化字符串（f-string），用于简化字符串插值操作。f-string 以字母 f 或 F 开头，在字符串内部可以通过 {} 包含变量或表达式，Python 会在运行时将其替换为相应的值。
        
        # nice -n 1：
        # nice 是一个 Unix/Linux 命令，用于启动一个进程并设定其优先级。-n 1 设置进程的优先级增量为 1，这意味着将以较低的优先级运行命令。这可以减少进程对 CPU 资源的竞争，使其他重要进程获得更多 CPU 资源。
        
        # {cpu_affinity_command}：
        # 这是一个嵌入在 f-string 中的变量或表达式，表示一个 Python 变量（或可选的代码片段），其值将在字符串格式化时插入到此处。cpu_affinity_command 是一个变量，可能代表设置 CPU 亲和性（进程绑定到特定 CPU 核心）的命令。
        
        # python -m vidur.main：
        # 这部分是使用 Python 解释器以模块形式运行一个 Python 脚本。-m 参数表示以模块方式运行，vidur.main 应该是模块路径，指向某个 Python 包中的 main 模块。
        
        # {scheduler_config.to_args()}：
        # 这也是一个嵌入在 f-string 中的表达式。这里调用了 scheduler_config 对象的 to_args() 方法，假定此方法生成一个字符串或字符串列表，这些字符串最终会添加到命令行参数中，传递给 vidur.main 模块。
        command = f"nice -n 1 {cpu_affinity_command} python -m vidur.main {scheduler_config.to_args()}"

        # 使用日志记录器输出调试信息，显示即将运行的完整命令字符串 command。
        # logger.debug 方法用于输出调试等级的日志信息。
        logger.debug(f"Running command: {command}")  # 输出调试信息

        return command  # 返回构建的命令

    def _get_result_file(self, run_dir: str) -> str:  # 获取结果文件
        # glob模块：glob 是文件模式匹配的一个模块，允许你使用符号匹配文件系统中的文件。
        # 路径匹配：f"{run_dir}/*/plots/request_scheduling_delay.csv" 是用来生成路径字符串，其中 f"{run_dir}" 是一个 f-string，占位符 run_dir 会被传入的参数替换。* 是通配符，表示匹配任意目录。
        # 文件路径：它试图在 run_dir 目录下的任何子目录中的 plots 子目录中查找 request_scheduling_delay.csv 文件。
        # >>fth run_dir=vidur/config_optimizer/config_explorer/search_output_fth/runs/7003b794/16.0
        scheduling_delay_file = glob.glob(  # 使用glob模块匹配结果文件
            f"{run_dir}/*/plots/request_scheduling_delay.csv"
        )
        if len(scheduling_delay_file) == 0:  # 如果没有匹配到文件，返回None
            return

        return scheduling_delay_file[0]  # 返回第一个匹配到的文件的路径

    def _is_under_sla(
        self,
        result_file: str,  # 结果文件路径
        simulator_config: SimulationConfig,  # 模拟器配置
    ) -> tuple[bool, float]:  # 返回一个bool和一个float组成的元组
        scheduling_delay_df = pd.read_csv(result_file)  # 使用pandas读取CSV文件
        scheduling_delay = scheduling_delay_df["request_scheduling_delay"].quantile(
            self.args.scheduling_delay_slo_quantile  # 获取指定分位数的调度延迟
        )
        is_under_scheduling_delay_sla = (
            scheduling_delay <= self.args.scheduling_delay_slo_value  # 判断是否满足调度延迟SLO
        )

        # 输出日志信息，包含模拟器配置名称和调度延迟
        logger.info(
            f"{simulator_config.to_human_readable_name()} - Scheduling delay (P{self.args.scheduling_delay_slo_quantile}): {scheduling_delay}",
        )
        return is_under_scheduling_delay_sla, scheduling_delay  # 返回结果

    def is_under_sla(self, qps: float) -> tuple[bool, float]:  # 通过QPS判断是否满足SLA
        simulator_config = SimulationConfig(
            output_dir=self.args.output_dir,
            cache_dir=self.args.cache_dir,
            qps=qps,  # 请求速率
            time_limit=self.args.time_limit,
            job_config=self.job_config,
        )
        run_dir = simulator_config.get_run_dir()  # 获取运行目录
        os.makedirs(run_dir, exist_ok=True)  # 创建运行目录

        cached_result_file = self._get_result_file(run_dir)
        if cached_result_file:  # 如果已有缓存结果文件，直接返回结果
            return self._is_under_sla(cached_result_file, simulator_config)

        command = self._generate_run_command(simulator_config)  # 生成运行命令

        output_file = open(f"{run_dir}/output.log", "w")  # 打开输出日志文件

        # 将运行命令写入日志文件
        output_file.write(f"Running command: {command}\n")

        try:
            args = shlex.split(command)  # 使用shlex拆分命令行参数
            p = Popen(args, stdout=output_file, stderr=output_file)  # 启动子进程
            p.wait()  # 等待子进程完成

            result_file = self._get_result_file(run_dir)  # 获取结果文件
            assert (
                result_file is not None
            ), f"Result file not found for {simulator_config.to_human_readable_name()}"
            return self._is_under_sla(result_file, simulator_config)  # 返回结果
        except Exception as e:
            logger.error(  # 输出错误信息
                f"Error running: {self.job_config.get_human_readable_name()}, failed with error: {e}",
            )
            return False, None  # 返回False和None表示出错

    def search(self):  # 搜索最大QPS
        """
        Perform binary search to find the maximum QPS under the SLO
        """
        logger.info(
            f"Starting search for {self.job_config.get_human_readable_name()}",
        )  # 输出日志信息

        left = 0  # 二分搜索左边界
        right = self.job_config.start_qps * 2  # 二分搜索右边界
        qps = 0  # 当前QPS
        max_qps_under_sla = None  # 满足SLA的最大QPS
        min_qps_over_sla = 2**32  # 不满足SLA的最小QPS

        for _ in range(self.args.max_iterations):  # 最多迭代max_iterations次
            # 停止条件 - 达到最小搜索粒度
            if abs(left - right) < self.args.min_search_granularity * qps / 100:
                break

            qps = (left + right) / 2  # 计算中间值QPS

            is_under_sla, scheduling_delay = self.is_under_sla(qps)  # 检查是否满足SLA

            if scheduling_delay is None:  # 如果调度延迟为空，退出循环
                break

            if is_under_sla:  # 如果满足SLA
                max_qps_under_sla = qps  # 更新满足SLA的最大QPS

                if scheduling_delay < self.args.scheduling_delay_slo_value / 8:  # 调度延迟非常小
                    # 如果调度延迟非常低，可以更加激进地增加QPS
                    right = min(right * 4, min_qps_over_sla)
                elif scheduling_delay < self.args.scheduling_delay_slo_value / 4:  # 调度延迟较小
                    right = min(right * 2, min_qps_over_sla)
                elif qps > 0.8 * right:  # 接近右边界
                    right = min(right * 2, min_qps_over_sla)

                left = qps  # 更新左边界
            else:  # 如果不满足SLA
                if scheduling_delay > 500:  # 调度延迟大于500
                    right = qps / 2
                elif scheduling_delay > 1000:  # 调度延迟大于1000
                    right = qps / 4
                else:
                    right = qps  # 更新右边界

                min_qps_over_sla = min(min_qps_over_sla, qps)  # 更新不满足SLA的最小QPS

        logger.info(
            f"Max QPS under SLO for {self.job_config.get_human_readable_name()}: {max_qps_under_sla}",
        )  # 输出日志信息

        self.release_cpu_core_id()  # 释放CPU核心ID

        # **: 这是Python中的字典解包操作符。它把字典中的键-值对解包成独立的参数传递给新的字典。
        # 所以当我们写**self.job_config.to_config_dict()时，实际上是把{"param1": "value1", "param2": "value2"}解包为param1="value1", param2="value2"。
        return {
            **self.job_config.to_config_dict(),
            "max_qps_under_sla": max_qps_under_sla,
        }  # 返回配置字典与最大QPS
