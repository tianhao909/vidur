import os
from functools import reduce
from typing import Dict, List

import pandas as pd
import plotly_express as px
import wandb

from vidur.config import SimulationConfig
from vidur.entities import Batch, BatchStage, ExecutionTime, Request
from vidur.logger import init_logger
from vidur.metrics.cdf_sketch import CDFSketch
from vidur.metrics.constants import (
    BatchMetricsCountDistribution,
    BatchMetricsTimeDistribution,
    CpuOperationMetrics,
    OperationMetrics,
    RequestCompletionMetricsTimeSeries,
    RequestMetricsHistogram,
    RequestMetricsTimeDistributions,
    TokenCompletionMetricsTimeSeries,
    TokenMetricsTimeDistribution,
)
from vidur.metrics.data_series import DataSeries
from vidur.metrics.series_average_meter import SeriesAverageMeter
from vidur.utils.mfu_calculator import MFUCalculator

logger = init_logger(__name__)  # 初始化日志记录器

def if_write_metrics(func):
    def wrapper(self, *args, **kwargs):
        if self._config.write_metrics:  # 检查配置中是否允许写入指标
            return func(self, *args, **kwargs)  # 调用原始函数

    return wrapper  # 返回包装函数

REQUEST_ID_STR = "Request Id"  # 请求ID的字符串标识
COUNT_STR = "Count"  # 计数的字符串标识
TIME_STR = "Time (sec)"  # 时间（秒）的字符串标识
BATCH_ID_STR = "Batch Id"  # 批次ID的字符串标识
MEMORY_USAGE_STR = "Memory Usage (%)"  # 内存使用率（百分比）的字符串标识
BUSY_TIME_PERCENT = "Busy Time (%)"  # 忙碌时间（百分比）的字符串标识
UTILIZATION_STR = "Utilization (%)"  # 利用率（百分比）的字符串标识
OPERATION_STR = "Operation"  # 操作的字符串标识
TIME_STR_MS = "Time (ms)"  # 时间（毫秒）的字符串标识

class MetricsStore:

    def __init__(self, simulation_config: SimulationConfig) -> None:
        self._simulation_config = simulation_config  # 存储模拟配置
        self._config = self._simulation_config.metrics_config  # 存储指标配置
        self._last_request_arrived_at = None  # 初始化最后一个请求到达时间

        # 复制配置
        self._num_replicas = self._simulation_config.cluster_config.num_replicas  # 复制配置中的副本数量
        self._num_pipeline_stages = (
            self._simulation_config.cluster_config.replica_config.num_pipeline_stages
        )  # 复制配置中的流水线阶段数

        # 初始化请求指标
        self._request_metrics_time_distributions: Dict[
            RequestMetricsTimeDistributions, DataSeries
        ] = {}
        for metric_name in RequestMetricsTimeDistributions:
            self._request_metrics_time_distributions[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )  # 为每个请求指标时间分布初始化数据系列

        self._token_metrics_time_distribution: Dict[
            TokenMetricsTimeDistribution, DataSeries
        ] = {}
        for metric_name in TokenMetricsTimeDistribution:
            self._token_metrics_time_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )  # 为每个令牌指标时间分布初始化CDF草图

        self._request_metrics_histogram: Dict[RequestMetricsHistogram, DataSeries] = {}
        for metric_name in RequestMetricsHistogram:
            self._request_metrics_histogram[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )  # 为每个请求指标直方图初始化数据系列

        # 初始化批处理指标
        self._batch_metrics_count_distribution: Dict[
            BatchMetricsCountDistribution, CDFSketch
        ] = {}
        self._batch_metrics_count_distribution_per_batch: Dict[
            BatchMetricsCountDistribution, DataSeries
        ] = {}
        for metric_name in BatchMetricsCountDistribution:
            self._batch_metrics_count_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )  # 为每个批处理指标计数分布初始化CDF草图
            self._batch_metrics_count_distribution_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )  # 为每个批处理计数分布初始化数据系列

        self._batch_metrics_time_distribution: Dict[
            BatchMetricsTimeDistribution, CDFSketch
        ] = {}
        self._batch_metrics_time_distribution_per_batch: Dict[
            BatchMetricsTimeDistribution, DataSeries
        ] = {}
        for metric_name in BatchMetricsTimeDistribution:
            self._batch_metrics_time_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )  # 为每个批处理指标时间分布初始化CDF草图
            self._batch_metrics_time_distribution_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )  # 为每个批处理时间分布初始化数据系列

        # 初始化完成指标
        self._request_completion_metrics_time_series: Dict[
            RequestCompletionMetricsTimeSeries, DataSeries
        ] = {}
        for metric_name in RequestCompletionMetricsTimeSeries:
            self._request_completion_metrics_time_series[metric_name] = DataSeries(
                TIME_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )  # 为每个请求完成指标时间系列初始化数据系列
        self._token_completion_metrics_time_series: Dict[
            TokenCompletionMetricsTimeSeries, DataSeries
        ] = {}
        for metric_name in TokenCompletionMetricsTimeSeries:
            self._token_completion_metrics_time_series[metric_name] = DataSeries(
                TIME_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )  # 为每个令牌完成指标时间系列初始化数据系列

        # 初始化操作指标
        self._operation_metrics: Dict[OperationMetrics, CDFSketch] = {}
        self._operation_metrics_per_batch: Dict[OperationMetrics, DataSeries] = {}
        for metric_name in OperationMetrics:
            self._operation_metrics[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )  # 为每个操作指标初始化CDF草图
            self._operation_metrics_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )  # 为每个操作指标每批次初始化数据系列

        self._cpu_operation_metrics: Dict[CpuOperationMetrics, CDFSketch] = {}
        self._cpu_operation_metrics_per_batch: Dict[CpuOperationMetrics, DataSeries] = (
            {}
        )
        for metric_name in CpuOperationMetrics:
            self._cpu_operation_metrics[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )  # 为每个CPU操作指标初始化CDF草图
            self._cpu_operation_metrics_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )  # 为每个CPU操作指标每批次初始化数据系列

        # 每个副本的指标
        self._replica_memory_usage = []  # 初始化副本内存使用率列表
        # 每个副本阶段的指标
        self._replica_busy_time = []  # 初始化副本忙碌时间列表
        self._replica_mfu = []  # 初始化副本MFU（最少频率使用）列表
        self._mfu_calculator = MFUCalculator(
            self._simulation_config.cluster_config.replica_config
        )  # 初始化MFU计算器

        for replica_idx in range(self._num_replicas):
            self._replica_memory_usage.append(
                SeriesAverageMeter(
                    TIME_STR,
                    MEMORY_USAGE_STR,
                    self._config.save_table_to_wandb,
                )
            )  # 为每个副本初始化内存使用率平均计量器
            self._replica_memory_usage[replica_idx].put(0, 0)  # 将初始值（0，0）放入内存使用率列表中

            self._replica_busy_time.append([])  # 为每个副本初始化忙碌时间列表
            self._replica_mfu.append([])  # 为每个副本初始化MFU列表

            for stage_idx in range(self._num_pipeline_stages):
                self._replica_busy_time[replica_idx].append(
                    SeriesAverageMeter(
                        TIME_STR,
                        BUSY_TIME_PERCENT,
                        save_table_to_wandb=self._config.save_table_to_wandb,
                    )
                )  # 为每个副本阶段初始化忙碌时间平均计量器
                self._replica_busy_time[replica_idx][stage_idx].put(0, 0)  # 将初始值（0，0）放入忙碌时间列表中

                self._replica_mfu[replica_idx].append(
                    SeriesAverageMeter(
                        TIME_STR,
                        UTILIZATION_STR,
                        save_table_to_wandb=self._config.save_table_to_wandb,
                    )
                )  # 为每个副本阶段初始化MFU平均计量器
                self._replica_mfu[replica_idx][stage_idx].put(0, 0)  # 将初始值（0，0）放入MFU列表中

        self._init_wandb()  # 初始化wandb

    def _init_wandb(self):
        if (
            not self._config.write_metrics
            or not self._config.wandb_project
            or not self._config.wandb_group
        ):
            return  # 如果没有配置写入指标或项目或组，则返回

        wandb.init(
            project=self._config.wandb_project,
            group=self._config.wandb_group,
            name=self._config.wandb_run_name,
            config=self._simulation_config.to_dict(),
        )  # 初始化wandb，设置项目、组和运行名称

    def _save_as_csv(
        self,
        dataseries_list: List[DataSeries],  # dataseries_list是一个DataSeries类型的列表
        key_to_join: str,  # key_to_join是用于合并DataFrame的键
        base_path: str,  # base_path是保存CSV文件的基础路径
        file_name: str,  # file_name是保存的CSV文件的名称
    ):
        os.makedirs(base_path, exist_ok=True)  # 创建存储文件的目录，如果已存在则不会报错

        merged_df = reduce(
            lambda left, right: pd.merge(left, right, on=[key_to_join], how="outer"),  # 使用外连接合并DataFrame
            [dataseries._to_df() for dataseries in dataseries_list],  # 将DataSeries转换为DataFrame
        )
        merged_df.to_csv(f"{base_path}/{file_name}.csv", index=False)  # 保存合并后的DataFrame为CSV文件
        if wandb.run and self._config.save_table_to_wandb:  # 如果启用了wandb.run并且配置允许保存表格
            wand_table = wandb.Table(dataframe=merged_df)  # 创建一个wandb.Table
            wandb.log({f"{file_name}_table": wand_table}, step=0)  # 记录wandb日志
    def _store_bar_plot(
        self,
        base_path: str,  # base_path是保存柱状图的基础路径
        plot_name: str,  # plot_name是保存的柱状图的名称
        x_label: str,  # x_label是柱状图的x轴标签
        y_label: str,  # y_label是柱状图的y轴标签
        data: Dict[str, float],  # data是一个字典，包含用于绘制柱状图的数据
    ):
        if wandb.run:  # 如果启用了wandb.run
            wandb.log(
                {
                    plot_name: wandb.plot.bar(
                        wandb.Table(
                            dataframe=pd.DataFrame(
                                data=data.items(), columns=[x_label, y_label]  # 将数据转换为DataFrame
                            )
                        ),
                        x_label,
                        y_label,
                        title=plot_name,
                    )
                },
                step=0,
            )
        if self._config.store_plots:  # 如果配置允许存储图表
            fig = px.bar(
                x=list(data.keys()),
                y=list(data.values()),
                labels={"x": x_label, "y": y_label},
            )
            fig.write_image(f"{base_path}/{plot_name}.png")  # 将条形图保存为PNG文件

    def _store_operation_metrics(self, base_plot_path: str):
        if not self._config.store_operation_metrics:  # 如果配置不允许存储操作指标
            return

        total_operation_runtimes: Dict[str, float] = {}  # 保存所有操作的运行时间

        total_operation_runtimes["model_execution_e2e"] = 0
        for dataseries in self._operation_metrics.values():
            dataseries.plot_cdf(
                base_plot_path, f"{dataseries._metric_name}_execution_time", TIME_STR_MS  # 绘制CDF图
            )
            total_operation_runtimes[dataseries._metric_name] = dataseries.sum  # 记录每个数据的总和
            total_operation_runtimes["model_execution_e2e"] += dataseries.sum  # 累加每个数据的总和

        for dataseries in self._cpu_operation_metrics.values():
            dataseries.plot_cdf(
                base_plot_path, f"{dataseries._metric_name}_execution_time", TIME_STR_MS  # 绘制CDF图
            )
            total_operation_runtimes[dataseries._metric_name] = dataseries.sum  # 记录每个数据的总和

        self._store_bar_plot(
            base_plot_path,
            "total_operation_runtimes",
            OPERATION_STR,
            TIME_STR_MS,
            total_operation_runtimes,
        )

        if not self._config.keep_individual_batch_metrics:  # 如果配置不保留单独的批处理指标
            return

        for dataseries in self._operation_metrics_per_batch.values():
            dataseries.consolidate()  # 合并数据
            dataseries.plot_step(
                base_plot_path,
                f"{dataseries._metric_name}_per_batch",
                y_axis_label=TIME_STR_MS,
                y_cumsum=False,
            )
        operations_dataseries_list = list(self._operation_metrics_per_batch.values())
        self._save_as_csv(
            dataseries_list=operations_dataseries_list,
            key_to_join=BATCH_ID_STR,
            base_path=self._config.output_dir,
            file_name="operation_metrics",
        )

        for dataseries in self._cpu_operation_metrics_per_batch.values():
            dataseries.consolidate()  # 合并数据
            dataseries.plot_step(
                base_plot_path,
                f"{dataseries._metric_name}_per_batch",
                y_axis_label=TIME_STR_MS,
                y_cumsum=False,
            )
        cpu_operations_dataseries_list = list(
            self._cpu_operation_metrics_per_batch.values()
        )
        self._save_as_csv(
            dataseries_list=cpu_operations_dataseries_list,
            key_to_join=BATCH_ID_STR,
            base_path=self._config.output_dir,
            file_name="cpu_operation_metrics",
        )

    def _store_request_metrics(self, base_plot_path: str):
        if not self._config.store_request_metrics:  # 如果配置不允许存储请求指标
            return

        all_request_metrics = list(
            self._request_metrics_time_distributions.values()
        ) + list(self._request_metrics_histogram.values())

        self._save_as_csv(
            dataseries_list=all_request_metrics,
            key_to_join=REQUEST_ID_STR,
            base_path=self._config.output_dir,
            file_name="request_metrics",
        )

        for dataseries in self._request_metrics_histogram.values():
            dataseries.plot_histogram(base_plot_path, dataseries._y_name)  # 绘制直方图

        for dataseries in self._request_metrics_time_distributions.values():
            dataseries.plot_cdf(base_plot_path, dataseries._y_name, TIME_STR)  # 绘制CDF图

    def _store_batch_metrics(self, base_plot_path: str):
        if not self._config.store_batch_metrics:  # 如果配置不允许存储批处理指标
            return

        for dataseries in self._batch_metrics_time_distribution.values():
            y_axis_label = (
                TIME_STR_MS
                if "model_execution" in dataseries._metric_name
                else TIME_STR
            )
            dataseries.plot_cdf(base_plot_path, dataseries._metric_name, y_axis_label)  # 绘制CDF图

        for dataseries in self._batch_metrics_count_distribution.values():
            dataseries.plot_cdf(base_plot_path, dataseries._metric_name, COUNT_STR)  # 绘制CDF图

        if not self._config.keep_individual_batch_metrics:  # 如果配置不保留单独的批处理指标
            return

        for dataseries in self._batch_metrics_time_distribution_per_batch.values():
            y_axis_label = (
                TIME_STR_MS
                if "model_execution" in dataseries._metric_name
                else TIME_STR
            )
            dataseries.plot_step(
                base_plot_path,
                f"{dataseries._metric_name}_per_batch",
                y_axis_label=y_axis_label,
                y_cumsum=False,
            ),

        for dataseries in self._batch_metrics_count_distribution_per_batch.values():
            dataseries.plot_step(
                base_plot_path,
                f"{dataseries._metric_name}_per_batch",
                y_axis_label=COUNT_STR,
                y_cumsum=False,
            ),

        all_batch_metrics = list(
            self._batch_metrics_count_distribution_per_batch.values()
        ) + list(self._batch_metrics_time_distribution_per_batch.values())

        self._save_as_csv(
            dataseries_list=all_batch_metrics,
            key_to_join=BATCH_ID_STR,
            base_path=self._config.output_dir,
            file_name="batch_metrics",
        )

    def _store_operation_metrics(self, base_plot_path: str):  # 这是一个用于存储操作指标的方法, 接受一个字符串参数 base_plot_path 用于指定基本绘图路径
        if not self._config.store_operation_metrics:  # 如果配置中不需要存储操作指标
            return  # 直接返回

        total_operation_runtimes: Dict[str, float] = {}  # 初始化总操作运行时间的字典

        total_operation_runtimes["model_execution_e2e"] = 0  # 初始化模型执行的端到端总时间为 0
        for dataseries in self._operation_metrics.values():  # 遍历所有的操作指标数据系列
            dataseries.plot_cdf(  # 绘制累积分布函数图
                base_plot_path, f"{dataseries._metric_name}_execution_time", TIME_STR_MS  # 文件名格式为该指标名称加上执行时间, 单位是毫秒
            )
            total_operation_runtimes[dataseries._metric_name] = dataseries.sum  # 将该指标的总和存入总操作运行时间字典
            total_operation_runtimes["model_execution_e2e"] += dataseries.sum  # 增加到端到端运行时间中

        for dataseries in self._cpu_operation_metrics.values():  # 遍历所有 CPU 操作指标
            dataseries.plot_cdf(  # 绘制累积分布函数图
                base_plot_path, f"{dataseries._metric_name}_execution_time", TIME_STR_MS  # 文件名格式为该指标名称加上执行时间, 单位是毫秒
            )
            total_operation_runtimes[dataseries._metric_name] = dataseries.sum  # 将该指标的总和存入总操作运行时间字典

        self._store_bar_plot(  # 存储总操作时间的条形图
            base_plot_path,
            "total_operation_runtimes",
            OPERATION_STR,  # x轴标签
            TIME_STR_MS,  # y轴标签
            total_operation_runtimes,  # 数据
        )

        if not self._config.keep_individual_batch_metrics:  # 如果配置中不需要保留每个批次的指标
            return  # 直接返回

        for dataseries in self._operation_metrics_per_batch.values():  # 遍历每批次的操作指标
            dataseries.consolidate()  # 合并数据
            dataseries.plot_step(  # 绘制阶梯图
                base_plot_path,
                f"{dataseries._metric_name}_per_batch",  # 文件名格式为该指标名称加上每批次后缀
                y_axis_label=TIME_STR_MS,  # y轴标签是时间，以毫秒为单位
                y_cumsum=False,  # 不累加
            )
        operations_dataseries_list = list(self._operation_metrics_per_batch.values())  # 转为列表
        self._save_as_csv(  # 将操作指标数据系列列表保存为CSV文件
            dataseries_list=operations_dataseries_list,
            key_to_join=BATCH_ID_STR,  # 用批次ID来联合数据
            base_path=self._config.output_dir,  # 基本路径是配置中的输出目录
            file_name="operation_metrics",  # 文件名是操作指标
        )

        for dataseries in self._cpu_operation_metrics_per_batch.values():  # 遍历每批次的 CPU 操作指标
            dataseries.consolidate()  # 合并数据
            dataseries.plot_step(  # 绘制阶梯图
                base_plot_path,
                f"{dataseries._metric_name}_per_batch",  # 文件名格式为该指标名称加上每批次后缀
                y_axis_label=TIME_STR_MS,  # y轴标签是时间，以毫秒为单位
                y_cumsum=False,  # 不累加
            )
        cpu_operations_dataseries_list = list(
            self._cpu_operation_metrics_per_batch.values()
        )  # 转为列表
        self._save_as_csv(  # 将 CPU 操作指标数据系列列表保存为CSV文件
            dataseries_list=cpu_operations_dataseries_list,
            key_to_join=BATCH_ID_STR,  # 用批次ID来联合数据
            base_path=self._config.output_dir,  # 基本路径是配置中的输出目录
            file_name="cpu_operation_metrics",  # 文件名是CPU操作指标
        )

    def _store_request_metrics(self, base_plot_path: str):  # 这是一个用于存储请求指标的方法, 其接受一个字符串参数 base_plot_path 用于指定基本绘图路径
        if not self._config.store_request_metrics:  # 如果配置中不需要存储请求指标
            return  # 直接返回

        all_request_metrics = list(
            self._request_metrics_time_distributions.values()
        ) + list(self._request_metrics_histogram.values())  # 合并时间分布指标和直方图指标为一个列表

        self._save_as_csv(  # 将所有请求指标数据保存为CSV文件
            dataseries_list=all_request_metrics,
            key_to_join=REQUEST_ID_STR,  # 用请求ID来联合数据
            base_path=self._config.output_dir,  # 基本路径是配置中的输出目录
            file_name="request_metrics",  # 文件名是请求指标
        )

        for dataseries in self._request_metrics_histogram.values():  # 对于每一个请求直方图指标
            dataseries.plot_histogram(base_plot_path, dataseries._y_name)  # 绘制直方图

        for dataseries in self._request_metrics_time_distributions.values():  # 对于每一个请求时间分布指标
            dataseries.plot_cdf(base_plot_path, dataseries._y_name, TIME_STR)  # 绘制累积分布函数图

    def _store_batch_metrics(self, base_plot_path: str):  # 这是一个用于存储批次指标的方法, 其接受一个字符串参数 base_plot_path 用于指定基本绘图路径
        if not self._config.store_batch_metrics:  # 如果配置中不需要存储批次指标
            return  # 直接返回

        for dataseries in self._batch_metrics_time_distribution.values():  # 对于每一个批次时间分布指标
            y_axis_label = (  # 根据条件选择 y 轴标签
                TIME_STR_MS
                if "model_execution" in dataseries._metric_name
                else TIME_STR
            )
            dataseries.plot_cdf(base_plot_path, dataseries._metric_name, y_axis_label)  # 绘制累积分布函数图

        for dataseries in self._batch_metrics_count_distribution.values():  # 对于每一个批次次数分布指标
            dataseries.plot_cdf(base_plot_path, dataseries._metric_name, COUNT_STR)  # 绘制累积分布函数图

        if not self._config.keep_individual_batch_metrics:  # 如果配置中不需要保留每个批次的指标
            return  # 直接返回

        for dataseries in self._batch_metrics_time_distribution_per_batch.values():  # 对于每一个批次时间分布指标
            y_axis_label = (  # 根据条件选择 y 轴标签
                TIME_STR_MS
                if "model_execution" in dataseries._metric_name
                else TIME_STR
            )
            dataseries.plot_step(  # 绘制阶梯图
                base_plot_path,
                f"{dataseries._metric_name}_per_batch",  # 文件名格式为该指标名称加上每批次后缀
                y_axis_label=y_axis_label,  # y轴标签是根据条件选择的
                y_cumsum=False,  # 不累加
            ),

        for dataseries in self._batch_metrics_count_distribution_per_batch.values():  # 对于每一个批次次数分布指标
            dataseries.plot_step(  # 绘制阶梯图
                base_plot_path,
                f"{dataseries._metric_name}_per_batch",  # 文件名格式为该指标名称加上每批次后缀
                y_axis_label=COUNT_STR,  # y轴标签是次数
                y_cumsum=False,  # 不累加
            ),

        all_batch_metrics = list(
            self._batch_metrics_count_distribution_per_batch.values()
        ) + list(self._batch_metrics_time_distribution_per_batch.values())  # 合并时间分布和次数分布为一个列表

        self._save_as_csv(  # 将所有批次指标数据保存为CSV文件
            dataseries_list=all_batch_metrics,
            key_to_join=BATCH_ID_STR,  # 用批次ID来联合数据
            base_path=self._config.output_dir,  # 基本路径是配置中的输出目录
            file_name="batch_metrics",  # 文件名是批次指标
        )

    def _store_completion_metrics(self, base_plot_path: str):  # 这是一个用于存储完成指标的方法, 其接受一个字符串参数 base_plot_path 用于指定基本绘图路径
        if self._config.store_request_metrics:  # 如果配置中需要存储请求指标
            for dataseries in self._request_completion_metrics_time_series.values():  # 对于每一个请求完成时间序列
                dataseries.plot_step(  # 绘制阶梯图
                    base_plot_path, f"{dataseries._y_name}_time_series", COUNT_STR  # 文件名格式为该指标名称加上时间序列后缀, y轴标签是次数
                )

        if not self._config.store_token_completion_metrics:  # 如果配置中不需要存储token完成指标
            return  # 直接返回

        for dataseries in self._token_metrics_time_distribution.values():  # 对于每一个token时间分布指标
            dataseries.plot_cdf(base_plot_path, dataseries._metric_name, TIME_STR)  # 绘制累积分布函数图

        for dataseries in self._token_completion_metrics_time_series.values():  # 对于每一个token完成时间序列
            dataseries.plot_step(  # 绘制阶梯图
                base_plot_path, f"{dataseries._y_name}_time_series", COUNT_STR  # 文件名格式为该指标名称加上时间序列后缀, y轴标签是次数
            )

    def _store_utilization_metrics(self, base_plot_path: str):  # 这是一个用于存储利用率指标的方法, 其接受一个字符串参数 base_plot_path 用于指定基本绘图路径
        if not self._config.store_utilization_metrics:  # 如果配置中不需要存储利用率指标
            return  # 直接返回

        for replica_idx in range(self._num_replicas):  # 对于每一个副本
            self._replica_memory_usage[replica_idx].print_stats(  # 打印内存使用统计信息
                f"replica_{replica_idx + 1}_memory_usage", base_plot_path  # 文件名格式为副本索引加上内存使用后缀
            )
            for stage_idx in range(self._num_pipeline_stages):  # 对于每一个流水线阶段
                self._replica_busy_time[replica_idx][stage_idx].print_stats(  # 打印忙碌时间百分比统计信息
                    f"replica_{replica_idx + 1}_stage_{stage_idx + 1}_busy_time_percent",
                    base_plot_path,  # 文件名格式为副本索引加上阶段索引和忙碌时间百分比后缀
                )
                self._replica_mfu[replica_idx][stage_idx].print_stats(  # 打印MFU统计信息
                    f"replica_{replica_idx + 1}_stage_{stage_idx + 1}_mfu",
                    base_plot_path,  # 文件名格式为副本索引加上阶段索引和MFU后缀
                )

    @if_write_metrics  # 此装饰器表示如果需要写入指标则执行此方法
    def plot(self) -> None:  # 定义一个没有返回值的方法用于绘图
        dir_plot_path = f"{self._config.output_dir}/plots"  # 定义绘图路径为输出目录下的plots文件夹
        os.makedirs(dir_plot_path, exist_ok=True)  # 如果目录不存在则创建

        self._store_request_metrics(dir_plot_path)  # 存储请求指标
        self._store_batch_metrics(dir_plot_path)  # 存储批次指标
        self._store_completion_metrics(dir_plot_path)  # 存储完成指标
        self._store_operation_metrics(dir_plot_path)  # 存储操作指标
        self._store_utilization_metrics(dir_plot_path)  # 存储利用率指标

    @if_write_metrics  # 此装饰器表示如果需要写入指标则执行此方法
    def on_request_arrival(self, time: float, request: Request) -> None:  # 定义一个没有返回值的方法, 参数为请求到达时间和请求对象
        if not self._config.store_request_metrics:  # 如果配置中不需要存储请求指标
            return  # 直接返回

        self._request_completion_metrics_time_series[  # 更新请求完成时间序列数据
            RequestCompletionMetricsTimeSeries.REQUEST_ARRIVAL
        ].put(time, 1)

        self._request_metrics_histogram[RequestMetricsHistogram.REQUEST_NUM_TOKENS].put(
            request.id, request.total_tokens  # 记录请求的总token数量到请求指标直方图
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_PREFILL_TOKENS
        ].put(request.id, request.num_prefill_tokens)  # 记录请求的预填充token数量到请求指标直方图
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_DECODE_TOKENS
        ].put(request.id, request.num_decode_tokens)  # 记录请求的解码token数量到请求指标直方图
        self._request_metrics_histogram[RequestMetricsHistogram.REQUEST_PD_RATIO].put(
            request.id, request.pd_ratio  # 记录请求的PD比率到请求指标直方图
        )
        if self._last_request_arrived_at is not None:  # 如果上一个请求的到达时间不为None
            self._request_metrics_histogram[
                RequestMetricsHistogram.REQUEST_INTER_ARRIVAL_DELAY
            ].put(request.id, request.arrived_at - self._last_request_arrived_at)  # 记录请求间的到达延迟到请求指标直方图
        self._last_request_arrived_at = request.arrived_at  # 更新上一个请求的到达时间



    @if_write_metrics
    def _on_request_end(self, time: float, request: Request) -> None:
        if not self._config.store_request_metrics:  # 如果配置没有储存请求指标，则返回
            return

        self._request_completion_metrics_time_series[
            RequestCompletionMetricsTimeSeries.REQUEST_COMPLETION
        ].put(request.completed_at, 1)  # 向请求完成时间序列中添加完成时间和完成值

        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME
        ].put(request.id, request.e2e_time)  # 添加请求端到端时间
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME_NORMALIZED
        ].put(request.id, request.e2e_time_normalized)  # 添加规范化的端到端时间
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_TIME
        ].put(request.id, request.execution_time)  # 添加请求执行时间
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_TIME_NORMALIZED
        ].put(request.id, request.execution_time_normalized)  # 添加规范化的执行时间
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_MODEL_EXECUTION_TIME
        ].put(request.id, request.model_execution_time)  # 添加模型执行时间
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_MODEL_EXECUTION_TIME_NORMALIZED
        ].put(request.id, request.model_execution_time_normalized)  # 添加规范化的模型执行时间
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_PREEMPTION_TIME
        ].put(request.id, request.preempted_time)  # 添加抢占时间
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_SCHEDULING_DELAY
        ].put(request.id, request.scheduling_delay)  # 添加调度延迟
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME
        ].put(request.id, request.execution_time + request.preempted_time)  # 添加执行加抢占时间
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME_NORMALIZED
        ].put(
            request.id,
            (request.execution_time + request.preempted_time)
            / request.num_decode_tokens,  # 添加规范化的执行加抢占时间
        )
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_E2E
        ].put(request.id, request.prefill_completed_at - request.arrived_at)  # 添加预填充端到端时间
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION
        ].put(request.id, request.prefill_completed_at - request.scheduled_at)  # 添加预填充执行加抢占时间
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
        ].put(
            request.id,
            (request.prefill_completed_at - request.scheduled_at)
            / request.num_prefill_tokens,  # 添加规范化的预填充执行加抢占时间
        )
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.DECODE_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
        ].put(
            request.id,
            (request.completed_at - request.prefill_completed_at)
            / request.num_decode_tokens,  # 添加规范化的解码执行加抢占时间
        )

        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_NUM_RESTARTS
        ].put(request.id, request.num_restarts)  # 添加请求重启次数

    def _update_per_token_execution_times(
        self, time: float, request: Request, batch: Batch
    ) -> None:
        # if prefill has just finished in this iteration, update the prefill completion time series
        if (
            time == request.prefill_completed_at
            and self._config.store_token_completion_metrics
        ):  # 如果本次迭代中刚完成预填充，更新预填充完成时间序列
            self._token_completion_metrics_time_series[
                TokenCompletionMetricsTimeSeries.PREFILL_COMPLETIONS
            ].put(
                time,
                request.num_prefill_tokens,  # 添加预填充完成次数
            )

        # determine if this was prefill or decode token
        if not request.has_started_decode:  # 判断是否开始解码
            return

        if not self._config.store_token_completion_metrics:  # 若未配置存储令牌完成指标，则返回
            return

        self._token_metrics_time_distribution[
            TokenMetricsTimeDistribution.DECODE_TOKEN_EXECUTION_PLUS_PREMPTION_TIME
        ].put(
            time - batch.scheduled_at + request.latest_iteration_scheduling_delay,
        )  # 存储解码令牌执行加抢占时间

        self._token_completion_metrics_time_series[
            TokenCompletionMetricsTimeSeries.DECODE_COMPLETIONS
        ].put(time, 1)  # 添加解码完成次数

    def _push_metric(
        self, metric_name: OperationMetrics, batch_id: int, value: float
    ) -> None:
        if metric_name in OperationMetrics:  # 若指标名称属于操作指标
            self._operation_metrics[metric_name].put(value)  # 存储该指标值
            self._operation_metrics_per_batch[metric_name].put(batch_id, value)  # 存储每批次的指标值
        elif metric_name in CpuOperationMetrics:  # 若指标名称属于CPU操作指标
            self._cpu_operation_metrics[metric_name].put(value)  # 存储该CPU指标值
            self._cpu_operation_metrics_per_batch[metric_name].put(batch_id, value)  # 存储每批次的CPU指标值
        elif metric_name in BatchMetricsTimeDistribution:  # 若指标名称属于批处理时间分布
            self._batch_metrics_time_distribution[metric_name].put(value)  # 存储批处理时间分布指标值
            self._batch_metrics_time_distribution_per_batch[metric_name].put(
                batch_id, value  # 存储每批次的批处理时间分布指标值
            )
        elif metric_name in BatchMetricsCountDistribution:  # 若指标名称属于批处理计数分布
            self._batch_metrics_count_distribution[metric_name].put(value)  # 存储批处理计数分布指标值
            self._batch_metrics_count_distribution_per_batch[metric_name].put(
                batch_id, value  # 存储每批次的批处理计数分布指标值
            )
        else:
            raise ValueError(f"Invalid metric name {metric_name}")  # 抛出异常：无效指标名称

    @if_write_metrics
    def on_batch_end(
        self, time: float, batch: Batch, replica_id: int, memory_usage_percent: int
    ) -> None:
        if (
            self._config.min_batch_index and batch.id < self._config.min_batch_index
        ) or (self._config.max_batch_index and batch.id > self._config.max_batch_index):  # 如果批次索引不在配置的范围内，则返回
            return

        for request in batch.completed_requests:  # 对每个已完成请求
            self._on_request_end(time, request)  # 调用请求结束处理

        if self._config.store_utilization_metrics:  # 如果配置存储利用率指标
            self._replica_memory_usage[replica_id - 1].put(time, memory_usage_percent)  # 添加副本内存使用率

        for request in batch.requests:  # 对批次中的每个请求
            self._update_per_token_execution_times(time, request, batch)  # 更新每个令牌的执行时间

        if not self._config.store_batch_metrics:  # 如果未配置存储批处理指标，则返回
            return

        self._push_metric(
            BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME,
            batch.id,
            time - batch.scheduled_at,  # 存储批处理执行时间
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_TOKENS,
            batch.id,
            batch.total_num_tokens,  # 存储批处理总令牌数
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_PREFILL_TOKENS,
            batch.id,
            batch.num_prefill_tokens,  # 存储批处理预填充令牌数
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_DECODE_TOKENS,
            batch.id,
            batch.num_decode_tokens,  # 存储批处理解码令牌数
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_SIZE, batch.id, batch.size  # 存储批次大小
        )

    @if_write_metrics
    def on_replica_schedule(
        self, time: float, replica_id: int, memory_usage_percent: int
    ) -> None:
        if not self._config.store_utilization_metrics:  # 如果未配置存储利用率指标，则返回
            return

        self._replica_memory_usage[replica_id - 1].put(time, memory_usage_percent)  # 存储副本内存使用率

    @if_write_metrics
    def on_replica_stage_schedule(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        batch_stage: BatchStage,
        execution_time: ExecutionTime,
    ) -> None:
        if not self._config.store_utilization_metrics:  # 如果未配置存储利用率指标，则返回
            return

        self._replica_busy_time[replica_id - 1][stage_id - 1].put(time, 100)  # 设置副本在当前阶段的忙时间为100%
        mfu = self._mfu_calculator.get_mfu(batch_stage)  # 获取多功能利用率(MFU)
        self._replica_mfu[replica_id - 1][stage_id - 1].put(time, mfu)  # 存储多功能利用率

        if not self._config.store_operation_metrics:  # 如果未配置存储操作指标，则返回
            return

        batch_id = batch_stage._batch_id
        for _ in range(execution_time.num_layers):  # 遍历每个层级
            self._push_metric(
                OperationMetrics.MLP_UP_PROJ,
                batch_id,
                execution_time.mlp_layer_up_proj_execution_time,  # 存储MLP上升投影执行时间
            )
            self._push_metric(
                OperationMetrics.MLP_ACTIVATION,
                batch_id,
                execution_time.mlp_layer_act_execution_time,  # 存储MLP激活执行时间
            )
            self._push_metric(
                OperationMetrics.MLP_DOWN_PROJ,
                batch_id,
                execution_time.mlp_layer_down_proj_execution_time,  # 存储MLP下降投影执行时间
            )
            self._push_metric(
                OperationMetrics.MLP_DOWN_PROJ_ALL_REDUCE,
                batch_id,
                execution_time.mlp_all_reduce_time,  # 存储MLP全规约时间
            )
            self._push_metric(
                OperationMetrics.ATTN_PRE_PROJ,
                batch_id,
                execution_time.attention_pre_proj_time,  # 存储注意力前投影时间
            )
            self._push_metric(
                OperationMetrics.ATTN_POST_PROJ,
                batch_id,
                execution_time.attention_post_proj_time,  # 存储注意力后投影时间
            )
            self._push_metric(
                OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE,
                batch_id,
                execution_time.attention_all_reduce_time,  # 存储注意力全规约时间
            )

            if execution_time.attention_prefill_execution_time != 0:  # 如果预填充执行时间不为零
                self._push_metric(
                    OperationMetrics.ATTN_PREFILL,
                    batch_id,
                    execution_time.attention_prefill_execution_time,  # 存储注意力预填充执行时间
                )

            if execution_time.attention_decode_execution_time != 0:  # 如果解码执行时间不为零
                self._push_metric(
                    OperationMetrics.ATTN_DECODE,
                    batch_id,
                    execution_time.attention_decode_execution_time,  # 存储注意力解码执行时间
                )
            self._push_metric(
                OperationMetrics.ATTN_KV_CACHE_SAVE,
                batch_id,
                execution_time.attention_kv_cache_save_execution_time,  # 存储注意力KV缓存保存执行时间
            )
            self._push_metric(
                OperationMetrics.ATTN_ROPE,
                batch_id,
                execution_time.attention_rope_execution_time,  # 存储注意力ROPE执行时间
            )
            self._push_metric(
                OperationMetrics.ADD, batch_id, execution_time.add_time * 2  # 存储加法执行时间
            )
            self._push_metric(
                OperationMetrics.INPUT_LAYERNORM,
                batch_id,
                execution_time.attn_norm_time,  # 存储输入层归一化时间
            )
            self._push_metric(
                OperationMetrics.POST_ATTENTION_LAYERNORM,
                batch_id,
                execution_time.mlp_norm_time,  # 存储注意力后层归一化时间
            )

        self._push_metric(
            OperationMetrics.PIPELINE_SEND_RECV,
            batch_id,
            execution_time.pipeline_parallel_communication_time,  # 存储流水线并行通信时间
        )
        self._push_metric(
            CpuOperationMetrics.SCHEDULE, batch_id, execution_time.schedule_time  # 存储调度时间
        )
        self._push_metric(
            CpuOperationMetrics.SAMPLER_E2E, batch_id, execution_time.sampler_e2e_time  # 存储采样器端到端时间
        )
        self._push_metric(
            CpuOperationMetrics.PREPARE_INPUTS_E2E,
            batch_id,
            execution_time.prepare_inputs_e2e_time,  # 存储准备输入端到端时间
        )
        self._push_metric(
            CpuOperationMetrics.MODEL_EXECUTION_E2E,
            batch_id,
            execution_time.model_time_ms,  # 存储模型执行端到端时间
        )
        self._push_metric(
            CpuOperationMetrics.PROCESS_MODEL_OUTPUTS,
            batch_id,
            execution_time.process_model_outputs_time,  # 存储处理模型输出时间
        )
        self._push_metric(
            CpuOperationMetrics.RAY_COMM_TIME, batch_id, execution_time.ray_comm_time  # 存储Ray通信时间
        )

    @if_write_metrics
    def on_batch_stage_end(
        self, batch_stage: BatchStage, time: float, replica_id: int, stage_id: int
    ) -> None:
        if not self._config.store_utilization_metrics:  # 如果未配置存储利用率指标，则返回
            return
        self._replica_busy_time[replica_id - 1][stage_id - 1].put(time, 0)  # 复位副本忙时间
        self._replica_mfu[replica_id - 1][stage_id - 1].put(time, 0)  # 复位副本MFU