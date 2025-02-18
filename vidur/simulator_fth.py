import atexit
import heapq
import json
from typing import List

from vidur.config import SimulationConfig
from vidur.entities import Cluster
from vidur.events import BaseEvent, RequestArrivalEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.request_generator import RequestGeneratorRegistry
from vidur.scheduler import BaseGlobalScheduler, GlobalSchedulerRegistry

logger = init_logger(__name__)  # 初始化日志记录器

class Simulator:

    # 在 Python 中，冒号用于类型注释，也称为类型提示。在你的代码中，self._config: SimulationConfig = config 中的 : SimulationConfig 是用来表示 self._config 是 SimulationConfig 类型的变量。
    # 类型提示是 Python 3.5 引入的功能，主要是为了提高代码的可读性和可维护性。它有助于开发者和代码编辑器了解变量应该是什么类型，这在大型项目中尤其有用。它们也可以与静态类型检查工具（如 mypy）一起使用，以便在运行代码之前检测到类型错误。
    # 至于是否可以不加类型提示，答案是可以的。Python 是动态类型语言，因此类型声明并不是强制性的，代码在没有类型提示的情况下也会正常运行。不过，添加类型提示会使代码更具可读性，从而在开发和维护阶段提供一定的帮助。
    def __init__(self, config: SimulationConfig) -> None:
        self._config: SimulationConfig = config  # 保存配置 类型注释

        self._time = 0  # 初始化模拟时间
        self._terminate = False  # 初始化终止标志
        self._time_limit = self._config.time_limit  # 获取时间限制
        if not self._time_limit:
            self._time_limit = float("inf")  # 如果没有时间限制，则设置为无限大

        self._event_queue = []  # 初始化事件队列

        self._event_trace = []  # 初始化事件跟踪记录
        self._event_chrome_trace = []  # 初始化Chrome事件跟踪记录

        self._cluster = Cluster(
            self._config.cluster_config,
            self._config.metrics_config,
            self._config.request_generator_config,
        )  # 初始化集群
        self._metric_store = MetricsStore(self._config)  # 初始化度量存储
        self._request_generator = RequestGeneratorRegistry.get(
            self._config.request_generator_config.get_type(),
            self._config.request_generator_config,
        )  # 获取请求生成器
        self._scheduler = GlobalSchedulerRegistry.get(
            self._config.cluster_config.global_scheduler_config.get_type(),
            self._config,
            self._cluster.replicas,
        )  # 获取调度器



        self._init_event_queue()  # 初始化事件队列
        atexit.register(self._write_output)  # 注册退出时写输出的函数

    @property
    def scheduler(self) -> BaseGlobalScheduler:
        return self._scheduler  # 返回全局调度器

    @property
    def metric_store(self) -> MetricsStore:
        return self._metric_store  # 返回度量存储

    def run(self) -> None:
        logger.info(
            f"Starting simulation with cluster: {self._cluster} and {len(self._event_queue)} requests"
        )  # 记录模拟开始的信息

        while self._event_queue and not self._terminate:
            _, event = heapq.heappop(self._event_queue)  # 从事件队列中弹出优先级最高的事件
            self._set_time(event._time)  # 设置当前时间为事件的时间
            new_events = event.handle_event(self._scheduler, self._metric_store)  # 处理事件并获取新事件
            self._add_events(new_events)  # 将新事件加入队列

            if self._config.metrics_config.write_json_trace:
                self._event_trace.append(event.to_dict())  # 如果开启JSON跟踪，记录事件信息

            if self._config.metrics_config.enable_chrome_trace:
                chrome_trace = event.to_chrome_trace()  # 生成Chrome跟踪信息
                if chrome_trace:
                    self._event_chrome_trace.append(chrome_trace)  # 如果有Chrome跟踪信息，则记录

        assert self._scheduler.is_empty() or self._terminate  # 确保调度器空或已终止

        logger.info(f"Simulation ended at: {self._time}s")  # 记录模拟结束的信息

    def _write_output(self) -> None:
        logger.info("Writing output")  # 记录写输出的操作
        self._metric_store.plot()  # 绘制度量图
        logger.info("Metrics written")  # 记录度量已写

        if self._config.metrics_config.write_json_trace:
            self._write_event_trace()  # 如果启用JSON跟踪，写入事件跟踪
            logger.info("Json event trace written")  # 记录JSON事件跟踪已写

        if self._config.metrics_config.enable_chrome_trace:
            self._write_chrome_trace()  # 如果启用Chrome跟踪，写入Chrome跟踪
            logger.info("Chrome event trace written")  # 记录Chrome事件跟踪已写

    def _add_event(self, event: BaseEvent) -> None:
        heapq.heappush(self._event_queue, (event._priority_number, event))  # 将事件加入优先队列

    def _add_events(self, events: List[BaseEvent]) -> None:
        for event in events:
            self._add_event(event)  # 批量添加事件

    def _init_event_queue(self) -> None:
        requests = self._request_generator.generate()  # 请求生成器生成请求

        for request in requests:
            self._add_event(RequestArrivalEvent(request.arrived_at, request))  # 为每个请求生成到达事件并添加至队列

    def _set_time(self, time: float) -> None:
        self._time = time  # 设置当前时间
        if self._time > self._time_limit:
            logger.info(
                f"Time limit reached: {self._time_limit}s terminating the simulation."
            )  # 超过时间限制时记录日志
            self._terminate = True  # 设置终止标志

    def _write_event_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/event_trace.json"
        with open(trace_file, "w") as f:
            json.dump(self._event_trace, f)  # 写入事件跟踪至JSON文件

    def _write_chrome_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/chrome_trace.json"

        chrome_trace = {"traceEvents": self._event_chrome_trace}  # 构造Chrome跟踪事件

        with open(trace_file, "w") as f:
            json.dump(chrome_trace, f)  # 写入Chrome跟踪至JSON文件