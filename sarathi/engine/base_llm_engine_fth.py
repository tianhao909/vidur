import copy  # 导入copy模块
import math  # 导入math模块
import time  # 导入time模块
from functools import partial  # 从functools导入partial函数
from typing import Any, Dict, List, Optional, Tuple  # 从typing导入一些类型

import zmq  # 导入zmq模块

from sarathi.config import ModelConfig, SystemConfig  # 从sarathi.config模块导入ModelConfig和SystemConfig类
from sarathi.core.datatypes.comm_info import CommInfo  # 从sarathi.core.datatypes.comm_info模块中导入CommInfo类
from sarathi.core.datatypes.request_output import RequestOutput  # 从sarathi.core.datatypes.request_output模块中导入RequestOutput类
from sarathi.core.datatypes.sampling_params import SamplingParams  # 从sarathi.core.datatypes.sampling_params模块中导入SamplingParams类
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs  # 从sarathi.core.datatypes.scheduler_output模块中导入SchedulerOutputs类
from sarathi.core.datatypes.sequence import SamplerOutputs, Sequence, SequenceMetadata  # 从sarathi.core.datatypes.sequence模块中导入SamplerOutputs, Sequence, SequenceMetadata类
from sarathi.core.datatypes.step_inputs import StepInputs  # 从sarathi.core.datatypes.step_inputs模块中导入StepInputs类
from sarathi.core.scheduler.scheduler_registry import SchedulerRegistry  # 从sarathi.core.scheduler.scheduler_registry模块中导入SchedulerRegistry类
from sarathi.core.sequence_manager.engine_sequence_manager import EngineSequenceManager  # 从sarathi.core.sequence_manager.engine_sequence_manager模块中导入EngineSequenceManager类
from sarathi.engine.ray_utils import RayWorker, initialize_cluster, ray  # 从sarathi.engine.ray_utils模块中导入RayWorker, initialize_cluster, ray
from sarathi.logger import init_logger  # 从sarathi.logger模块中导入init_logger函数
from sarathi.metrics.constants import CpuOperationMetrics  # 从sarathi.metrics.constants模块中导入CpuOperationMetrics
from sarathi.metrics.cpu_timer import CpuTimer  # 从sarathi.metrics.cpu_timer模块中导入CpuTimer类
from sarathi.metrics.metrics_store import MetricsStore  # 从sarathi.metrics.metrics_store模块中导入MetricsStore类
from sarathi.transformers_utils.tokenizer import get_tokenizer  # 从sarathi.transformers_utils.tokenizer模块中导入get_tokenizer函数
from sarathi.utils import Counter, get_ip, unset_cuda_visible_devices  # 从sarathi.utils模块中导入Counter, get_ip, unset_cuda_visible_devices
from sarathi.utils.threading_utils import synchronized  # 从sarathi.utils.threading_utils模块中导入synchronized

logger = init_logger(__name__)  # 初始化日志记录器

_MAX_WORKER_CONCURRENCY = 1  # 定义最大并发工作数为1

ModelParallelRank = Tuple[int, int]  # 定义模型并行排名的类型为元组，包含两个整数


class BaseLLMEngine:  # 定义一个大语言模型引擎基类
    """An LLM engine that receives requests and generates texts.

    这是一个接收请求并生成文本的大语言模型引擎。

    This is the main class for the Sarathi engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    这是Sarathi引擎的主要类。它接收来自客户端的请求，并从大语言模型生成文本。包括一个分词器、语言模型（可能分布在多个GPU上）以及为中间状态分配的GPU内存空间（也称为KV缓存）。该类利用迭代级别调度和高效内存管理来最大化服务吞吐量。

    Args:
        config; System Config: The system configuration for the engine.

        参数：
        config; System Config: 引擎的系统配置。
    """

    def __init__(
        self,
        config: SystemConfig,  # 初始化时接受一个系统配置
    ) -> None:  # 返回类型为空
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={config.model_config.model!r}, "
            f"dtype={config.model_config.dtype}, "
            f"tensor_parallel_size={config.parallel_config.tensor_parallel_size}, "
            f"pipeline_parallel_size={config.parallel_config.pipeline_parallel_size}, "
            f"seed={config.model_config.seed})"
        )  # 记录初始化引擎的配置信息
        # TODO(woosuk): Print more configs in debug mode.
        # TODO: 在调试模式下打印更多的配置信息。

        self.config = config  # 将配置存储在实例变量中
        self._verify_args()  # 验证传入的参数

        self.tokenizer = get_tokenizer(
            config.model_config.model,
            trust_remote_code=config.model_config.trust_remote_code,
            revision=config.model_config.revision,
        )  # 获取分词器

        self.seq_manager = EngineSequenceManager(self.tokenizer, config)  # 初始化引擎序列管理器
        self.seq_counter = Counter()  # 初始化序列计数器

        self.metrics_store = MetricsStore.get_or_create_instance(
            config.replica_config,
            config.model_config,
            config.metrics_config,
        )  # 获取或创建指标存储实例

        self.worker_map: Dict[ModelParallelRank, int] = {}  # 初始化工作者映射

        # Initialize the cluster.
        # 初始化集群。
        initialize_cluster()  # 调用初始化集群函数

        # Create the parallel GPU workers.
        # 创建并行GPU工作者。
        self._init_workers_ray()  # 初始化Ray工作者

        # Setup ZMQ communication channels
        # 设置ZMQ通信通道
        self._init_zmq_sockets()  # 初始化ZMQ套接字

        # Profile the memory usage and initialize the cache.
        # 分析内存使用情况并初始化缓存。
        self._init_cache()  # 初始化缓存

        # Initialize the worker map.
        # 初始化工作者映射。
        self._init_worker_map()  # 初始化工作者映射

        self.mark_initial_memory_profiling_done()

        # Create the scheduler.
        # 创建调度器。
        self.scheduler = SchedulerRegistry.get(
            config.scheduler_config.get_type(),
            config.model_config,
            config.scheduler_config,
            config.cache_config,
            config.parallel_config,
        )  # 从调度器注册表中获取调度器实例

        self._scheduler_timer = CpuTimer(CpuOperationMetrics.SCHEDULE)  # 初始化调度器计时器
        self._process_model_outputs_timer = CpuTimer(
            CpuOperationMetrics.PROCESS_MODEL_OUTPUTS
        )  # 初始化模型输出处理计时器

        self.new_seqs: List[Sequence] = []  # 初始化新序列列表

        self._run_workers("wait_till_ready")  # 执行工作者等待就绪的方法

    def _init_zmq_sockets(self):  # 初始化ZMQ套接字
        self.zmq_context = zmq.Context()  # 创建ZMQ上下文
        self.enqueue_socket = self.zmq_context.socket(zmq.PUB)  # 创建一个发布套接字
        self.enqueue_socket.bind(f"tcp://*:{self.comm_info.enqueue_socket_port}")  # 绑定发布套接字到特定端口
        self.output_socket = self.zmq_context.socket(zmq.PULL)  # 创建一个拉取套接字
        self.output_socket.bind(f"tcp://*:{self.comm_info.output_socket_port}")  # 绑定拉取套接字到特定端口

    def _validate_parallel_config(self) -> None:  # 验证并行配置
        assert self.config.parallel_config.pipeline_parallel_size == 1  # 确保流水线并行大小为1

    def _get_worker_impl(self):  # 初始化工作者实现
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # 延迟导入Worker以避免在设置CUDA_VISIBLE_DEVICES之前导入torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from sarathi.worker.base_worker import (
            BaseWorker,  # pylint: disable=import-outside-toplevel
        )  # 从sarathi.worker.base_worker模块导入BaseWorker

        return BaseWorker  # 返回BaseWorker类

    def _init_workers_ray(self, **ray_remote_kwargs):  # 初始化Ray工作者
        resource_mapping = self.config.replica_config.get_resource_mapping(
            self.config.parallel_config.world_size
        )  # 获取资源映射
        logger.info(f"Starting workers with resource mapping: {resource_mapping}")  # 记录开始工作的资源映射

        self.workers: List[RayWorker] = []  # 初始化工作者列表

        unset_cuda_visible_devices()  # 取消设置CUDA可见设备

        driver_ip = None  # 初始化driver_ip为None
        for rank, (node_ip, _) in enumerate(resource_mapping):  # 遍历资源映射中的节点
            worker_class = ray.remote(
                num_cpus=1,
                # num_gpus=1, # we don't use ray for managing GPUs
                **ray_remote_kwargs,
            )(RayWorker)  # 创建RayWorker的Ray远程类

            if node_ip:
                worker_class = worker_class.options(
                    max_concurrency=_MAX_WORKER_CONCURRENCY,
                    resources={
                        node_ip: 0.01,
                    },
                )  # 如果有节点IP则设置资源选项
            else:
                worker_class = worker_class.options(
                    max_concurrency=_MAX_WORKER_CONCURRENCY,
                )  # 否则仅设置最大并发

            if rank == 0:
                if node_ip:
                    # remove node: prefix
                    # 移除节点前缀
                    driver_ip = node_ip.split(":")[1]  # 如果是第一个节点, 提取IP
                else:
                    driver_ip = get_ip()  # 获取自己的IP

            worker = worker_class.remote(self.config.model_config.trust_remote_code)  # 初始化远程Worker

            self.workers.append(worker)  # 将Worker添加到列表

        self.comm_info = CommInfo(driver_ip)  # 初始化通信信息

        # Initialize torch distributed process group for the workers.
        # 为工作者初始化torch分布式进程组。
        config = copy.deepcopy(self.config)  # 深拷贝配置
        config.metrics_config = self.metrics_store.get_config_for_worker()  # 为工作者获取量度配置

        worker_impl = self._get_worker_impl()  # 获取工作者实现

        for rank, worker in enumerate(self.workers):  # 遍历工作者
            local_rank = resource_mapping[rank][1]  # 获取本地路径
            promise = worker.init_worker.remote(
                lambda rank=rank, local_rank=local_rank: worker_impl(
                    config,
                    local_rank,
                    rank,
                    self.comm_info,
                )
            )  # 异步初始化工作者
            ray.get(promise)  # 获取异步结果

        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )  # 初始化模型

    def _verify_args(self) -> None:  # 验证参数
        self._validate_parallel_config()  # 验证并行配置
        self.config.model_config.verify_with_parallel_config(
            self.config.parallel_config
        )  # 验证模型配置

    def _init_cache(self) -> None:  # 初始化缓存
        """Profiles the memory usage and initializes the KV cache.
        分析内存使用情况并初始化KV缓存。
        """
        # Get the maximum number of blocks that can be allocated on GPU.
        # 获取可在GPU上分配的最大块数量。
        num_gpu_blocks_across_workers = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.config.cache_config.block_size,
            gpu_memory_utilization=self.config.worker_config.gpu_memory_utilization,
        )  # 分析所有工作者上的可用块数量

        # Since we use a shared centralized controller, we take the minimum
        # 因为我们使用一个共享的集中控制器，我们取
        # number of blocks across all workers to make sure all the memory
        # 所有工作者的块数量的最小值以确保所有内存
        # operators can be applied to all workers.
        # 操作符能够应用于所有工作者。
        num_gpu_blocks = min(num_gpu_blocks_across_workers)  # 获取所有工作者的最小块数量
        # FIXME(woosuk): Change to debug log.
        # FIXME: 改为调试日志。
        logger.info(f"# GPU blocks: {num_gpu_blocks}")  # 记录GPU块数量

        if num_gpu_blocks <= 0:  # 如果GPU块数量小于等于0
            raise ValueError(
                "No available memory for the cache blocks. "
                "缓存块没有可用的内存。 "
                "Try increasing `gpu_memory_utilization` when "
                "尝试增加gpu_memory_utilization。 "
                "initializing the engine."
                "初始化引擎时。"
            )
        max_blocks_per_request = math.ceil(
            self.config.model_config.max_model_len / self.config.cache_config.block_size
        )  # 计算每个请求的最大块数量
        if num_gpu_blocks < max_blocks_per_request:  # 如果GPU块数量小于每个块请求最大块数量
            raise ValueError(
                f"Not enough available memory to schedule a request will maximum allowed length {self.config.model_config.max_model_len}. "
                f"没有足够的可用内存来安排具有最大允许长度的请求 {self.config.model_config.max_model_len}. "
                f"Need {max_blocks_per_request}, available {num_gpu_blocks} gpu blocks. "
                f"需要 {max_blocks_per_request}, 可用 {num_gpu_blocks} GPU块。 "
                f"Try decreasing `max_batch_size`, `max_model_len`."
                f"尝试减少max_batch_size, max_model_len."
            )
        self.config.cache_config.num_gpu_blocks = num_gpu_blocks  # 设置GPU块数量

        # Initialize the cache.
        # 初始化缓存。
        self._run_workers(
            "init_cache_engine",
            cache_config=self.config.cache_config,
            get_all_outputs=True,
        )  # 初始化缓存引擎

    def _init_worker_map(self) -> None:  # 初始化工作者地图
        model_parallel_ranks = self._run_workers(
            "get_model_parallel_ranks",
            get_all_outputs=True,
        )  # 获取模型并行排名

        self.worker_map = {mp_rank: i for i, mp_rank in enumerate(model_parallel_ranks)}  # 创建工作者地图

    def _on_step_completed(
        self,
        scheduler_outputs: SchedulerOutputs,
        ignored_seqs: List[SequenceMetadata],
        seq_metadata_list: List[SequenceMetadata],
        sampler_outputs: Optional[SamplerOutputs],
        start_time: float,
    ) -> List[RequestOutput]:  # 定义步骤完成时的动作
        with self._process_model_outputs_timer:
            self.seq_manager.on_step_completed(
                scheduler_outputs,
                sampler_outputs,
            )  # 更新序列管理器
            self.scheduler.on_step_completed()  # 更新调度器

        end_time = time.perf_counter()  # 获取结束时间

        self.metrics_store.on_batch_end(
            seq_metadata_list=seq_metadata_list,
            scheduler_outputs=scheduler_outputs,
            batch_start_time=start_time,
            batch_end_time=end_time,
        )  # 更新指标存储
        all_request_outputs = self.seq_manager.generate_request_outputs(
            ignored_seqs, seq_metadata_list
        )  # 生成所有请求的输出
        return all_request_outputs  # 返回所有请求的输出

    def get_model_config(self) -> ModelConfig:  # 获取模型配置
        return self.config.model_config  # 返回配置中的模型配置

    def add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        seq_id: Optional[str] = None,
    ) -> None:  # 添加请求
        """Add a request to the engine's request pool.

        将请求添加到引擎的请求池中。

        The request is added to the request pool and will be processed by the
        请求将被添加到请求池并由调度器处理
        scheduler as `engine.step()` is called. The exact scheduling policy is
        当调用engine.step()时。精确的调度策略是
        determined by the scheduler.

        由调度器决定。

        Args:
            seq_id: The unique ID of the request.
            序列ID：请求的唯一ID。
            prompt: The prompt string. Can be None if prompt_token_ids is
            提示：提示字符串。如果提供了prompt_token_ids，可以为None。
                provided.
                已提供的。
            sampling_params: The sampling parameters for text generation.
            采样参数：生成文本的采样参数。
            prompt_token_ids: The token IDs of the prompt. If None, we
            提示的标记ID。如果为None，我们
                use the tokenizer to convert the prompts to token IDs.
                使用分词器将提示转换为标记ID。
            arrival_time: The arrival time of the request. If None, we use
            到达时间：请求的到达时间。如果为None，我们将使用
                the current time.
                当前时间。
        """
        if arrival_time is None:
            arrival_time = time.monotonic()  # 如果没有提供到达时间，使用当前时间

        if not seq_id:
            seq_id = str(next(self.seq_counter))  # 生成一个新的序列ID

        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)  # 使用分词器编码提示

        # Create the sequences.
        # 创建序列。
        block_size = self.config.cache_config.block_size  # 获取块大小
        eos_token_id = self.tokenizer.eos_token_id  # 获取EOS标记ID

        seq = Sequence(
            seq_id,
            prompt,
            prompt_token_ids,
            block_size,
            eos_token_id,
            arrival_time,
            sampling_params,
        )  # 创建新的序列
        # Add the sequence to the scheduler.
        # 将序列添加到调度器。
        self.seq_manager.add_seq(seq)  # 将序列添加到管理器
        # we create a copy of the seq so that the workers
        # 我们创建序列的一个副本，以便工作者
        # receive an unmodified version of the seq
        # 接收一个未修改的序列版本
        # which is unaffected by the engine's actions
        # 这不受引擎的操作影响
        self._append_new_seq(copy.deepcopy(seq))  # 添加序列副本
        self.scheduler.add_seq(seq)  # 将序列添加到调度器
        self.metrics_store.on_request_arrival(seq)  # 在请求到达时更新指标存储

    @synchronized
    def _append_new_seq(self, seq: Sequence) -> None:  # 添加新序列
        self.new_seqs.append(seq)  # 将新序列添加到列表

    @synchronized
    def _get_new_seqs(
        self,
    ) -> List[Sequence]:  # 获取新序列
        new_seqs = self.new_seqs  # 获取新序列列表
        self.new_seqs = []  # 重置新序列列表
        return new_seqs  # 返回新序列列表

    def get_num_unfinished_requests(self) -> int:  # 获取未完成请求的数量
        """Gets the number of unfinished requests.
        获取未完成请求的数量。
        """
        return self.scheduler.get_num_unfinished_seqs()  # 返回未完成请求数量

    def has_unfinished_requests(self) -> bool:  # 检查是否有未完成请求
        """Returns True if there are unfinished requests.
        如果有未完成的请求，则返回True。
        """
        return self.scheduler.has_unfinished_seqs()  # 返回未完成请求的状态

    def step(self) -> List[RequestOutput]:  # 执行一步
        """Performs one decoding iteration and returns newly generated results.

        执行一次解码迭代并返回新生成的结果。

        This function performs one decoding iteration of the engine. It first
        该函数执行引擎的一次解码迭代。 它首先激活调度程序

        schedules the sequences to be executed in the next iteration.
        将序列安排在下一次迭代中执行。
        Then, it executes the model and updates the scheduler with the model outputs.
        然后，执行模型并使用模型输出更新调度器。
        Finally, it decodes the sequences and returns the newly generated results.
        最后，它解码序列并返回新生成的结果。
        """
        start_time = time.perf_counter()  # 获取开始时间

        with self._scheduler_timer:
            scheduler_outputs = self.scheduler.schedule()  # 执行调度
        
        if scheduler_outputs.is_empty():  # 如果调度输出为空
            return []  # 返回空列表

        ignored_seqs, seq_metadata_list = self.seq_manager.on_schedule(
            scheduler_outputs
        )  # 调度序列并获取忽略的序列和序列元数据列表

        self.enqueue_socket.send_pyobj(
            StepInputs(
                scheduler_outputs,
                new_seqs=self._get_new_seqs(),
            )
        )  # 发送步骤输入对象
        sampler_outputs = self.output_socket.recv_pyobj()  # 接收采样器输出

        return self._on_step_completed(
            scheduler_outputs,
            ignored_seqs,
            seq_metadata_list,
            sampler_outputs,
            start_time,
        )  # 在步骤完成时执行操作返回请求输出

    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        ignore_output: bool = False,
        **kwargs,
    ) -> Any:  # 在工作者中运行给定的方法
        """Runs the given method on all workers.
        在所有工作者中运行给定的方法。
        """
        all_outputs = []  # 初始化所有输出
        for worker in self.workers:  # 遍历工作者
            executor = partial(worker.execute_method.remote, method)  # 创建部分应用

            output = executor(*args, **kwargs)  # 执行异步方法
            all_outputs.append(output)  # 添加输出到列表

        if ignore_output:  # 如果忽略输出
            return  # 返回

        while True:  # 不断尝试获取异步结果
            try:
                all_outputs = ray.get(all_outputs, timeout=0)  # 尝试获取异步结果
                break  # 获取成功后退出循环
            except ray.exceptions.GetTimeoutError:  # 如果超时则重新尝试
                time.sleep(0)  # 短暂休眠继续尝试
                continue  # 继续循环

        if get_all_outputs:  # 如果需要所有输出
            return all_outputs  # 返回所有输出

        # Make sure all workers have the same results.
        # 确保所有工人都有相同的结果。
        output = all_outputs[0]  # 获取第一个输出
        for other_output in all_outputs[1:]:  # 确保其他输出与第一个相同
            assert output == other_output  # 断言两个输出相同
        return output  # 返回输出

    def _run_worker(
        self,
        model_parallel_rank: ModelParallelRank,
        method: str,
        *args,
        **kwargs,
    ) -> Any:  # 在特定工作者中运行给定的方法
        """Runs the given method on all workers.
        在所有工作者中运行给定的方法。
        """
        worker = self.workers[self.worker_map[model_parallel_rank]]  # 获取指定的工作者
        executor = partial(worker.execute_method.remote, method)  # 创建部分应用

        output = executor(*args, **kwargs)  # 执行异步方法

        while True:  # 不断尝试获取异步结果
            try:
                output = ray.get(output, timeout=0)  # 尝试获取异步结果
                break  # 获取成功后退出循环
            except ray.exceptions.GetTimeoutError:  # 如果超时则重新尝试
                time.sleep(0)  # 短暂休眠继续尝试
                continue  # 继续循环

        return output  # 返回输出

    def plot_metrics(self) -> None:  # 绘制指标
        self.metrics_store.plot()  # 绘制存储的指标

    def pull_worker_metrics(self) -> None:  # 获取工作者的指标
        worker_metrics = self._run_workers(
            "get_metrics_store",
            get_all_outputs=True,
        )  # 获取所有工作的指标存储
        for worker_metric in worker_metrics:  # 合并每个工作者的指标
            self.metrics_store.merge(worker_metric)  # 合并指标存储

    def mark_initial_memory_profiling_done(self):
        self.metrics_store.mark_initial_memory_profiling_done()  # 标记初始内存分析完成
        self._run_workers("mark_initial_memory_profiling_done", get_all_outputs=True)  # 标记所有工作者初始内存分析完成

    def reset_metrics(self) -> None:  # 重置指标
        self.scheduler.reset_state()  # 重置调度器状态
        self.metrics_store.reset()  # 重置指标存储
        self._run_workers("reset_metrics", get_all_outputs=True)  # 重置所有工作者的指标

    def start_profiling(self) -> None:  # 开始分析
        self._run_workers("start_profiling")  # 开始分析所有工作者

    def stop_profiling(self) -> None:  # 停止分析
        self._run_workers("stop_profiling")  # 停止分析所有工作者

    def get_metric_store(self) -> MetricsStore:  # 获取指标存储
        return self.metrics_store  # 返回指标存储
