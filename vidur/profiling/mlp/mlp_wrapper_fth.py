import os

import sarathi.metrics.cuda_timer
import torch

from vidur.profiling.common.cuda_timer import CudaTimer

# 猴子补丁技术：将 CudaTimer 类替换为 sarathi 实现
# monkey patching the CudaTimer class to use the sarathi implementation
sarathi.metrics.cuda_timer.CudaTimer = CudaTimer

from sarathi.model_executor.weight_utils import initialize_dummy_weights

from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.common.timer_stats_store import TimerStatsStore
from vidur.profiling.mlp.mlp_impl import GPTModel
from vidur.profiling.utils import ProfileMethod
from vidur.profiling.utils.record_function_tracer import RecordFunctionTracer

WARMUP_STEPS = 2  # 预热步骤数
ACTIVE_STEPS = 20  # 活跃步骤数


class MlpWrapper:
    def __init__(
        self,
        model_config: ModelConfig,  # 模型配置
        num_tensor_parallel_workers: int,  # 张量并行的工作者数量
        profile_method: str,  # 轮廓分析方法
        rank: int,  # 排名
        output_dir: str,  # 输出目录
    ):
        super().__init__()

        self.timer_stats_store = TimerStatsStore(profile_method=profile_method)  # 计时器统计存储

        self.model_config = model_config  # 模型配置
        self.num_tensor_parallel_workers = num_tensor_parallel_workers  # 张量并行的工作者数量
        self.profile_method = profile_method  # 轮廓分析方法
        self.rank = rank  # 排名
        self.output_dir = output_dir  # 输出目录
        os.makedirs(f"{self.output_dir}/profiler_traces/", exist_ok=True)  # 创建输出目录 

        self.model = GPTModel(
            model_config,
            num_tensor_parallel_workers,
            (
                ACTIVE_STEPS
                if self.profile_method == ProfileMethod.RECORD_FUNCTION.value  # 检查是否使用记录功能方法
                else 1
            ),
        )
        initialize_dummy_weights(self.model)  # 初始化虚拟权重
        self.model = self.model.to(dtype=torch.float16).cuda().eval()  # 设置模型精度为 float16 并在 GPU 上执行

    @torch.inference_mode()
    def profile(self, num_tokens: int):  # 轮廓分析函数，输入为令牌数量
        vocab_range = self.model_config.vocab_size // self.num_tensor_parallel_workers  # 计算词汇范围
        input_ids = torch.randint(
            low=0,
            high=vocab_range,
            size=(num_tokens,),
            device="cuda",
            dtype=torch.long,
        )  # 生成随机输入ID
        positions = torch.arange(num_tokens, device="cuda", dtype=torch.long)  # 生成位置序列

        if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:
            # 运行模型一次不捕获图。这是为了确保捕获的图不包括初始基准测试的内核启动（例如，Triton 自动调优）。
            self.model(
                input_ids,
                positions,
            )
            torch.cuda.synchronize()  # 同步 CUDA 设备

            self.timer_stats_store.clear_stats()  # 清除计时器统计

            record_function_tracer = RecordFunctionTracer(self.output_dir)  # 记录功能跟踪器

            with record_function_tracer:
                self.model(
                    input_ids,
                    positions,
                )

            time_stats = record_function_tracer.get_operation_time_stats()  # 获取操作时间统计
        else:
            for _ in range(WARMUP_STEPS):
                self.model(
                    input_ids,
                    positions,
                )

            torch.cuda.synchronize()  # 同步 CUDA 设备

            self.timer_stats_store.clear_stats()  # 清除计时器统计

            for _ in range(ACTIVE_STEPS):
                self.model(
                    input_ids,
                    positions,
                )

            torch.cuda.synchronize()  # 同步 CUDA 设备

            time_stats = self.timer_stats_store.get_stats()  # 获取统计信息

        stats = {
            "time_stats": time_stats,  # 时间统计信息
            "n_head": self.model_config.num_q_heads,  # 查询头数
            "n_kv_head": self.model_config.num_kv_heads,  # 键值头数
            "n_embd": self.model_config.embedding_dim,  # 嵌入维度
            "n_expanded_embd": self.model_config.mlp_hidden_dim,  # 扩展后的嵌入维度
            "vocab_size": self.model_config.vocab_size,  # 词汇大小
            "use_gated_mlp": self.model_config.use_gated_mlp,  # 是否使用 gated MLP
            "num_tokens": num_tokens,  # 令牌数量
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,  # 张量并行的工作者数量
        }
        self.timer_stats_store.clear_stats()  # 清除计时器统计

        return stats  # 返回统计信息