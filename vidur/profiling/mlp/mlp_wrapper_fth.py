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
 
    @torch.inference_mode()  # 装饰器，禁用梯度计算，优化推理性能
    def profile(self, num_tokens: int):  # 定义profile方法，接受num_tokens参数
        vocab_range = self.model_config.vocab_size // self.num_tensor_parallel_workers  # 计算每个并行工作者的词汇范围
        input_ids = torch.randint(  # 生成随机的输入token ids
            low=0,  # 随机数的下限为0
            high=vocab_range,  # 随机数的上限为vocab_range
            size=(num_tokens,),  # 生成的张量大小为(num_tokens,)
            device="cuda",  # 张量放在GPU上
            dtype=torch.long,  # 数据类型为长整型
        )
        positions = torch.arange(num_tokens, device="cuda", dtype=torch.long)  # 生成位置id张量

        if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:  # 如果使用RECORD_FUNCTION方法进行profiling
            # Run the model once without capturing the graph.  # 运行模型一次，不捕捉计算图
            # This is to make sure that the captured graph does not include the  #
            # kernel launches for initial benchmarking (e.g., Triton autotune). # 这是为了确保捕获的图不包括初始基准测试的内核启动（例如，Triton 自动调优）
            self.model(  # 运行模型前向传播
                input_ids,  # 输入token ids
                positions,  # 位置ids
            )
            torch.cuda.synchronize()  # 同步CUDA操作

            self.timer_stats_store.clear_stats()  # 清空计时统计数据

            record_function_tracer = RecordFunctionTracer(self.output_dir)  # 初始化记录函数追踪器

            with record_function_tracer:  # 使用记录函数追踪器上下文
                self.model(  # 再次运行模型前向传播
                    input_ids,  # 输入token ids
                    positions,  # 位置ids
                )

            time_stats = record_function_tracer.get_operation_time_stats()  # 获取记录的操作时间统计
        else:  # 如果不是使用RECORD_FUNCTION方法
            for _ in range(WARMUP_STEPS):  # 进行预热步骤
                self.model(  # 运行模型前向传播
                    input_ids,  # 输入token ids
                    positions,  # 位置ids
                )

            torch.cuda.synchronize()  # 同步CUDA操作

            self.timer_stats_store.clear_stats()  # 清空计时统计数据

            for _ in range(ACTIVE_STEPS):  # 进行活跃的profiling步骤
                self.model(  # 运行模型前向传播
                    input_ids,  # 输入token ids
                    positions,  # 位置ids
                )

            torch.cuda.synchronize()  # 同步CUDA操作

            time_stats = self.timer_stats_store.get_stats()  # 获取计时统计数据

        stats = {  # 构建统计结果字典
            "time_stats": time_stats,  # 时间统计
            "n_head": self.model_config.num_q_heads,  # 查询头数
            "n_kv_head": self.model_config.num_kv_heads,  # 关键值头数
            "n_embd": self.model_config.embedding_dim,  # 嵌入维度
            "n_expanded_embd": self.model_config.mlp_hidden_dim,  # MLP隐藏层扩展维度
            "vocab_size": self.model_config.vocab_size,  # 词汇表大小
            "use_gated_mlp": self.model_config.use_gated_mlp,  # 是否使用有门控的MLP
            "num_tokens": num_tokens,  # token数量
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,  # 张量并行工作者数量
        }
        self.timer_stats_store.clear_stats()  # 清空计时统计数据

        return stats  # 返回统计结果
