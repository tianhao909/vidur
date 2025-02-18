from math import ceil  # 导入ceil函数，用于向上取整
from typing import List  # 导入List类型，用于类型注解

import numpy as np  # 导入numpy库，用于数值计算
import sarathi.metrics.cuda_timer  # 导入sarathi库中的cuda_timer模块，用于CUDA时间测量
import torch  # 导入PyTorch库，用于深度学习计算

from vidur.profiling.common.cuda_timer import CudaTimer  # 导入自定义的CudaTimer类，用于CUDA时间测量

# 将sarathi库中的CudaTimer类替换为自定义的CudaTimer类，实现猴子补丁
sarathi.metrics.cuda_timer.CudaTimer = CudaTimer

from sarathi.config import ParallelConfig  # 导入ParallelConfig类，用于并行配置
from sarathi.model_executor.attention import (  # 导入注意力相关的类和函数
    AttentionBackend,  # 注意力后端枚举类
    get_attention_wrapper,  # 获取注意力包装器的函数
    set_attention_backend,  # 设置注意力后端的函数
)

from vidur.profiling.attention.attention_input import AttentionInput  # 导入AttentionInput类，用于注意力输入
from vidur.profiling.attention.sequence_proxy import SequenceMetadataProxy  # 导入SequenceMetadataProxy类，用于序列元数据代理
from vidur.profiling.common.model_config import ModelConfig  # 导入ModelConfig类，用于模型配置
from vidur.profiling.common.timer_stats_store import TimerStatsStore  # 导入TimerStatsStore类，用于存储时间统计信息

WARMUP_STEPS = 2  # 定义预热步数，用于稳定性能测试
ACTIVE_STEPS = 5  # 定义活跃步数，用于实际性能测试


class AttentionWrapper:  # 定义注意力包装器类
    def __init__(  # 初始化方法
        self,
        model_config: ModelConfig,  # 模型配置对象
        parallel_config: ParallelConfig,  # 并行配置对象
        max_num_blocks: int,  # 最大块数量
        max_model_len: int,  # 最大模型长度
        block_size: int,  # 块大小
        attention_backend: AttentionBackend,  # 注意力后端
        dtype: torch.dtype,  # 数据类型
    ):
        self.time_stats_store = TimerStatsStore(profile_method="kineto")  # 初始化时间统计存储对象

        self._model_config = model_config  # 存储模型配置
        self._parallel_config = parallel_config  # 存储并行配置
        self._dtype = dtype  # 存储数据类型
        self._device = torch.device("cuda")  # 设置设备为CUDA

        self._max_model_len = max_model_len  # 存储最大模型长度
        self._n_worker_q_heads = self._model_config.get_num_q_heads(  # 获取每个worker的查询头数量
            self._parallel_config
        )
        self._n_worker_kv_heads = self._model_config.get_num_kv_heads(  # 获取每个worker的键值头数量
            self._parallel_config
        )
        self._head_dim = self._model_config.get_head_size()  # 获取每个头的维度

        self._block_size = block_size  # 存储块大小

        self._attention_backend = attention_backend  # 存储注意力后端
        set_attention_backend(attention_backend)  # 设置注意力后端
        get_attention_wrapper().init(  # 初始化注意力包装器
            self._model_config,
            self._parallel_config,
            self._block_size,
            self._device,
        )
        self._max_blocks_per_sequence = ceil(max_model_len / self._block_size)  # 计算每个序列的最大块数
        # 创建并复用大的KV张量
        self.max_num_blocks = max_num_blocks  # 存储最大块数量
        self.kv_cache = get_attention_wrapper().get_cache_block(  # 获取缓存块
            self.max_num_blocks, dtype=self._dtype, device=self._device
        )

    def _get_input_tensors(  # 获取输入张量的方法
        self,
        attention_input: AttentionInput,  # 注意力输入对象
    ):
        num_tokens_per_seq = (  # 计算每个序列的token数量
            attention_input.prefill_chunk_size if attention_input.is_prefill else 1
        )
        batch_size = attention_input.batch_size  # 获取批量大小
        query = torch.randn(  # 随机生成查询张量
            batch_size * num_tokens_per_seq,
            self._n_worker_q_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        key = torch.randn(  # 随机生成键张量
            batch_size * num_tokens_per_seq,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        value = torch.randn(  # 随机生成值张量
            batch_size * num_tokens_per_seq,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        # 创建与AttentionInput对应的SequenceMetadataProxy对象列表
        seq_metadata_list: List[SequenceMetadataProxy] = []
        for _ in range(attention_input.batch_size):  # 遍历批量大小
            num_blocks = ceil(  # 计算块数量
                (num_tokens_per_seq + attention_input.kv_cache_size) / self._block_size
            )
            # TODO(nitinkedia7): 调查为什么high=max_num_blocks会导致CUDA非法内存访问
            seq_metadata = SequenceMetadataProxy(  # 创建序列元数据代理对象
                is_prompt=attention_input.is_prefill,  # 是否为提示阶段
                total_len=num_tokens_per_seq + attention_input.kv_cache_size,  # 总长度
                processed_len=attention_input.kv_cache_size,  # 已处理长度
                block_table=np.random.default_rng()  # 随机生成块表
                .integers(low=0, high=self.max_num_blocks - 1, size=num_blocks)
                .tolist(),
            )
            seq_metadata_list.append(seq_metadata)  # 添加到列表中
        return seq_metadata_list, query, key, value, self.kv_cache  # 返回生成的张量和元数据

    @torch.inference_mode()  # 推理模式装饰器，禁用梯度计算
    def profile(  # 性能分析方法
        self,
        attention_input: AttentionInput,  # 注意力输入对象
    ):
        # 批量大小在预填充阶段始终为1，在解码阶段可以不同
        assert attention_input.is_valid(self._max_model_len)  # 确保输入有效

        seq_metadata_list, query, key, value, kv_cache = self._get_input_tensors(  # 获取输入张量
            attention_input,
        )
        get_attention_wrapper().begin_forward(seq_metadata_list)  # 开始前向传播

        for _ in range(WARMUP_STEPS):  # 预热步骤
            get_attention_wrapper().forward(query, key, value, kv_cache)  # 前向传播
        torch.cuda.synchronize()  # 同步CUDA设备

        self.time_stats_store.clear_stats()  # 清除时间统计信息

        for _ in range(ACTIVE_STEPS):  # 活跃步骤
            get_attention_wrapper().forward(query, key, value, kv_cache)  # 前向传播
        torch.cuda.synchronize()  # 同步CUDA设备

        get_attention_wrapper().end_forward()  # 结束前向传播

        return {  # 返回性能分析结果
            "time_stats": self.time_stats_store.get_stats(),  # 时间统计信息
            "n_embd": self._model_config.embedding_dim,  # 嵌入维度
            "n_q_head": self._model_config.num_q_heads,  # 查询头数量
            "n_kv_head": self._model_config.num_kv_heads,  # 键值头数量
            "block_size": self._block_size,  # 块大小
            "num_tensor_parallel_workers": self._parallel_config.tensor_parallel_size,  # 张量并行worker数量
            "max_model_len": self._max_model_len,  # 最大模型长度
            "batch_size": attention_input.batch_size,  # 批量大小
            "prefill_chunk_size": attention_input.prefill_chunk_size,  # 预填充块大小
            "kv_cache_size": attention_input.kv_cache_size,  # KV缓存大小
            "is_prefill": attention_input.is_prefill,  # 是否为预填充阶段
            "attention_backend": self._attention_backend,  # 注意力后端
        }
