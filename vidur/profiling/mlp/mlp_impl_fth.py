from typing import Optional  # 导入Optional类型，用于可选参数

import torch  # 导入PyTorch库，用于构建神经网络
from sarathi.model_executor.layers.activation import SiluAndMul  # 导入SiluAndMul激活函数
from sarathi.model_executor.layers.layernorm import RMSNorm  # 导入RMSNorm层归一化方法
from sarathi.model_executor.layers.rotary_embedding import get_rope  # 导入旋转位置编码（RoPE）生成函数
from sarathi.model_executor.parallel_utils.tensor_parallel.layers import (  # 导入分布式线性层
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

from vidur.profiling.common.cuda_timer import CudaTimer  # 导入CUDA计时器，用于性能分析
from vidur.profiling.common.model_config import ModelConfig  # 导入模型配置类

REUSE_MEMORY = True  # 定义是否重用内存的全局变量


class CausalSelfAttention(torch.nn.Module):  # 定义因果自注意力模块
    def __init__(self, config: ModelConfig, world_size: int):  # 初始化函数
        super().__init__()  # 调用父类构造函数
        assert config.embedding_dim % config.num_q_heads == 0  # 确保嵌入维度能被查询头数整除
        assert config.embedding_dim % world_size == 0  # 确保嵌入维度能被世界大小整除
        assert config.num_q_heads % world_size == 0  # 确保查询头数能被世界大小整除
        assert config.num_kv_heads % world_size == 0  # 确保键值头数能被世界大小整除

        self.head_dim = config.embedding_dim // config.num_q_heads  # 计算每个头的维度
        self.num_q_heads_per_worker = config.num_q_heads // world_size  # 每个worker的查询头数
        self.num_kv_heads_per_worker = config.num_kv_heads // world_size  # 每个worker的键值头数

        self.q_size = self.num_q_heads_per_worker * self.head_dim  # 查询向量的总大小
        self.kv_size = self.num_kv_heads_per_worker * self.head_dim  # 键值向量的总大小
        self.scaling = self.head_dim**-0.5  # 缩放因子，用于点积注意力

        self.qkv_proj = ColumnParallelLinear(  # 定义QKV投影层（列并行）
            config.embedding_dim,
            (config.num_q_heads + 2 * config.num_kv_heads) * self.head_dim,
            bias=config.use_bias or config.use_qkv_bias,  # 是否使用偏置
            gather_output=False,  # 不收集输出
            linear_metric_name="attn_pre_proj",  # 指定线性层的度量名称
            world_size=world_size,  # 分布式世界的大小
        )

        self.o_proj = RowParallelLinear(  # 定义输出投影层（行并行）
            config.num_q_heads * self.head_dim,
            config.embedding_dim,
            bias=config.use_bias,  # 是否使用偏置
            input_is_parallel=True,  # 输入是并行的
            reduce_results=False,  # 不减少结果
            linear_metric_name="attn_post_proj",  # 指定线性层的度量名称
            world_size=world_size,  # 分布式世界的大小
        )
        self.rotary_emb = None  # 初始化旋转位置编码为None
        if isinstance(config.rope_theta, int) or isinstance(config.rope_theta, float):  # 如果rope_theta是整数或浮点数
            self.rotary_emb = get_rope(  # 获取旋转位置编码
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,  # 最大位置数
                base=config.rope_theta,  # 基础频率
                is_neox_style=config.is_neox_style,  # 是否使用NeoX风格
                rope_scaling=config.rope_scaling,  # 旋转位置编码的缩放
            )
        self._attn_rope_timer = CudaTimer("attn_rope")  # 初始化CUDA计时器，用于记录旋转位置编码的时间

    def forward(self, hidden_states, positions):  # 定义前向传播函数
        qkv, _ = self.qkv_proj(hidden_states)  # 通过QKV投影层计算QKV
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 将QKV拆分为查询、键和值
        with self._attn_rope_timer:  # 使用计时器记录旋转位置编码时间
            q, k = self.rotary_emb(positions, q, k)  # 应用旋转位置编码
        # 输出从注意力机制中得到的结果，形状与q相同
        attn_output = torch.randn_like(q)  # 随机生成注意力输出（模拟）
        output, _ = self.o_proj(attn_output)  # 通过输出投影层计算最终输出
        return output  # 返回输出


class MLP(torch.nn.Module):  # 定义多层感知机（MLP）模块
    def __init__(self, config: ModelConfig, world_size: int):  # 初始化函数
        super().__init__()  # 调用父类构造函数
        assert config.embedding_dim % world_size == 0  # 确保嵌入维度能被世界大小整除

        if config.use_gated_mlp:  # 如果使用门控MLP
            self.up_proj = ColumnParallelLinear(  # 定义上投影层（列并行）
                config.embedding_dim,
                2 * config.mlp_hidden_dim,  # 输出维度为隐藏层的两倍
                bias=config.use_bias,  # 是否使用偏置
                gather_output=False,  # 不收集输出
                world_size=world_size,  # 分布式世界的大小
                linear_metric_name="mlp_up_proj",  # 指定线性层的度量名称
            )
            self.act = SiluAndMul()  # 使用SiluAndMul激活函数
        else:  # 如果不使用门控MLP
            self.up_proj = ColumnParallelLinear(  # 定义上投影层（列并行）
                config.embedding_dim,
                config.mlp_hidden_dim,  # 输出维度为隐藏层大小
                bias=config.use_bias,  # 是否使用偏置
                gather_output=False,  # 不收集输出
                world_size=world_size,  # 分布式世界的大小
                linear_metric_name="mlp_up_proj",  # 指定线性层的度量名称
            )
            self.act = torch.nn.GELU()  # 使用GELU激活函数

        self.down_proj = RowParallelLinear(  # 定义下投影层（行并行）
            config.mlp_hidden_dim,
            config.embedding_dim,
            bias=config.use_bias,  # 是否使用偏置
            input_is_parallel=True,  # 输入是并行的
            world_size=world_size,  # 分布式世界的大小
            reduce_results=False,  # 不减少结果
            linear_metric_name="mlp_down_proj",  # 指定线性层的度量名称
        )

        self.mlp_act_timer = CudaTimer("mlp_act")  # 初始化CUDA计时器，用于记录激活函数的时间

    def forward(self, hidden_states):  # 定义前向传播函数
        hidden_states, _ = self.up_proj(hidden_states)  # 通过上投影层计算隐藏状态
        with self.mlp_act_timer:  # 使用计时器记录激活函数时间
            hidden_states = self.act(hidden_states)  # 应用激活函数
        hidden_states, _ = self.down_proj(hidden_states)  # 通过下投影层计算最终隐藏状态
        return hidden_states  # 返回隐藏状态


class GPTBlock(torch.nn.Module):  # 定义GPT块模块
    def __init__(self, config: ModelConfig, world_size: int):  # 初始化函数
        super().__init__()  # 调用父类构造函数

        if config.norm == "layer_norm":  # 如果使用LayerNorm
            self.input_layernorm = torch.nn.LayerNorm(config.embedding_dim)  # 定义输入层归一化
        elif config.norm == "rms_norm":  # 如果使用RMSNorm
            self.input_layernorm = RMSNorm(config.embedding_dim)  # 定义输入层归一化
        else:
            raise ValueError(f"Unknown norm: {config.norm} for input_layernorm")  # 抛出未知归一化类型的错误

        self._post_attn_norm = config.post_attn_norm  # 是否使用后注意力归一化
        if config.post_attn_norm:  # 如果使用后注意力归一化
            if config.norm == "rms_norm":  # 如果使用RMSNorm
                self.post_attention_layernorm = RMSNorm(config.embedding_dim)  # 定义后注意力归一化
            else:
                raise ValueError(
                    f"Unknown norm: {config.norm} for post_attention_layernorm"
                )  # 抛出未知归一化类型的错误

        self.attn = CausalSelfAttention(config, world_size)  # 定义因果自注意力模块
        self.mlp = MLP(config, world_size)  # 定义MLP模块

        self.input_layernorm_timer = CudaTimer("input_layernorm")  # 初始化CUDA计时器，用于记录输入归一化时间
        self.post_attention_layernorm_timer = CudaTimer("post_attention_layernorm")  # 初始化CUDA计时器，用于记录后注意力归一化时间
        self.add_timer = CudaTimer("add")  # 初始化CUDA计时器，用于记录加法操作时间

    def forward(self, positions, hidden_states, residual):  # 定义前向传播函数
        if self._post_attn_norm:  # 如果使用后注意力归一化
            return self._forward_with_post_attn_norm(positions, hidden_states, residual)  # 调用带后归一化的前向传播
        else:
            return self._forward_without_post_attn_norm(positions, hidden_states)  # 调用不带后归一化的前向传播

    def _forward_with_post_attn_norm(  # 定义带后注意力归一化的前向传播
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        # 自注意力
        residual = hidden_states  # 保存残差
        with self.input_layernorm_timer:  # 使用计时器记录输入归一化时间
            hidden_states = self.input_layernorm(hidden_states)  # 应用输入归一化
        hidden_states = self.attn(  # 通过自注意力模块计算隐藏状态
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states  # 添加残差
        # 全连接层
        residual = hidden_states  # 保存残差
        with self.post_attention_layernorm_timer:  # 使用计时器记录后注意力归一化时间
            hidden_states = self.post_attention_layernorm(hidden_states)  # 应用后注意力归一化
        hidden_states = self.mlp(hidden_states)  # 通过MLP模块计算隐藏状态
        with self.add_timer:  # 使用计时器记录加法操作时间
            hidden_states = residual + hidden_states  # 添加残差
        return hidden_states  # 返回隐藏状态

    def _forward_without_post_attn_norm(  # 定义不带后注意力归一化的前向传播
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        residual = hidden_states  # 保存残差
        with self.input_layernorm_timer:  # 使用计时器记录输入归一化时间
            hidden_states = self.input_layernorm(hidden_states)  # 应用输入归一化
        attn_outputs = self.attn(  # 通过自注意力模块计算注意力输出
            positions=positions,
            hidden_states=hidden_states,
        )
        feed_forward_hidden_states = self.mlp(hidden_states)  # 通过MLP模块计算隐藏状态
        with self.add_timer:  # 使用计时器记录加法操作时间
            hidden_states = attn_outputs + feed_forward_hidden_states + residual  # 添加残差
        return hidden_states  # 返回隐藏状态


class GPTModel(torch.nn.Module):  # 定义GPT模型
    def __init__(self, config: ModelConfig, world_size: int, num_repeat_steps: int = 1):  # 初始化函数
        super().__init__()  # 调用父类构造函数

        self.num_repeat_steps = num_repeat_steps  # 设置重复步骤数

        self.embed_tokens = VocabParallelEmbedding(  # 定义词汇表并行嵌入层
            config.vocab_size,
            config.embedding_dim,
            linear_metric_name="emb",  # 指定线性层的度量名称
            reduce_results=False,  # 不减少结果
            world_size=world_size,  # 分布式世界的大小
            rank=0,  # 当前rank为0
        )

        self.block = GPTBlock(config, world_size=world_size)  # 定义GPT块

    def forward(self, input_ids, positions):  # 定义前向传播函数
        hidden_states = self.embed_tokens(input_ids)  # 通过嵌入层计算隐藏状态
        residual = hidden_states  # 保存残差
        for _ in range(self.num_repeat_steps):  # 重复多次
            hidden_states = self.embed_tokens(input_ids)  # 再次通过嵌入层计算隐藏状态
            hidden_states = self.block(  # 通过GPT块计算隐藏状态
                positions,
                hidden_states,
                residual,
            )

        return hidden_states  # 返回隐藏状态
