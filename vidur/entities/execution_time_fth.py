from vidur.entities.base_entity import BaseEntity

class ExecutionTime(BaseEntity):
    def __init__(  # 初始化函数
        self,
        num_layers_per_pipeline_stage: int,  # 每个管道阶段的层数
        attention_rope_execution_time: float,  # 注意力机制中的 ROPE 执行时间
        attention_kv_cache_save_execution_time: float,  # 注意力 KV 缓存保存执行时间
        attention_decode_execution_time: float,  # 注意力解码执行时间
        attention_prefill_execution_time: float,  # 注意力预填充执行时间
        attention_layer_pre_proj_execution_time: float,  # 注意力层前投影执行时间
        attention_layer_post_proj_execution_time: float,  # 注意力层后投影执行时间
        mlp_layer_up_proj_execution_time: float,  # MLP 层上投影执行时间
        mlp_layer_down_proj_execution_time: float,  # MLP 层下投影执行时间
        mlp_layer_act_execution_time: float,  # MLP 层激活执行时间
        attn_norm_time: float,  # 注意力归一化时间
        mlp_norm_time: float,  # MLP 归一化时间
        add_time: float,  # 加法操作时间
        tensor_parallel_communication_time: float,  # 张量并行通信时间
        pipeline_parallel_communication_time: float,  # 管道并行通信时间
        schedule_time: float,  # 调度时间
        sampler_e2e_time: float,  # 采样器端到端时间
        prepare_inputs_e2e_time: float,  # 准备输入端到端时间
        process_model_outputs_time: float,  # 处理模型输出时间
        ray_comm_time: float,  # Ray 通信时间
    ) -> None:
        self._id = ExecutionTime.generate_id()  # 生成ID

        self._num_layers_per_pipeline_stage = num_layers_per_pipeline_stage  # 保存管道阶段层数
        self._attention_rope_execution_time = attention_rope_execution_time  # 保存ROPE执行时间
        self._attention_kv_cache_save_execution_time = (  # 保存KV缓存保存执行时间
            attention_kv_cache_save_execution_time
        )
        self._attention_decode_execution_time = attention_decode_execution_time  # 保存解码执行时间
        self._attention_prefill_execution_time = attention_prefill_execution_time  # 保存预填充执行时间
        self._attention_layer_pre_proj_execution_time = (  # 保存前投影执行时间
            attention_layer_pre_proj_execution_time
        )
        self._attention_layer_post_proj_execution_time = (  # 保存后投影执行时间
            attention_layer_post_proj_execution_time
        )
        self._mlp_layer_up_proj_execution_time = mlp_layer_up_proj_execution_time  # 保存MLP上投影执行时间
        self._mlp_layer_down_proj_execution_time = mlp_layer_down_proj_execution_time  # 保存MLP下投影执行时间
        self._mlp_layer_act_execution_time = mlp_layer_act_execution_time  # 保存MLP激活执行时间
        self._mlp_norm_time = mlp_norm_time  # 保存MLP归一化时间
        self._attn_norm_time = attn_norm_time  # 保存注意力归一化时间
        self._add_time = add_time  # 保存加法操作时间
        self._tensor_parallel_communication_time = tensor_parallel_communication_time  # 保存张量并行通信时间
        self._pipeline_parallel_communication_time = (  # 保存管道并行通信时间
            pipeline_parallel_communication_time
        )
        self._schedule_time = schedule_time  # 保存调度时间
        self._sampler_e2e_time = sampler_e2e_time  # 保存采样器端到端时间
        self._prepare_inputs_e2e_time = prepare_inputs_e2e_time  # 保存准备输入端到端时间
        self._process_model_outputs_time = process_model_outputs_time  # 保存处理模型输出时间
        self._ray_comm_time = ray_comm_time  # 保存Ray通信时间

    def _get_mlp_layer_execution_time(self) -> float:
        # 计算 MLP 层的执行时间
        return (
            self._mlp_layer_up_proj_execution_time  # MLP上投影执行时间
            + self._mlp_layer_down_proj_execution_time  # MLP下投影执行时间
            + self._mlp_layer_act_execution_time  # MLP激活执行时间
            + self._tensor_parallel_communication_time  # 张量并行通信时间
            + self._mlp_norm_time  # MLP归一化时间
        )

    def _get_attention_layer_execution_time(self) -> float:
        # 计算注意力层的执行时间
        return (
            self._attention_layer_pre_proj_execution_time  # 注意力层前投影执行时间
            + self._attention_layer_post_proj_execution_time  # 注意力层后投影执行时间
            + self._attention_rope_execution_time  # ROPE执行时间
            + self._attention_kv_cache_save_execution_time  # KV缓存保存执行时间
            + self._attention_decode_execution_time  # 解码执行时间
            + self._attention_prefill_execution_time  # 预填充执行时间
            + self._tensor_parallel_communication_time  # 张量并行通信时间
            + self._attn_norm_time  # 注意力归一化时间
        )

    def _get_block_execution_time(self) -> float:
        # 获取每块的执行时间
        return (
            self._get_attention_layer_execution_time()  # 注意力层执行时间
            + self._get_mlp_layer_execution_time()  # MLP层执行时间
            + self._add_time  # 加法操作时间
        )

    def _get_cpu_overhead(self) -> float:
        # 获取CPU开销
        return (
            self._schedule_time  # 调度时间
            + self._sampler_e2e_time  # 采样器端到端时间
            + self._prepare_inputs_e2e_time  # 准备输入端到端时间
            + self._process_model_outputs_time  # 处理模型输出时间
            + self._ray_comm_time  # Ray通信时间
        )

    @property
    def num_layers(self) -> int:
        # 每个管道阶段的层数
        return self._num_layers_per_pipeline_stage

    @property
    def mlp_layer_up_proj_execution_time(self) -> float:
        # 返回MLP层上投影执行时间
        return self._mlp_layer_up_proj_execution_time

    @property
    def mlp_layer_down_proj_execution_time(self) -> float:
        # 返回MLP层下投影执行时间
        return self._mlp_layer_down_proj_execution_time

    @property
    def mlp_layer_act_execution_time(self) -> float:
        # 返回MLP层激活执行时间
        return self._mlp_layer_act_execution_time

    @property
    def mlp_all_reduce_time(self) -> float:
        # 返回MLP全归约时间，即张量并行通信时间
        return self._tensor_parallel_communication_time

    @property
    def attention_pre_proj_time(self) -> float:
        # 返回注意力层前投影执行时间
        return self._attention_layer_pre_proj_execution_time

    @property
    def attention_post_proj_time(self) -> float:
        # 返回注意力层后投影执行时间
        return self._attention_layer_post_proj_execution_time

    @property
    def attention_all_reduce_time(self) -> float:
        # 返回注意力全归约时间，即张量并行通信时间
        return self._tensor_parallel_communication_time

    @property
    def attention_rope_execution_time(self) -> float:
        # 返回ROPE执行时间
        return self._attention_rope_execution_time

    @property
    def attention_kv_cache_save_execution_time(self) -> float:
        # 返回KV缓存保存执行时间
        return self._attention_kv_cache_save_execution_time

    @property
    def attention_decode_execution_time(self) -> float:
        # 返回解码执行时间
        return self._attention_decode_execution_time

    @property
    def attention_prefill_execution_time(self) -> float:
        # 返回预填充执行时间
        return self._attention_prefill_execution_time

    @property
    def pipeline_parallel_communication_time(self) -> float:
        # 返回管道并行通信时间
        return self._pipeline_parallel_communication_time

    @property
    def schedule_time(self) -> float:
        # 返回调度时间
        return self._schedule_time

    @property
    def sampler_e2e_time(self) -> float:
        # 返回采样器端到端时间
        return self._sampler_e2e_time

    @property
    def prepare_inputs_e2e_time(self) -> float:
        # 返回准备输入端到端时间
        return self._prepare_inputs_e2e_time

    @property
    def process_model_outputs_time(self) -> float:
        # 返回处理模型输出时间
        return self._process_model_outputs_time

    @property
    def ray_comm_time(self) -> float:
        # 返回Ray通信时间
        return self._ray_comm_time

    @property
    def mlp_norm_time(self) -> float:
        # 返回MLP归一化时间
        return self._mlp_norm_time

    @property
    def attn_norm_time(self) -> float:
        # 返回注意力归一化时间
        return self._attn_norm_time

    @property
    def add_time(self) -> float:
        # 返回加法操作时间
        return self._add_time

    @property
    def model_time(self) -> float:
        # 返回模型时间，以秒为单位
        block_execution_time = self._get_block_execution_time()  # 获取每块的执行时间
        pipeline_stage_execution_time = (  # 管道阶段的执行时间
            block_execution_time * self._num_layers_per_pipeline_stage  # 每个管道阶段的层数
        )
        # 返回以秒为单位的时间
        return (
            pipeline_stage_execution_time + self.pipeline_parallel_communication_time  # 再加上管道并行通信时间
        ) * 1e-3

    @property
    def model_time_ms(self) -> float:
        # 返回模型时间，以毫秒为单位
        return self.model_time * 1e3

    @property
    def total_time(self) -> float:
        # 返回总时间，以秒为单位
        return self.model_time + self._get_cpu_overhead() * 1e-3  # 加上以秒为单位的CPU开销
