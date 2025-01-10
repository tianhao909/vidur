from typing import Tuple

from vidur.entities import Batch, BatchStage, ExecutionTime  # 导入所需的类型

from vidur.execution_time_predictor import BaseExecutionTimePredictor  # 导入 BaseExecutionTimePredictor

class ReplicaStageScheduler:  # 定义 ReplicaStageScheduler 类
    def __init__(  # 定义构造函数
        self,
        replica_id: int,  # 复制品ID
        stage_id: int,  # 阶段ID
        is_last_stage: bool,  # 是否是最后一个阶段
        execution_time_predictor: BaseExecutionTimePredictor,  # 执行时间预测器
    ) -> None:
        self._replica_id = replica_id  # 设置复制品ID
        self._stage_id = stage_id  # 设置阶段ID
        self._is_last_stage = is_last_stage  # 设置是否是最后一个阶段
        self._execution_time_predictor = execution_time_predictor  # 设置执行时间预测器

        self._batch_queue = []  # 初始化批处理队列
        self._is_busy = False  # 初始化为不忙状态

    @property
    def is_last_stage(self) -> bool:  # 定义 is_last_stage 属性
        return self._is_last_stage  # 返回是否是最后一个阶段

    def is_empty(self) -> bool:  # 判断队列是否为空
        return len(self._batch_queue) == 0  # 返回队列是否为空的布尔值

    def add_batch(self, batch: Batch) -> None:  # 添加批处理
        self._batch_queue.append(batch)  # 将批处理添加到队列尾部

    def on_stage_end(self) -> None:  # 当阶段结束时执行
        self._is_busy = False  # 设置为不忙状态

    def on_schedule(self) -> Tuple[Batch, BatchStage, ExecutionTime]:  # 调度方法
        if self._is_busy or not self._batch_queue:  # 如果忙或者队列为空
            return None, None, None  # 返回空值

        self._is_busy = True  # 设置为忙状态
        batch = self._batch_queue.pop(0)  # 取出队列中第一个批处理
        execution_time = self._execution_time_predictor.get_execution_time(  # 获取执行时间
            batch,
            self._stage_id,
        )
        total_execution_time = execution_time.total_time  # 获取执行总时间
        model_execution_time = execution_time.model_time  # 获取模型执行时间
        batch_stage = BatchStage(  # 创建 BatchStage 实例
            batch.id,
            self._replica_id,
            self._stage_id,
            total_execution_time,
            model_execution_time,
            batch.requests,
            batch.num_tokens,
        )

        return batch, batch_stage, execution_time  # 返回批处理，批处理阶段和执行时间