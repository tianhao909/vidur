from typing import Tuple

from vidur.entities import Batch, BatchStage, ExecutionTime
from vidur.execution_time_predictor import BaseExecutionTimePredictor


class ReplicaStageScheduler:
    def __init__(
        self,
        replica_id: int,
        stage_id: int,
        is_last_stage: bool,
        execution_time_predictor: BaseExecutionTimePredictor,
    ) -> None:
        self._replica_id = replica_id
        self._stage_id = stage_id
        self._is_last_stage = is_last_stage
        self._execution_time_predictor = execution_time_predictor

        self._batch_queue = []
        self._is_busy = False

    @property
    def is_last_stage(self) -> bool:
        return self._is_last_stage

    def is_empty(self) -> bool:
        return len(self._batch_queue) == 0

    def add_batch(self, batch: Batch) -> None:
        self._batch_queue.append(batch)

    def on_stage_end(self) -> None:
        self._is_busy = False

    def on_schedule(self) -> Tuple[Batch, BatchStage, ExecutionTime]:
        # print('>>fth def on_schedule() vidur/vidur/scheduler/replica_stage_scheduler/replica_stage_schduler.py')
        if self._is_busy or not self._batch_queue:
            return None, None, None

        self._is_busy = True
        batch = self._batch_queue.pop(0)
        execution_time = self._execution_time_predictor.get_execution_time(
            batch,
            self._stage_id,
        )
        total_execution_time = execution_time.total_time
        model_execution_time = execution_time.model_time
        batch_stage = BatchStage(
            batch.id,
            self._replica_id,
            self._stage_id,
            total_execution_time,
            model_execution_time,
            batch.requests,
            batch.num_tokens,
        )

        return batch, batch_stage, execution_time
