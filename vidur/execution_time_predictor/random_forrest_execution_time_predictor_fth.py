from sklearn.ensemble import RandomForestRegressor  # 从sklearn的ensemble模块中导入RandomForestRegressor

from vidur.config import (  # 从vidur.config模块中导入以下配置类
    BaseReplicaSchedulerConfig,  # 基本副本调度器配置类
    MetricsConfig,               # 指标配置类
    RandomForrestExecutionTimePredictorConfig,  # 随机森林执行时间预测器配置类
    ReplicaConfig,               # 副本配置类
)
from vidur.execution_time_predictor.sklearn_execution_time_predictor import (  # 从vidur.execution_time_predictor.sklearn_execution_time_predictor模块中导入
    SklearnExecutionTimePredictor,  # Sklearn执行时间预测器类
)


class RandomForrestExecutionTimePredictor(SklearnExecutionTimePredictor):  # 定义一个继承自SklearnExecutionTimePredictor的类
    def __init__(  # 类的初始化方法
        self,
        predictor_config: RandomForrestExecutionTimePredictorConfig,  # 预测器配置参数
        replica_config: ReplicaConfig,  # 副本配置参数
        replica_scheduler_config: BaseReplicaSchedulerConfig,  # 副本调度器配置参数
        metrics_config: MetricsConfig,  # 指标配置参数
    ) -> None:
        # will trigger model training  # 将触发模型训练
        super().__init__(  # 调用父类的初始化方法
            predictor_config=predictor_config,  # 传递预测器配置参数
            replica_config=replica_config,  # 传递副本配置参数
            replica_scheduler_config=replica_scheduler_config,  # 传递副本调度器配置参数
            metrics_config=metrics_config,  # 传递指标配置参数
        )

    def _get_grid_search_params(self):  # 定义获取网格搜索参数的方法
        return {  # 返回参数字典
            "n_estimators": self._config.num_estimators,  # 决策树的数量
            "max_depth": self._config.max_depth,  # 树的最大深度
            "min_samples_split": self._config.min_samples_split,  # 拆分节点时所需的最小样本数
        }

    def _get_estimator(self):  # 定义获取估计器的方法
        return RandomForestRegressor()  # 返回随机森林回归器实例
