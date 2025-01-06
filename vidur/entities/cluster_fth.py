import json

from vidur.config import BaseRequestGeneratorConfig, ClusterConfig, MetricsConfig  # 从vidur.config导入BaseRequestGeneratorConfig, ClusterConfig, MetricsConfig
from vidur.entities.base_entity import BaseEntity  # 从vidur.entities.base_entity导入BaseEntity
from vidur.entities.replica import Replica  # 从vidur.entities.replica导入Replica
from vidur.logger import init_logger  # 从vidur.logger导入init_logger

logger = init_logger(__name__)  # 初始化日志记录器

class Cluster(BaseEntity):  # 定义Cluster类，继承自BaseEntity
    def __init__(  # 初始化方法
        self,
        cluster_config: ClusterConfig,  # 接收一个ClusterConfig类型的参数
        metrics_config: MetricsConfig,  # 接收一个MetricsConfig类型的参数
        generator_config: BaseRequestGeneratorConfig,  # 接收一个BaseRequestGeneratorConfig类型的参数
    ) -> None:
        self._id = Cluster.generate_id()  # 生成一个唯一的集群ID
        self._config = cluster_config  # 保存传入的集群配置

        # get metrics config
        self._output_dir = metrics_config.output_dir  # 获取metrics配置中的输出目录

        # Init replica object handles
        self._replicas = {}  # 初始化一个字典，用于保存副本对象

        for _ in range(self._config.num_replicas):  # 根据配置中的副本数量创建副本
            replica = Replica(self._config.replica_config, generator_config)  # 创建一个副本对象
            self._replicas[replica.id] = replica  # 将副本对象按照ID存入字典

        if metrics_config.write_json_trace:  # 如果需要写入JSON追踪信息
            self._write_cluster_info_to_file()  # 调用方法将集群信息写入文件

    @property
    def replicas(self):  # 定义一个属性方法，用于获取副本信息
        return self._replicas

    def to_dict(self) -> dict:  # 将集群信息转换为字典格式
        return {
            "id": self._id,  # 集群ID
            "num_replicas": len(self._replicas),  # 副本数量
        }

    def _write_cluster_info_to_file(self) -> None:  # 将集群信息写入文件
        replica_dicts = [replica.to_dict() for replica in self._replicas.values()]  # 获取所有副本的信息字典
        cluster_info = {"replicas": replica_dicts}  # 将副本信息放入集群信息字典中

        cluster_file = f"{self._output_dir}/cluster.json"  # 定义要写入的文件路径
        with open(cluster_file, "w") as f:  # 打开文件进行写入
            json.dump(cluster_info, f)  # 将集群信息写入文件