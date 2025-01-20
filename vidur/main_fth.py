from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


def main() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()  # 从命令行参数创建模拟配置

    set_seeds(config.seed)  # 设置随机数种子

    # print('fth pass')  # 打印调试信息

    # print('>>>fth config {config}') # fth 打印配置信息

    simulator = Simulator(config)  # 创建模拟器实例
    # print('fth pass')  # 打印调试信息

    simulator.run()  # 运行模拟器


if __name__ == "__main__":
    main()  # 如果是主程序，则执行main函数