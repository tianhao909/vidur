from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds
import pdb

from pprint import pprint # fth 假设config是一个类实例

def main() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()
    # pdb.set_trace() # fth 
    print('fth 打印配置信息pprint')
    pprint(vars(config)) # fth 打印配置信息

    set_seeds(config.seed)

    # print('fth pass')
    # print('>>>fth config {config}') # fth 打印配置信息
    # breakpoint() # fth 断点
    simulator = Simulator(config)
    # print('fth pass')

    simulator.run()


if __name__ == "__main__":
    main()
