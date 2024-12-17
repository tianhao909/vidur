from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


def main() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()

    set_seeds(config.seed)

    # print('fth pass')

    simulator = Simulator(config)
    # print('fth pass')

    simulator.run()


if __name__ == "__main__":
    main()
