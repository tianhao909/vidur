import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_FILE = f"{ROOT_DIR}/config/default.yml"

LOGGER_FORMAT = (
    "[%(asctime)s][%(filename)s:%(lineno)d:%(funcName)s]" "[%(levelname)s] %(message)s"
)
LOGGER_TIME_FORMAT = "%H:%M:%S"
