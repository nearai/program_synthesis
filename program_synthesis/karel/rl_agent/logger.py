import logging
from os.path import join, exists
from os import mkdir
from sys import stdout

LOGS_PATH = join('logdirs', 'rl')


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s| %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(stdout)
    stdout_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stdout_handler)

    return logger


if not exists(LOGS_PATH):
    mkdir(LOGS_PATH)

logger_debug = setup_logger('logger_debug', join(LOGS_PATH, 'logger_debug.log'))
logger_debug.disabled = False

logger_train = setup_logger('logger_train', join(LOGS_PATH, 'logger_train.log'))
logger_train.disabled = False

logger_task = setup_logger('logger_task', join(LOGS_PATH, 'logger_task.log'))
logger_task.disabled = False

logger_model = setup_logger('logger_model', join(LOGS_PATH, 'logger_model.log'))
logger_model.disabled = False
