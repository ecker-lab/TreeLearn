import logging
import os
import shutil
import time
from tensorboardX import SummaryWriter as _SummaryWriter


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger('TreeLearn')
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


class SummaryWriter(_SummaryWriter):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def add_scalar(self, *args, **kwargs):
        return super().add_scalar(*args, **kwargs)

    def flush(self, *args, **kwargs):
        return super().flush(*args, **kwargs)


def init_train_logger(cfg, args):
    save_directory = cfg.work_dir
    os.makedirs(os.path.abspath(save_directory), exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(save_directory, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)
    logger.info(f'Config:\n{cfg}')
    logger.info(f'Mix precision training: {cfg.fp16}')
    shutil.copy(args.config, os.path.join(cfg.work_dir, os.path.basename(args.config)))
    writer = SummaryWriter(save_directory)
    return logger, writer
