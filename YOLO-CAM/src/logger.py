# 创建一个自定义的日志记录器（Logger）
import os
import sys
import logging
from datetime import datetime


class LOGGER(logging.Logger):
    """
    Logger.

    Args:
         logger_name: String. Logger name.
         rank: Integer. Rank id.
    """

    def __init__(self, logger_name, rank=0):
        super(LOGGER, self).__init__(logger_name)
        self.rank = rank
        if rank % 8 == 0:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
            console.setFormatter(formatter)
            self.addHandler(console)

    def setup_logging_file(self, log_dir, rank=0):
        """Setup logging file."""
        self.rank = rank
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_name = datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S') + '_rank_{}.log'.format(rank)
        self.log_fn = os.path.join(log_dir, log_name)
        fh = logging.FileHandler(self.log_fn)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        fh.setFormatter(formatter)
        self.addHandler(fh)

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)

    def save_args(self, args):
        self.info('Args:')
        args_dict = vars(args)
        for key in args_dict.keys():
            self.info('--> %s: %s', key, args_dict[key])
        self.info('')

    def important_info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO) and self.rank == 0:
            line_width = 2
            important_msg = '\n'
            important_msg += ('*' * 70 + '\n') * line_width
            important_msg += ('*' * line_width + '\n') * 2
            important_msg += '*' * line_width + ' ' * 8 + msg + '\n'
            important_msg += ('*' * line_width + '\n') * 2
            important_msg += ('*' * 70 + '\n') * line_width
            self.info(important_msg, *args, **kwargs)


def get_logger(path, rank):
    """Get Logger."""
    logger = LOGGER('YOLOV5', rank)
    logger.setup_logging_file(path, rank)
    return logger
