import os
import logging


def create_logger(name, level, filename):
    """
    创建日记对象
    :param name:日记名称，在日记文件中体现
    :param level:日记等级
    :param filename:日记文件所在目录及名称
    :return:日记对象
    """
    log_dir, log_file_name = os.path.split(filename)

    print("log_dir ", log_dir)
    if log_dir and not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    if not logger.handlers:
        if level == 'info':
            logger.setLevel(logging.INFO)
        elif level == 'debug':
            logger.setLevel(logging.DEBUG)
        elif level == 'error':
            logger.setLevel(logging.ERROR)
        elif level == 'warning':
            logger.setLevel(logging.WARNING)
        elif level == 'critical':
            logger.setLevel(logging.CRITICAL)
        else:
            return 'level is error'
        p_stream = logging.StreamHandler()
        fh = logging.FileHandler(filename)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        p_stream.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(p_stream)
    return logger


def logger_for_test(test_name, log_file_path):
    print("Log Path:", log_file_path)
    # log
    return create_logger(f"{test_name}.log", "info", log_file_path)
