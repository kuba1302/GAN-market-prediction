import logging
import sys


def prepare_logger(log_level):
    logger = logging.getLogger()
    logger.setLevel(level=log_level)
    formatter = logging.Formatter("%(asctime)s,%(msecs)d %(levelname)-8s %(message)s")
    if not logger.handlers:
        lh = logging.StreamHandler(sys.stdout)
        lh.setFormatter(formatter)
        logger.addHandler(lh)
    return logger
