import logging
import sys

def create_logger(filepath: str=None, name: str='aic21', level: int=logging.INFO, stdout=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if filepath:
        f_handler = logging.FileHandler(filepath)
        f_format = logging.Formatter('%(levelname)s: %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)
    
    if stdout:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    return logger