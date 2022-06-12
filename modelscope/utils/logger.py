# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
from typing import Optional

init_loggers = {}


def get_logger(log_file: Optional[str] = None,
               log_level: int = logging.INFO,
               file_mode: str = 'w'):
    """ Get logging logger

    Args:
        log_file: Log filename, if specified, file handler will be added to
            logger
        log_level: Logging level.
        file_mode: Specifies the mode to open the file, if filename is
            specified (if filemode is unspecified, it defaults to 'w').
    """
    logger_name = __name__.split('.')[0]
    logger = logging.getLogger(logger_name)

    if logger_name in init_loggers:
        return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    # TODO @wenmeng.zwm add logger setting for distributed environment
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)
    init_loggers[logger_name] = True

    return logger
