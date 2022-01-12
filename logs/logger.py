import functools
import logging
import os

# LOG_FORMAT = "%(levelname)s %(asctime)s %(message)s"
# logging.basicConfig(filename="logs/logs.log",
#                     level=logging.INFO,
#                     format=LOG_FORMAT)
# logger = logging.getLogger()


def log(funct):
    LOG_FORMAT = "%(levelname)s %(asctime)s %(message)s"
    logging.basicConfig(filename="logs.log",
                        level=logging.INFO,
                        format=LOG_FORMAT)
    logger = logging.getLogger()

    @functools.wraps(funct)
    def wrapper(*args, **kwargs):
        try:
            logger.info(f"Function:{funct.__name__}")
            result = funct(*args, **kwargs)
            return result
        except Exception as e:
            logger.exception(
                f"Exception raised in {func.__name__}. Exception: {str(e)}")
            raise e
    return wrapper
