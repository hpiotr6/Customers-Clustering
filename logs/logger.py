import functools
import logging
import os
import logs
# LOG_FORMAT = "%(levelname)s %(asctime)s %(message)s"
# logging.basicConfig(filename="logs/logs.log",
#                     level=logging.INFO,
#                     format=LOG_FORMAT)
# logger = logging.getLogger()


def log(funct):

    @functools.wraps(funct)
    def wrapper(*args, **kwargs):
        path1 = os.path.dirname(logs.__file__)
        full_path = os.path.join(path1, "logs.log")
        LOG_FORMAT = "%(levelname)s %(asctime)s %(message)s"
        logging.basicConfig(filename=full_path,
                            filemode='a',
                            level=logging.DEBUG,
                            format=LOG_FORMAT)
        logger = logging.getLogger("loger")
        try:
            logger.info(f"Function:{funct.__name__}")
            result = funct(*args, **kwargs)
            return result
        except Exception as e:
            logger.exception(
                f"Exception raised in {funct.__name__}. Exception: {str(e)}")
            raise e
    return wrapper
