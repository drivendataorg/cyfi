import logging
import sys

try:
    from loguru import logger
except ImportError:
    # Fallback to standard logging if loguru is not installed
    logger = logging.getLogger("cyfi")
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Add SUCCESS level to match loguru
    SUCCESS_LEVEL_NUM = 25
    logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

    def success(self, message, *args, **kws):
        if self.isEnabledFor(SUCCESS_LEVEL_NUM):
            self._log(SUCCESS_LEVEL_NUM, message, args, **kws)

    logging.Logger.success = success

    # Add remove and add methods to mimic loguru's API for basic usage
    def logger_remove():
        for h in logger.handlers[:]:
            logger.removeHandler(h)

    def logger_add(sink, level="INFO"):
        if sink == sys.stderr:
            h = logging.StreamHandler(sys.stderr)
        elif isinstance(sink, (str, sys.path.__class__)):
            h = logging.FileHandler(sink)
        else:
            return  # Not supported in fallback

        h.setFormatter(formatter)
        logger.addHandler(h)
        logger.setLevel(level)

    logger.remove = logger_remove
    logger.add = logger_add
