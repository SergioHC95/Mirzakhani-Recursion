import logging
import sys

def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        log.addHandler(handler)
        log.setLevel(logging.INFO)
    return log
