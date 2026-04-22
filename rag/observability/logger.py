import time
import logging
import functools
from pathlib import Path


_configured = False


def setup_logging(log_level: str = "INFO", log_file: str = "rag_system.log"):
  
    global _configured
    if _configured:
        return


    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)


    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # --- إعداد المسجل الجذري ---
    root_logger = logging.getLogger("rag")
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:


    if not _configured:
        setup_logging()
    return logging.getLogger(f"rag.{name}")


def timed(func=None, *, label: str = None):

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            fn_label = label or fn.__name__
            logger = get_logger("timing")
            logger.info(f"⏱️  Start: {fn_label}")
            start = time.perf_counter()

            result = fn(*args, **kwargs)

            elapsed = time.perf_counter() - start
            logger.info(f"Finish ✅ : {fn_label} — {elapsed:.2f} seconds")
            return result

        return wrapper


    if func is not None:
        return decorator(func)
    return decorator
