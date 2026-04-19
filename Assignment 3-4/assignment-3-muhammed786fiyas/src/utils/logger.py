import logging
import os
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    
    os.makedirs("logs", exist_ok=True)

    # Timestamp for this specific run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger(f"{name}_{timestamp}")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    # File handler — timestamped, DEBUG and above
    file_handler = logging.FileHandler(
        f"logs/{name}_{timestamp}.log", mode='w'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger