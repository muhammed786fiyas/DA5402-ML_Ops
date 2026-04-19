import logging
import os

def get_logger(name: str) -> logging.Logger:
    # Always create logs/ relative to project root, not relative to where script is run from
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    # File handler - separate file per script, append mode
    log_file = os.path.join(log_dir, f"{name}.log")
    file = logging.FileHandler(log_file, mode='a')
    file.setLevel(logging.INFO)
    file.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file)
    logger.info("=" * 60)

    return logger