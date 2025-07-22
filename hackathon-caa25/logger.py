from logging import INFO, Formatter, StreamHandler, getLogger
from logging.handlers import RotatingFileHandler
from os import makedirs, path

ROOT_DIR_MODULE = path.dirname(path.abspath(__file__))

LOG_DIR = path.join(ROOT_DIR_MODULE, "logs")
makedirs(LOG_DIR, exist_ok=True)


def setup_logge(name: str, log_path: str = ROOT_DIR_MODULE, level: int = INFO):
    """
    Sets up a logger with both file and console handlers.
    Args:
        name (str): The name of the logger.
        log_path (str): The directory where log files will be stored.
        level (int): The logging level (default is INFO).
    Returns:
        Logger: Configured logger instance.
    """

    if len(path.split(log_path)) > 1:
        name = path.split(log_path)[-1]

    logger = getLogger(name)
    logger.setLevel(level)

    # Create a file handler
    file_handler = RotatingFileHandler(
        path.join(log_path, f"{name}.log"),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
    )
    file_handler.setLevel(level)

    # Create a console handler
    console_handler = StreamHandler()
    console_handler.setLevel(level)

    # Create a formatter and set it for both handlers
    formatter = Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
