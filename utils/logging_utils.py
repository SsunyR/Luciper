import logging
import sys

def setup_logger(name='luciper', level=logging.INFO):
    """
    Sets up a basic logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers: # Avoid adding multiple handlers if called repeatedly
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Optionally add a file handler
        # log_file = "luciper_app.log"
        # fh = logging.FileHandler(log_file)
        # fh.setLevel(level)
        # fh.setFormatter(formatter)
        # logger.addHandler(fh)

    return logger

# Example usage:
# logger = setup_logger()
# logger.info("Logger setup complete.")
