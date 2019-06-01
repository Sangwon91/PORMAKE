import logging

# Make a logger. Cyanide uses a single logger.
logger = logging.getLogger("unique_logger")
# To prevent colloisions with loggers in other libraries.
logger.propagate = False
# Log all levels.
logger.setLevel(logging.DEBUG)

# Setting for the file logs.
file_log_handler = logging.FileHandler(filename="test.log", mode="w")
file_log_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt="[%(asctime)s %(levelname)s %(filename)s:%(lineno)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_log_handler.setFormatter(formatter)

# Setting for the console logs.
console_log_handler = logging.StreamHandler()
console_log_handler.setLevel(logging.INFO)
# Simple formatter.
formatter = logging.Formatter(fmt=">>> %(message)s")
console_log_handler.setFormatter(formatter)

# Add the handlers to the logger.
logger.addHandler(file_log_handler)
logger.addHandler(console_log_handler)
