"""Configure PORMAKE's single shared logger.

PORMAKE routes all of its diagnostics through one module-level
``logger`` named ``"unique_logger"``. Centralizing on a single logger
keeps the assembly pipeline's output consistent and lets callers toggle
verbosity from one place. The logger sets ``propagate = False`` so its
records do not bubble up to the root logger and collide with logging
configured by other libraries.

Two handlers are attached. A :class:`logging.FileHandler` writes every
level (``DEBUG`` and up) to ``runtime.log`` for a detailed trace, while
a :class:`logging.StreamHandler` prints a simplified ``>>>``-prefixed
message to the console at ``INFO`` and above. The helper functions in
this module raise or lower each handler's threshold so users can quiet
or restore console and file output independently at runtime.
"""

import logging

# Make a logger. Pormake uses a single logger.
logger = logging.getLogger("unique_logger")
# To prevent colloisions with loggers in other libraries.
logger.propagate = False
# Log all levels.
logger.setLevel(logging.DEBUG)

# Setting for the file logs.
file_log_handler = logging.FileHandler(filename="runtime.log", mode="w")
file_log_handler.setLevel(logging.DEBUG)

_format = (
    "[%(asctime)s (%(levelname)s) " "%(filename)s:%(lineno)s] " "%(message)s"
)

formatter = logging.Formatter(
    fmt=_format,
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


def disable_print():
    """Silence console messages below the ``WARNING`` level.

    Raises the console handler's threshold to ``WARNING`` so routine
    ``INFO``/``DEBUG`` progress messages no longer print to the screen.
    """
    console_log_handler.setLevel(logging.WARNING)
    logger.warning("Console logs (under WARNING level) are disabled.")


def enable_print():
    """Restore console messages at the ``INFO`` level and above.

    Lowers the console handler's threshold back to ``INFO``, undoing a
    previous :func:`disable_print` call.
    """
    console_log_handler.setLevel(logging.INFO)
    logger.warning("Console logs (under WARNING level) are enabled.")


def disable_file_print():
    """Silence file-log messages below the ``WARNING`` level.

    Raises the file handler's threshold to ``WARNING`` so the
    ``runtime.log`` file records only warnings and errors.
    """
    file_log_handler.setLevel(logging.WARNING)
    logging.warning("File logs (under WARNING level) are disabled.")


def enable_file_print():
    """Restore full ``DEBUG``-level logging to the file.

    Lowers the file handler's threshold back to ``DEBUG`` so every level
    is written to ``runtime.log`` again, undoing
    :func:`disable_file_print`.
    """
    file_log_handler.setLevel(logging.DEBUG)
    logging.warning("File logs (all levels) are enabled.")
