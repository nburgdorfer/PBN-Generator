from __future__ import annotations

import logging


LOG_FORMAT = "%(asctime)s %(levelname)s %(module)s.%(funcName)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColorLevelFormatter(logging.Formatter):
    COLORS = {
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        color = self.COLORS.get(record.levelno)
        if color is not None:
            record.levelname = f"{color}[{record.levelname}]{self.RESET}"
        else:
            record.levelname = f"[{record.levelname}]"

        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def configure_logging() -> None:
    formatter = ColorLevelFormatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    logging.basicConfig(
        # level=logging.WARNING,
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    # logging.getLogger("pbn_generator").setLevel(logging.WARNING)
    logging.getLogger("pbn_generator").setLevel(logging.INFO)
