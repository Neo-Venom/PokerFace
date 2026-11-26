"""PII-free logging, warning suppression, and sanitized global exception hook."""
from __future__ import annotations

import logging
import sys
import warnings


class SafeFormatter(logging.Formatter):
    """Emits only fixed level and message without stack traces or raw variables."""

    def format(self, record: logging.LogRecord) -> str:
        return f"[{record.levelname}] {record.getMessage()}"


def install_pii_free_guards() -> logging.Logger:
    warnings.filterwarnings("ignore")

    def _safe_excepthook(exc_type, _exc_value, _exc_tb) -> None:
        sys.stderr.write(f"FATAL: {getattr(exc_type, '__name__', 'Error')}\n")
        sys.exit(1)

    sys.excepthook = _safe_excepthook

    logger = logging.getLogger("raksha")
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(SafeFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    return logger
