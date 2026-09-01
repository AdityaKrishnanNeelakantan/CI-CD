"""Structured, redacting logger factory. No print() anywhere in package code."""
from __future__ import annotations

import logging
import re

_KV = re.compile(r"(?i)\b(password|passwd|pwd|secret|token|api_key|apikey|credential|dsn)\s*=\s*\S+")
_URL_CRED = re.compile(r"://[^@\s/]+@")


def redact(text: str) -> str:
    text = _KV.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    return _URL_CRED.sub("://[REDACTED]@", text)


class _Redactor(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        return True


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        h.addFilter(_Redactor())
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger
