from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_SENSITIVE_KEY = re.compile(r"(?:account|token|secret|api[_-]?key|authorization)", re.IGNORECASE)
_LONG_DIGITS = re.compile(r"(?<!\d)\d{7,}(?!\d)")
_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+")


def redact_text(value: str) -> str:
    value = _BEARER.sub("Bearer [REDACTED]", value)
    return _LONG_DIGITS.sub("[REDACTED_IDENTIFIER]", value)


def redact(value: Any, key: str = "") -> Any:
    if _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {str(item_key): redact(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list | tuple):
        return [redact(item) for item in value]
    return value
