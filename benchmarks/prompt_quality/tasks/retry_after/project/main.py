"""Helpers for polite HTTP retries."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Mapping


def _get_header(headers: Mapping[str, str], name: str) -> str | None:
    """Look up an HTTP header without depending on key casing."""
    return next(
        (value for key, value in headers.items() if key.casefold() == name.casefold()),
        None,
    )


def parse_retry_after(
    headers: Mapping[str, str], *, now: datetime | None = None, default: float = 1.0
) -> float:
    """Return a non-negative retry delay in seconds."""
    raise NotImplementedError
