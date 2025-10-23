"""Helpers for parsing config date fields with relative tokens."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Optional


_RELATIVE_PATTERN = re.compile(r"^today(?:-(\d+))?(?:\$)?$", re.IGNORECASE)


def _field_label(field: str | None) -> str:
    return f"Config key '{field}'" if field else "Date value"


def _coerce_date_text(value: Any, field: str | None, allow_empty: bool) -> Optional[str]:
    if value is None:
        if allow_empty:
            return None
        raise ValueError(f"{_field_label(field)} cannot be null")

    text = str(value).strip()
    if text == "":
        if allow_empty:
            return None
        raise ValueError(f"{_field_label(field)} cannot be empty")
    return text


def _parse_date_text(text: str) -> datetime:
    match = _RELATIVE_PATTERN.fullmatch(text.lower())
    if match:
        offset_days = int(match.group(1) or 0)
        target = datetime.now().date() - timedelta(days=offset_days)
        return datetime.combine(target, datetime.min.time())
    return datetime.strptime(text, "%Y-%m-%d")


def parse_config_date(value: Any, field: str | None = None) -> datetime:
    """Parse a required config date supporting YYYY-MM-DD and 'today[-X$]'."""

    text = _coerce_date_text(value, field, allow_empty=False)
    assert text is not None  # for mypy; handled by allow_empty flag
    try:
        return _parse_date_text(text)
    except ValueError as exc:
        raise ValueError(f"{_field_label(field)} must use YYYY-MM-DD or 'today[-X$]' format") from exc


def parse_optional_config_date(value: Any, field: str | None = None) -> Optional[datetime]:
    """Parse an optional config date, returning None when value is missing/empty."""

    text = _coerce_date_text(value, field, allow_empty=True)
    if text is None:
        return None
    try:
        return _parse_date_text(text)
    except ValueError as exc:
        raise ValueError(f"{_field_label(field)} must use YYYY-MM-DD or 'today[-X$]' format") from exc


def normalize_config_date_str(
    value: Any,
    field: str | None = None,
    allow_empty: bool = False,
) -> Optional[str]:
    """Return config date formatted as 'YYYY-MM-DD'."""

    if allow_empty:
        dt = parse_optional_config_date(value, field)
        if dt is None:
            return None
    else:
        dt = parse_config_date(value, field)
    return dt.strftime("%Y-%m-%d")
