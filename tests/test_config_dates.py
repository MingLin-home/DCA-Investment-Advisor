from datetime import datetime, timedelta
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_dates import normalize_config_date_str, parse_config_date, parse_optional_config_date


def test_parse_config_date_absolute() -> None:
    dt = parse_config_date("2024-01-02")
    assert dt == datetime(2024, 1, 2)


def test_parse_config_date_today_token() -> None:
    before = datetime.now().date()
    dt = parse_config_date("today")
    after = datetime.now().date()
    assert before <= dt.date() <= after


def test_parse_config_date_today_offset() -> None:
    offset = 5
    before = (datetime.now() - timedelta(days=offset)).date()
    dt = parse_config_date(f"today-{offset}")
    after = (datetime.now() - timedelta(days=offset)).date()
    assert before <= dt.date() <= after


def test_parse_optional_config_date_empty() -> None:
    assert parse_optional_config_date("", "some_field") is None
    assert parse_optional_config_date(None, "some_field") is None


def test_normalize_config_date_str_returns_iso() -> None:
    normalized = normalize_config_date_str("today", allow_empty=False)
    assert len(normalized.split("-")) == 3


@pytest.mark.parametrize("value", ["", None, "2024/01/01", "today-abc"])
def test_parse_config_date_invalid(value: object) -> None:
    with pytest.raises(ValueError):
        parse_config_date(value, field="test_field")
