#!/usr/bin/env python3
"""Download EPS data for configured symbols via Alpha Vantage."""
import argparse
import csv
import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

try:
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen
except Exception as exc:  # pragma: no cover
    urlopen = None  # type: ignore[assignment]


ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download EPS data for symbols listed in a config file.",
    )
    parser.add_argument(
        "--config",
        "--confing",
        dest="config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    return parser.parse_args()


def load_config(path: str) -> Tuple[List[str], date, date, Path]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh) or {}

    raw_symbols = cfg.get("stock_symbols")
    if not isinstance(raw_symbols, list) or not raw_symbols:
        raise ValueError("config.yaml must define a non-empty 'stock_symbols' list")
    symbols = [str(sym).strip().upper() for sym in raw_symbols if str(sym).strip()]
    if not symbols:
        raise ValueError("No valid entries found in 'stock_symbols'")

    output_dir = cfg.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("config.yaml must define 'output_dir' as a non-empty string")
    output_path = Path(output_dir.strip()) / "raw_data"

    try:
        start = parse_date(cfg["stock_start_date"])
        end = parse_date(cfg.get("stock_end_date", date.today()))
    except KeyError as exc:
        raise ValueError(f"Missing required config key: {exc.args[0]}") from exc

    if start > end:
        raise ValueError("'stock_start_date' must be on or before 'stock_end_date'")

    output_path.mkdir(parents=True, exist_ok=True)
    return symbols, start, end, output_path


def parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if value is None:
        raise ValueError("Date value cannot be None")
    text = str(value).strip()
    if text.lower() == "today":
        return date.today()
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid date format '{value}'; expected YYYY-MM-DD or 'today'") from exc


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
        return float(value)
    except Exception:
        return None


def http_json(params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    if urlopen is None:  # pragma: no cover
        raise RuntimeError("urllib is unavailable in this environment")
    query = urlencode(params)
    url = f"{ALPHA_VANTAGE_BASE}?{query}"
    req = Request(url, headers={"User-Agent": "alpha-vantage-client/1.0"})
    with urlopen(req, timeout=timeout) as resp:  # nosec - controlled URL base
        body = resp.read().decode("utf-8", errors="replace")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Non-JSON response from Alpha Vantage") from exc
    if isinstance(payload, dict):
        for key in ("Error Message", "Note", "Information"):
            if key in payload and payload[key]:
                raise RuntimeError(str(payload[key]))
    return payload


def _parse_eps_items(items: Iterable[Dict[str, Any]], start: date, end: date) -> Dict[date, float]:
    values: Dict[date, float] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        fiscal = item.get("fiscalDateEnding")
        eps = as_float(item.get("reportedEPS"))
        if eps is None:
            continue
        try:
            fiscal_date = datetime.strptime(str(fiscal), "%Y-%m-%d").date()
        except Exception:
            continue
        if start <= fiscal_date <= end and fiscal_date not in values:
            values[fiscal_date] = eps
    return values


def fetch_eps_series(symbol: str, api_key: str, start: date, end: date) -> List[Tuple[date, float]]:
    payload = http_json({"function": "EARNINGS", "symbol": symbol, "apikey": api_key})
    if not isinstance(payload, dict):
        return []

    series: Dict[date, float] = {}
    for key in ("quarterlyEarnings", "annualEarnings"):
        subset = _parse_eps_items(payload.get(key) or [], start, end)
        for fiscal_date, eps in subset.items():
            series.setdefault(fiscal_date, eps)
    return sorted(series.items(), key=lambda item: item[0])


def write_eps_csv(path: Path, symbol: str, rows: Iterable[Tuple[date, float]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["stock_symbol", "date", "EPS", "PE_ratio"])
        for fiscal_date, eps in rows:
            writer.writerow([symbol, fiscal_date.strftime("%Y-%m-%d"), f"{eps:g}", ""])
            count += 1
    return count


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("alpha_vantage_api_key")
    if not api_key:
        print("[ERROR] Missing environment variable 'alpha_vantage_api_key'.", file=sys.stderr)
        sys.exit(2)

    try:
        symbols, start, end, output_dir = load_config(args.config)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    any_success = False
    for idx, symbol in enumerate(symbols):
        out_path = output_dir / f"{symbol}_eps.csv"
        try:
            if out_path.exists():
                print(f"[SKIP] Found existing {out_path}; skipping {symbol}.")
                any_success = True
                continue

            print(f"[INFO] Fetching EPS data for {symbol} ...")
            eps_rows = fetch_eps_series(symbol, api_key, start, end)
            if not eps_rows:
                print(f"[WARN] No EPS data found for {symbol} between {start} and {end}.")
                continue

            written = write_eps_csv(out_path, symbol, eps_rows)
            print(f"[OK] Wrote {written} rows to {out_path}")
            any_success = True
            if idx + 1 < len(symbols):
                time.sleep(1.5)
        except Exception as exc:
            print(f"[ERROR] Failed to process {symbol}: {exc}", file=sys.stderr)

    if not any_success:
        print("[ERROR] No EPS data written. Check symbols and configuration.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
