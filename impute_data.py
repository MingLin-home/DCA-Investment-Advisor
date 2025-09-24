#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Impute daily rows for each symbol, ensuring 24h steps and "
            "filling missing numeric values via linear interpolation/extrapolation."
        )
    )
    p.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    p.add_argument(
        "--output",
        default="outputs",
        help=(
            "Base output directory (default: outputs); reads from <output>/raw_data "
            "and writes imputed CSVs to <output>/data"
        ),
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    # Minimal validation
    if "stock_symbols" not in cfg or not isinstance(cfg["stock_symbols"], list):
        raise ValueError("config.yaml must define 'stock_symbols' as a list")
    if "stock_start_date" not in cfg:
        raise ValueError("config.yaml must define 'stock_start_date'")
    if "stock_end_date" not in cfg:
        raise ValueError("config.yaml must define 'stock_end_date'")
    return cfg


def parse_date(val: Optional[str]) -> date:
    if val is None:
        return date.today()
    s = str(val).strip().lower()
    if s == "today":
        return date.today()
    # Try ISO formats
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date format: {val}")


def date_to_epoch_utc(d: date) -> int:
    # midnight UTC for the given date
    dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return int(dt.timestamp())


def ensure_dirs(base_dir: str) -> tuple[str, str]:
    data_dir = os.path.join(base_dir, "data")
    raw_dir = os.path.join(base_dir, "raw_data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    return data_dir, raw_dir


def build_complete_calendar(start_d: date, end_d: date) -> pd.DataFrame:
    if start_d > end_d:
        raise ValueError("start date must be on or before end date")
    num_days = (end_d - start_d).days + 1
    all_dates = [start_d + timedelta(days=i) for i in range(num_days)]
    df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in all_dates],
        "timestamp": [date_to_epoch_utc(d) for d in all_dates],
    })
    return df


NUMERIC_COLS = ["avg_price", "ESP", "FCF", "PBR", "ROE"]


def impute_symbol(symbol: str, base_dir: str, start_d: date, end_d: date) -> Optional[str]:
    data_dir, raw_dir = ensure_dirs(base_dir)

    # Prefer Alpha Vantage style path first, fall back to processed data if needed
    candidates = [
        os.path.join(raw_dir, f"{symbol}.csv"),  # outputs/raw_data/SYM.csv
        os.path.join(data_dir, f"{symbol}.csv"),  # outputs/data/SYM.csv
        os.path.join(raw_dir, f"{symbol}.raw.csv"),  # outputs/raw_data/SYM.raw.csv
    ]
    src_path = next((p for p in candidates if os.path.exists(p)), None)
    if src_path is None:
        print(f"[WARN] No source CSV found for {symbol}; looked in {candidates}")
        return None

    try:
        df = pd.read_csv(src_path)
    except Exception as e:
        print(f"[ERROR] Failed to read {src_path}: {e}", file=sys.stderr)
        return None

    # Normalize column names and dtypes
    expected_cols = [
        "stock_symbol",
        "date",
        "avg_price",
        "timestamp",
        "ESP",
        "FCF",
        "PBR",
        "ROE",
    ]
    for c in expected_cols:
        if c not in df.columns:
            # Create missing numeric columns as NaN; stock_symbol will be set below
            df[c] = pd.NA

    # Coerce numeric columns to float
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only relevant columns
    df = df[expected_cols]

    # Build complete daily calendar from config dates
    calendar = build_complete_calendar(start_d, end_d)

    # Merge input onto full calendar
    merged = calendar.merge(df, on=["date", "timestamp"], how="left", suffixes=("", "_src"))

    # Set stock_symbol column
    merged["stock_symbol"] = symbol

    # Interpolate/extrapolate numeric columns linearly along time index
    # Use timestamp as numeric index to ensure 24h step and linear behavior
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged_indexed = merged.set_index("timestamp")

    for c in NUMERIC_COLS:
        s = merged_indexed[c].astype(float)
        # Linear interpolation on index values, extrapolate both ends
        s_interp = s.interpolate(method="index", limit_direction="both")
        merged_indexed[c] = s_interp

    # Restore timestamp and date columns and order
    merged = merged_indexed.reset_index()
    # Ensure date string matches timestamp (guard against mismatches)
    merged["date"] = pd.to_datetime(merged["timestamp"], unit="s", utc=True).dt.strftime("%Y-%m-%d")

    # Final column order
    merged = merged[[
        "stock_symbol",
        "date",
        "avg_price",
        "timestamp",
        "ESP",
        "FCF",
        "PBR",
        "ROE",
    ]]

    # Write output to data_dir/SYM.csv
    out_path = os.path.join(data_dir, f"{symbol}.csv")
    try:
        merged.to_csv(out_path, index=False)
    except Exception as e:
        print(f"[ERROR] Failed to write {out_path}: {e}", file=sys.stderr)
        return None

    return out_path


def main():
    args = parse_args()
    cfg = load_config(args.config)

    symbols: List[str] = [str(s).strip().upper() for s in cfg["stock_symbols"]]
    start_d = parse_date(cfg.get("stock_start_date"))
    end_d = parse_date(cfg.get("stock_end_date"))

    if start_d > end_d:
        raise ValueError("stock_start_date must be on or before stock_end_date")

    any_ok = False
    for sym in symbols:
        print(f"[INFO] Imputing {sym} from {start_d} to {end_d} ...")
        out = impute_symbol(sym, args.output, start_d, end_d)
        if out:
            print(f"[OK] Wrote imputed data to {out}")
            any_ok = True
        else:
            print(f"[WARN] Skipped {sym}; no output produced.")

    if not any_ok:
        print("[ERROR] No imputed data written. Check symbols and input files.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

