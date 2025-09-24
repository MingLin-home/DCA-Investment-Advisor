#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, List, Optional

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Impute daily rows for each symbol, ensuring 24h steps and "
            "filling missing numeric values via linear interpolation/extrapolation."
        )
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional override for base output directory. When omitted, the value "
            "from config.yaml (output_dir) is used."
        ),
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "stock_symbols" not in cfg or not isinstance(cfg["stock_symbols"], list):
        raise ValueError("config.yaml must define 'stock_symbols' as a list")

    for key in ("stock_start_date", "stock_end_date", "output_dir"):
        if key not in cfg:
            raise ValueError(f"config.yaml must define '{key}'")
        if isinstance(cfg[key], str):
            cfg[key] = cfg[key].strip()
        if cfg[key] in (None, ""):
            raise ValueError(f"'{key}' in config.yaml must not be empty")

    columns_cfg = cfg.get("columns")
    if columns_cfg is not None:
        if not isinstance(columns_cfg, list) or not all(isinstance(c, str) for c in columns_cfg):
            raise ValueError("'columns' in config.yaml must be a list of strings when provided")
        cfg["columns"] = [c.strip() for c in columns_cfg if c.strip()]

    numeric_cfg = cfg.get("numeric_columns")
    if numeric_cfg is not None:
        if not isinstance(numeric_cfg, list) or not all(isinstance(c, str) for c in numeric_cfg):
            raise ValueError("'numeric_columns' in config.yaml must be a list of strings when provided")
        cfg["numeric_columns"] = [c.strip() for c in numeric_cfg if c.strip()]

    output_columns_cfg = cfg.get("output_columns")
    if output_columns_cfg is not None:
        if not isinstance(output_columns_cfg, list) or not all(isinstance(c, str) for c in output_columns_cfg):
            raise ValueError("'output_columns' in config.yaml must be a list of strings when provided")
        cfg["output_columns"] = [c.strip() for c in output_columns_cfg if c.strip()]

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


DEFAULT_NUMERIC_COLS = ["avg_price", "ESP", "FCF", "PBR", "ROE"]
DEFAULT_OUTPUT_COLUMNS = [
    "stock_symbol",
    "date",
    "avg_price",
    "timestamp",
    "ESP",
    "FCF",
    "PBR",
    "ROE",
]


def _dedupe(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def resolve_numeric_columns(cfg: dict) -> List[str]:
    if cfg.get("numeric_columns"):
        numeric = [c.strip() for c in cfg["numeric_columns"] if c.strip()]
    elif cfg.get("columns"):
        numeric = [
            c.strip()
            for c in cfg["columns"]
            if c.strip() and c.strip() not in {"date", "timestamp", "stock_symbol"}
        ]
    else:
        numeric = list(DEFAULT_NUMERIC_COLS)

    if not numeric:
        raise ValueError("Unable to determine numeric columns from config.yaml")

    return _dedupe(numeric)


def resolve_output_columns(cfg: dict, numeric_cols: List[str]) -> List[str]:
    if cfg.get("output_columns"):
        output_columns = [c for c in cfg["output_columns"] if c]
    elif cfg.get("columns"):
        output_columns = ["stock_symbol", "date", *cfg["columns"]]
    else:
        output_columns = list(DEFAULT_OUTPUT_COLUMNS)

    mandatory = ["stock_symbol", "date", "timestamp"]
    for col in numeric_cols:
        if col not in output_columns:
            output_columns.append(col)
    for col in mandatory:
        if col not in output_columns:
            # Insert timestamp after date when possible for readability
            if col == "timestamp" and "date" in output_columns:
                insert_at = output_columns.index("date") + 1
                output_columns.insert(insert_at, col)
            else:
                output_columns.insert(0, col)

    return _dedupe(output_columns)


def impute_symbol(
    symbol: str,
    base_dir: str,
    start_d: date,
    end_d: date,
    numeric_cols: List[str],
    output_columns: List[str],
) -> Optional[str]:
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

    required_cols = _dedupe([
        "stock_symbol",
        "date",
        "timestamp",
        *numeric_cols,
        *output_columns,
    ])

    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Coerce numeric columns to float for interpolation
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[required_cols]

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

    for c in numeric_cols:
        if c not in merged_indexed:
            continue
        s = merged_indexed[c].astype(float)
        # Linear interpolation on index values, extrapolate both ends
        s_interp = s.interpolate(method="index", limit_direction="both")
        merged_indexed[c] = s_interp

    # Restore timestamp and date columns and order
    merged = merged_indexed.reset_index()
    # Ensure date string matches timestamp (guard against mismatches)
    merged["date"] = pd.to_datetime(merged["timestamp"], unit="s", utc=True).dt.strftime("%Y-%m-%d")

    # Final column order
    merged = merged[[col for col in output_columns if col in merged.columns]]

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

    base_output = args.output.strip() if isinstance(args.output, str) and args.output.strip() else cfg["output_dir"]

    symbols: List[str] = [str(s).strip().upper() for s in cfg["stock_symbols"]]
    start_d = parse_date(cfg.get("stock_start_date"))
    end_d = parse_date(cfg.get("stock_end_date"))

    if start_d > end_d:
        raise ValueError("stock_start_date must be on or before stock_end_date")

    numeric_cols = resolve_numeric_columns(cfg)
    output_columns = resolve_output_columns(cfg, numeric_cols)

    any_ok = False
    for sym in symbols:
        print(f"[INFO] Imputing {sym} from {start_d} to {end_d} ...")
        out = impute_symbol(sym, base_output, start_d, end_d, numeric_cols, output_columns)
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
