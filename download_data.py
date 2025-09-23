#!/usr/bin/env python3
import argparse
import os
import sys
import time
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Any, List, Optional

import pandas as pd
import yaml

try:
    import yfinance as yf
except Exception as e:  # pragma: no cover
    yf = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download historical stock data to CSVs.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Base output directory (default: outputs); data saved under <output>/data and raw under <output>/raw_data",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # Basic validation
    if "stock_symbols" not in cfg or not isinstance(cfg["stock_symbols"], list):
        raise ValueError("config.yaml must define 'stock_symbols' as a list")
    if "stock_start_date" not in cfg:
        raise ValueError("config.yaml must define 'stock_start_date'")
    if "stock_end_date" not in cfg:
        raise ValueError("config.yaml must define 'stock_end_date'")
    return cfg


def parse_date(s: str) -> date:
    if isinstance(s, date):
        return s
    s_norm = str(s).strip()
    if s_norm.lower() == "today":
        return date.today()
    try:
        return datetime.strptime(s_norm, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid date format '{s}'; expected YYYY-MM-DD or 'today'") from e


def ensure_yfinance_available():  # pragma: no cover
    if yf is None:
        print("yfinance is not installed. Please `pip install yfinance`.", file=sys.stderr)
        sys.exit(2)


def fetch_price_history(symbol: str, start_d: date, end_d: date) -> pd.DataFrame:
    """
    Fetch daily OHLCV history using yfinance. End date is inclusive.
    Returns a DataFrame indexed by date with columns needed to compute avg price.
    """
    # yfinance treats 'end' as exclusive; add 1 day to include the end date
    end_exclusive = end_d + timedelta(days=1)
    df = yf.download(  # type: ignore[attr-defined]
        tickers=symbol,
        start=start_d.strftime("%Y-%m-%d"),
        end=end_exclusive.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    # When a single ticker is provided, columns are typically a flat index, but
    # yfinance may sometimes return MultiIndex columns; handle later on extraction.
    if df is None or df.empty:
        return pd.DataFrame()
    # Normalize index to date (drop time/tz)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def try_get(d: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def fetch_fundamentals(symbol: str) -> Dict[str, Optional[float]]:
    """
    Attempt to fetch point-in-time fundamentals to attach to every row.
    Values are static (latest known) and will be forward-filled across dates.
    Returns dict with keys: EPS, FCF, PBR, ROE. Missing values may be None.
    """
    t = yf.Ticker(symbol)  # type: ignore[attr-defined]
    eps = None
    pbr = None
    roe = None
    fcf = None

    # Info dict (may be slow or partially unavailable; try/except defensively)
    info = {}
    try:
        # newer yfinance
        if hasattr(t, "get_info"):
            info = t.get_info() or {}
        else:
            # fallback (deprecated in some versions)
            info = getattr(t, "info", {}) or {}
    except Exception:
        info = {}

    # EPS (TTM)
    eps = try_get(info, "trailingEps", "epsTrailingTwelveMonths")

    # Price-to-Book Ratio (P/B)
    pbr = try_get(info, "priceToBook")
    # If not present, try to compute from price / bookValue
    if pbr is None:
        try:
            book_value = try_get(info, "bookValue")
            current_price = try_get(info, "currentPrice")
            if book_value and current_price:
                pbr = float(current_price) / float(book_value) if float(book_value) != 0 else None
        except Exception:
            pbr = None

    # ROE
    roe = try_get(info, "returnOnEquity", "roe")

    # Free Cash Flow (most recent annual or quarterly)
    try:
        # Try new accessor
        cashflow_df = None
        if hasattr(t, "get_cashflow"):
            cashflow_df = t.get_cashflow(freq="annual")  # yfinance >=0.2
        else:
            cashflow_df = getattr(t, "cashflow", None)
        if isinstance(cashflow_df, pd.DataFrame) and not cashflow_df.empty:
            # Try a few likely row labels
            candidates = [
                "FreeCashFlow",
                "Free Cash Flow",
                "freeCashFlow",
            ]
            fcf_series = None
            for cand in candidates:
                if cand in cashflow_df.index:
                    fcf_series = cashflow_df.loc[cand]
                    break
            # Some versions put metrics as columns with index as periods; transpose if needed
            if fcf_series is None and "FreeCashFlow" in cashflow_df.columns:
                fcf_series = cashflow_df["FreeCashFlow"]
            if fcf_series is None and "Free Cash Flow" in cashflow_df.columns:
                fcf_series = cashflow_df["Free Cash Flow"]
            if fcf_series is not None and not pd.isna(fcf_series).all():
                # Take the most recent non-null value
                try:
                    fcf = float(pd.to_numeric(fcf_series, errors="coerce").dropna().iloc[0])
                except Exception:
                    pass
    except Exception:
        pass

    return {"EPS": as_float_or_none(eps), "FCF": as_float_or_none(fcf), "PBR": as_float_or_none(pbr), "ROE": as_float_or_none(roe)}


def as_float_or_none(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _extract_price_series(df: pd.DataFrame, col: str, symbol: str) -> pd.Series:
    """Return a 1-D Series for the given OHLC column, handling MultiIndex columns."""
    # Direct flat column
    if col in df.columns:
        s = df[col]
        # If for some reason this is a DataFrame (e.g., duplicate names), pick column by symbol or first
        if isinstance(s, pd.DataFrame):
            if symbol in s.columns:
                s = s[symbol]
            else:
                s = s.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce")

    # MultiIndex handling
    if isinstance(df.columns, pd.MultiIndex):
        # Try (col, symbol)
        if (col, symbol) in df.columns:
            s = df[(col, symbol)]
            return pd.to_numeric(s, errors="coerce")
        # Try selecting by first level and then by symbol or first available
        try:
            sub = df[col]
            if isinstance(sub, pd.DataFrame):
                if symbol in sub.columns:
                    s = sub[symbol]
                else:
                    s = sub.iloc[:, 0]
                return pd.to_numeric(s, errors="coerce")
        except Exception:
            pass

    raise KeyError(f"Column '{col}' not found for symbol {symbol}")


def build_output_df(symbol: str, hist: pd.DataFrame, fundamentals: Dict[str, Optional[float]]) -> pd.DataFrame:
    # Compute typical/average price as (High + Low + Close) / 3, robust to column shapes
    h = _extract_price_series(hist, "High", symbol)
    l = _extract_price_series(hist, "Low", symbol)
    c = _extract_price_series(hist, "Close", symbol)
    avg_price = (h.astype(float) + l.astype(float) + c.astype(float)) / 3.0

    dates = pd.to_datetime(hist.index).tz_localize(None).normalize()
    # Unix timestamp in seconds at 00:00:00 UTC for each date
    timestamps = [
        int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
        for d in dates.to_pydatetime()
    ]

    out = pd.DataFrame(
        {
            "stock_symbol": [symbol] * len(avg_price),
            "date": dates.strftime("%Y-%m-%d"),
            "avg_price": avg_price.astype(float).to_numpy(),
            "timestamp": timestamps,
            "ESP": fundamentals.get("EPS"),  # keep requested column name ESP in CSV
            "FCF": fundamentals.get("FCF"),
            "PBR": fundamentals.get("PBR"),
            "ROE": fundamentals.get("ROE"),
        }
    )

    # Ensure column order as specified
    return out[["stock_symbol", "date", "avg_price", "timestamp", "ESP", "FCF", "PBR", "ROE"]]


def main():
    args = parse_args()

    # Create output directories
    base_dir = args.output
    data_dir = os.path.join(base_dir, "data")
    raw_dir = os.path.join(base_dir, "raw_data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    cfg = load_config(args.config)

    symbols: List[str] = [str(s).strip().upper() for s in cfg["stock_symbols"]]
    start_d = parse_date(cfg["stock_start_date"])
    end_d = parse_date(cfg["stock_end_date"]) if cfg.get("stock_end_date") is not None else date.today()

    if start_d > end_d:
        raise ValueError("stock_start_date must be on or before stock_end_date")

    ensure_yfinance_available()

    any_ok = False
    for sym in symbols:
        try:
            print(f"[INFO] Fetching prices for {sym} from {start_d} to {end_d} ...")
            hist = fetch_price_history(sym, start_d, end_d)
            if hist.empty:
                print(f"[WARN] No price data returned for {sym}; skipping.")
                continue

            # Save raw price history for debugging
            try:
                raw_path = os.path.join(raw_dir, f"{sym}.raw.csv")
                hist.to_csv(raw_path)
                print(f"[DEBUG] Wrote raw data to {raw_path}")
            except Exception as e:
                print(f"[WARN] Failed to write raw data for {sym}: {e}")

            print(f"[INFO] Fetching fundamentals for {sym} ...")
            fundamentals = fetch_fundamentals(sym)

            out_df = build_output_df(sym, hist, fundamentals)
            out_path = os.path.join(data_dir, f"{sym}.csv")
            out_df.to_csv(out_path, index=False)
            print(f"[OK] Wrote {len(out_df)} rows to {out_path}")
            any_ok = True
        except Exception as e:
            print(f"[ERROR] Failed to process {sym}: {e}", file=sys.stderr)

    if not any_ok:
        print("[ERROR] No data written. Check symbols and configuration.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
