#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Any, List, Optional

import pandas as pd
import yaml

try:
    # Use stdlib to avoid adding dependencies
    from urllib.parse import urlencode
    from urllib.request import urlopen, Request
except Exception as e:  # pragma: no cover
    urlopen = None  # type: ignore


ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download historical stock data to CSVs via Alpha Vantage.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help=(
            "Base output directory (default: outputs); data saved under "
            "<output>/data and raw under <output>/raw_data"
        ),
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
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


def as_float_or_none(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return None
        return float(x)
    except Exception:
        return None


def http_json(params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    if urlopen is None:  # pragma: no cover
        raise RuntimeError("urllib is unavailable in this environment")
    query = urlencode(params)
    url = f"{ALPHA_VANTAGE_BASE}?{query}"
    req = Request(url, headers={"User-Agent": "alpha-vantage-client/1.0"})
    with urlopen(req, timeout=timeout) as resp:  # nosec - URL is constructed from fixed base + encoded params
        data = resp.read().decode("utf-8", errors="replace")
    try:
        j = json.loads(data)
    except json.JSONDecodeError:
        # Some responses can be CSV (not requested here) or HTML on errors
        raise RuntimeError("Non-JSON response from Alpha Vantage")
    # Handle API error notes
    if isinstance(j, dict):
        if "Error Message" in j:
            raise RuntimeError(j["Error Message"])  # type: ignore[index]
        if "Note" in j:
            raise RuntimeError(j["Note"])  # type: ignore[index]
        if "Information" in j:
            raise RuntimeError(j["Information"])  # type: ignore[index]
    return j


def fetch_price_history(symbol: str, start_d: date, end_d: date, api_key: str) -> pd.DataFrame:
    """
    Fetch daily OHLC using Alpha Vantage TIME_SERIES_DAILY (unadjusted) with outputsize=full.
    End date is inclusive. Returns DataFrame indexed by date with High/Low/Close columns.
    """
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": api_key,
    }
    j = http_json(params)

    key = next((k for k in j.keys() if k.startswith("Time Series")), None)
    if not key or key not in j:
        return pd.DataFrame()

    ts: Dict[str, Dict[str, str]] = j[key]
    rows = []
    for d_str, vals in ts.items():
        try:
            d = datetime.strptime(d_str, "%Y-%m-%d").date()
        except Exception:
            continue
        if d < start_d or d > end_d:
            continue
        try:
            high = as_float_or_none(vals.get("2. high"))
            low = as_float_or_none(vals.get("3. low"))
            close = as_float_or_none(vals.get("4. close"))
        except Exception:
            high = low = close = None
        if high is None or low is None or close is None:
            continue
        rows.append({"Date": d_str, "High": high, "Low": low, "Close": close})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])  # naive datetime
    df.set_index("Date", inplace=True)
    # Normalize index to date
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    # Sort ascending by date
    df.sort_index(inplace=True)
    return df


def try_get(d: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if k in d and d[k] is not None and d[k] != "":
            return d[k]
    return default


def _date_key(dstr: str) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(dstr).tz_localize(None).normalize()
    except Exception:
        return None


def fetch_eps_series(symbol: str, api_key: str) -> Dict[pd.Timestamp, float]:
    series: Dict[pd.Timestamp, float] = {}
    try:
        er = http_json({"function": "EARNINGS", "symbol": symbol, "apikey": api_key})
        if isinstance(er, dict):
            # Prefer quarterly granularity when available
            for arr_key in ("quarterlyEarnings", "annualEarnings"):
                arr = er.get(arr_key) or []
                if isinstance(arr, list):
                    for item in arr:
                        d = _date_key(item.get("fiscalDateEnding"))
                        val = as_float_or_none(item.get("reportedEPS"))
                        if d is not None and val is not None:
                            series[d] = val
    except Exception:
        pass
    return series


def fetch_fcf_series(symbol: str, api_key: str) -> Dict[pd.Timestamp, float]:
    series: Dict[pd.Timestamp, float] = {}
    try:
        cash = http_json({"function": "CASH_FLOW", "symbol": symbol, "apikey": api_key})
        if isinstance(cash, dict):
            for arr_key in ("quarterlyReports", "annualReports"):
                arr = cash.get(arr_key) or []
                if isinstance(arr, list):
                    for item in arr:
                        d = _date_key(item.get("fiscalDateEnding"))
                        ocf = as_float_or_none(item.get("operatingCashflow"))
                        capex = as_float_or_none(item.get("capitalExpenditures"))
                        if d is not None and ocf is not None and capex is not None:
                            series[d] = ocf - capex
    except Exception:
        pass
    return series


def _fetch_equity_series(symbol: str, api_key: str) -> Dict[pd.Timestamp, float]:
    eq: Dict[pd.Timestamp, float] = {}
    try:
        bal = http_json({"function": "BALANCE_SHEET", "symbol": symbol, "apikey": api_key})
        if isinstance(bal, dict):
            for arr_key in ("quarterlyReports", "annualReports"):
                arr = bal.get(arr_key) or []
                if isinstance(arr, list):
                    for item in arr:
                        d = _date_key(item.get("fiscalDateEnding"))
                        val = as_float_or_none(item.get("totalShareholderEquity"))
                        if d is not None and val is not None:
                            eq[d] = val
    except Exception:
        pass
    return eq


def _fetch_shares_series(symbol: str, api_key: str) -> Dict[pd.Timestamp, float]:
    sh: Dict[pd.Timestamp, float] = {}
    try:
        inc = http_json({"function": "INCOME_STATEMENT", "symbol": symbol, "apikey": api_key})
        if isinstance(inc, dict):
            for arr_key in ("quarterlyReports", "annualReports"):
                arr = inc.get(arr_key) or []
                if isinstance(arr, list):
                    for item in arr:
                        d = _date_key(item.get("fiscalDateEnding"))
                        val = as_float_or_none(item.get("commonStockSharesOutstanding"))
                        if d is not None and val is not None and val != 0:
                            sh[d] = val
    except Exception:
        pass
    return sh


def compute_pbr_series(symbol: str, api_key: str, hist: pd.DataFrame) -> Dict[pd.Timestamp, float]:
    """Compute P/B at fiscal dates: Close / (Equity / SharesOutstanding). No interpolation; exact dates only."""
    eq = _fetch_equity_series(symbol, api_key)
    sh = _fetch_shares_series(symbol, api_key)
    series: Dict[pd.Timestamp, float] = {}
    for d, equity in eq.items():
        shares = sh.get(d)
        if shares is None or shares == 0:
            continue
        bvps = equity / shares
        # Use close price on the exact fiscal date if present
        if d in hist.index:
            try:
                price = float(hist.loc[d, "Close"])  # type: ignore[index]
            except Exception:
                continue
            if bvps != 0:
                series[d] = price / bvps
    return series


def compute_roe_series(symbol: str, api_key: str) -> Dict[pd.Timestamp, float]:
    """Approximate ROE at fiscal dates: netIncome / totalShareholderEquity at same date. No interpolation."""
    eq = _fetch_equity_series(symbol, api_key)
    series: Dict[pd.Timestamp, float] = {}
    try:
        inc = http_json({"function": "INCOME_STATEMENT", "symbol": symbol, "apikey": api_key})
        if isinstance(inc, dict):
            for arr_key in ("quarterlyReports", "annualReports"):
                arr = inc.get(arr_key) or []
                if isinstance(arr, list):
                    for item in arr:
                        d = _date_key(item.get("fiscalDateEnding"))
                        ni = as_float_or_none(item.get("netIncome"))
                        if d is not None and ni is not None:
                            equity = eq.get(d)
                            if equity is not None and equity != 0:
                                series[d] = ni / equity
    except Exception:
        pass
    return series


def _extract_price_series(df: pd.DataFrame, col: str, symbol: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found for symbol {symbol}")
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")


def build_output_df(
    symbol: str,
    hist: pd.DataFrame,
    eps_series: Dict[pd.Timestamp, float],
    fcf_series: Dict[pd.Timestamp, float],
    pbr_series: Dict[pd.Timestamp, float],
    roe_series: Dict[pd.Timestamp, float],
) -> pd.DataFrame:
    h = _extract_price_series(hist, "High", symbol)
    l = _extract_price_series(hist, "Low", symbol)
    c = _extract_price_series(hist, "Close", symbol)
    avg_price = (h.astype(float) + l.astype(float) + c.astype(float)) / 3.0

    dates = pd.to_datetime(hist.index).tz_localize(None).normalize()
    timestamps = [
        int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
        for d in dates.to_pydatetime()
    ]

    # Map values only on exact available dates; do not impute missing values
    def _exact(dates_index: pd.DatetimeIndex, mapping: Dict[pd.Timestamp, float]) -> List[Optional[float]]:
        if not mapping:
            return [None] * len(dates_index)
        return [mapping.get(d) for d in dates_index]

    eps_vals = _exact(dates, eps_series)
    fcf_vals = _exact(dates, fcf_series)
    pbr_vals = _exact(dates, pbr_series)
    roe_vals = _exact(dates, roe_series)

    out = pd.DataFrame(
        {
            "stock_symbol": [symbol] * len(avg_price),
            "date": dates.strftime("%Y-%m-%d"),
            "avg_price": avg_price.astype(float).to_numpy(),
            "timestamp": timestamps,
            "ESP": eps_vals,
            "FCF": fcf_vals,
            "PBR": pbr_vals,
            "ROE": roe_vals,
        }
    )

    return out[["stock_symbol", "date", "avg_price", "timestamp", "ESP", "FCF", "PBR", "ROE"]]


def main():
    args = parse_args()

    api_key = os.environ.get("alpha_vantage_api_key")
    if not api_key:
        print(
            "[ERROR] Missing environment variable 'alpha_vantage_api_key' for Alpha Vantage API key.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Create output directories
    base_dir = args.output
    raw_dir = os.path.join(base_dir, "raw_data")
    os.makedirs(raw_dir, exist_ok=True)

    cfg = load_config(args.config)

    symbols: List[str] = [str(s).strip().upper() for s in cfg["stock_symbols"]]
    start_d = parse_date(cfg["stock_start_date"])
    end_d = parse_date(cfg["stock_end_date"]) if cfg.get("stock_end_date") is not None else date.today()

    if start_d > end_d:
        raise ValueError("stock_start_date must be on or before stock_end_date")

    any_ok = False
    for idx, sym in enumerate(symbols):
        try:
            print(f"[INFO] Fetching prices for {sym} from {start_d} to {end_d} via Alpha Vantage ...")
            hist = fetch_price_history(sym, start_d, end_d, api_key)
            if hist.empty:
                print(f"[WARN] No price data returned for {sym}; skipping.")
                continue

            print(f"[INFO] Fetching fundamentals for {sym} via Alpha Vantage (historical series, no interpolation) ...")
            eps_series = fetch_eps_series(sym, api_key)
            fcf_series = fetch_fcf_series(sym, api_key)
            pbr_series = compute_pbr_series(sym, api_key, hist)
            roe_series = compute_roe_series(sym, api_key)

            out_df = build_output_df(sym, hist, eps_series, fcf_series, pbr_series, roe_series)
            # Save final CSV under raw_data/{SYMBOL}.csv as requested
            out_path = os.path.join(raw_dir, f"{sym}.csv")
            out_df.to_csv(out_path, index=False)
            print(f"[OK] Wrote {len(out_df)} rows to {out_path}")
            any_ok = True

            # Basic rate-limit friendliness: small pause per symbol
            time.sleep(1.5)
        except Exception as e:
            print(f"[ERROR] Failed to process {sym}: {e}", file=sys.stderr)

    if not any_ok:
        print("[ERROR] No data written. Check symbols and configuration.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
