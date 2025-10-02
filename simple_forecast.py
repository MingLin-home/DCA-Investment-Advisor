"""Simple linear-regression forecaster for stored price series.

Running this module loads avg_price data for each configured ticker, fits a
one-parameter trending line ``k * (i - T) + b`` over the historical window, and
stores both JSON metrics and a PNG plot per symbol.

Example
-------
python simple_forecast.py --config ./config.yaml

Config requirements
-------------------
- ``output_dir`` must point to the data root created by the download scripts.
- ``stock_symbols`` provides tickers to process (``etf_symbols`` is optionally
  honoured for backward compatibility).
- ``simple_forecast_history_start_date`` and ``simple_forecast_history_end_date``
  delimit the inclusive date range using ``YYYY-MM-DD``.

Outputs
-------
For each ticker ``X`` the script reads ``<output_dir>/data/X.csv`` and produces:
- ``<output_dir>/forecast/X.json`` with ``pred_k``, ``pred_b``, ``pred_std``
- ``<output_dir>/forecast/X.png`` visualising price vs. fitted trend
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

import matplotlib

matplotlib.use("Agg")  # Ensure headless environments can write plots.
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = "./config.yaml"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple linear forecast over stored CSV data")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    return parser.parse_args(argv)


def load_config(path: str) -> Mapping[str, Any]:
    cfg_path = Path(path).expanduser()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise TypeError("Configuration file must contain a mapping of keys")
    return data


def _normalize_symbol_list(raw: Any, field: str) -> List[str]:
    if raw is None:
        return []
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
        raise TypeError(f"Config field '{field}' must be a list of tickers")
    result: List[str] = []
    for item in raw:
        symbol = str(item).strip().upper()
        if not symbol or symbol in result:
            continue
        result.append(symbol)
    return result


def gather_symbols(cfg: Mapping[str, Any]) -> List[str]:
    symbols = _normalize_symbol_list(cfg.get("stock_symbols"), "stock_symbols")
    if not symbols:
        raise ValueError("Config must define at least one ticker via 'stock_symbols'")

    legacy_etf = _normalize_symbol_list(cfg.get("etf_symbols"), "etf_symbols")
    for symbol in legacy_etf:
        if symbol not in symbols:
            symbols.append(symbol)
    return symbols


def resolve_output_paths(cfg: Mapping[str, Any]) -> tuple[Path, Path]:
    output_root = cfg.get("output_dir")
    if output_root is None or str(output_root).strip() == "":
        raise ValueError("Config must define a non-empty 'output_dir'")
    root_path = Path(str(output_root)).expanduser().resolve()
    data_dir = root_path / "data"
    forecast_dir = root_path / "forecast"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    forecast_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, forecast_dir


def parse_date(value: Any, field: str) -> pd.Timestamp:
    if value is None:
        raise ValueError(f"Missing required date field '{field}'")
    try:
        ts = pd.to_datetime(str(value).strip(), format="%Y-%m-%d", utc=False)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Invalid date format for '{field}': {value}") from exc
    if pd.isna(ts):
        raise ValueError(f"Invalid date for '{field}': {value}")
    return ts.normalize()


def load_price_series(
    data_dir: Path,
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> np.ndarray:
    csv_path = data_dir / f"{symbol}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found for symbol '{symbol}': {csv_path}")
    df = pd.read_csv(csv_path, usecols=["date", "avg_price"])
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", utc=False)
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    filtered = df.loc[mask].copy()
    filtered = filtered.dropna(subset=["avg_price"])
    filtered.sort_values("date", inplace=True)
    values = filtered["avg_price"].to_numpy(dtype=np.float64)
    if values.size == 0:
        raise ValueError(
            f"No data points remain for {symbol} between {start_date.date()} and {end_date.date()}"
        )
    return values


def fit_trend(values: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    T = values.size
    idx = np.arange(T, dtype=np.float64)
    x = (idx - T)  # matches specification k * (i - T) + b
    A = np.vstack([x, np.ones_like(x)]).T
    params, *_ = np.linalg.lstsq(A, values, rcond=None)
    k, b = params
    predictions = k * x + b
    residuals = values - predictions
    pred_std = float(np.std(residuals, ddof=0))
    return float(k), float(b), pred_std, idx, predictions


def write_results(
    forecast_dir: Path,
    symbol: str,
    k: float,
    b: float,
    std: float,
    idx: np.ndarray,
    values: np.ndarray,
    predictions: np.ndarray,
) -> None:
    json_path = forecast_dir / f"{symbol}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"pred_k": k, "pred_b": b, "pred_std": std}, handle, indent=2)
    png_path = forecast_dir / f"{symbol}.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(idx, values, color="tab:red", label="stock price")
    ax.plot(idx, predictions, color="tab:blue", label="predicted")
    ax.set_xlabel("index")
    ax.set_ylabel("avg_price")
    ax.set_title(symbol)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def run(cfg: Mapping[str, Any]) -> None:
    data_dir, forecast_dir = resolve_output_paths(cfg)
    start = parse_date(cfg.get("simple_forecast_history_start_date"), "simple_forecast_history_start_date")
    end = parse_date(cfg.get("simple_forecast_history_end_date"), "simple_forecast_history_end_date")
    if end < start:
        raise ValueError("simple_forecast_history_end_date must be on or after the start date")

    symbols = gather_symbols(cfg)
    LOGGER.info("Processing %d symbols", len(symbols))

    for symbol in symbols:
        try:
            values = load_price_series(data_dir, symbol, start, end)
            k, b, std, idx, predictions = fit_trend(values)
            write_results(forecast_dir, symbol, k, b, std, idx, values, predictions)
            LOGGER.info(
                "%-6s k=%7.4f b=%10.4f std=%7.4f (%d points)",
                symbol,
                k,
                b,
                std,
                values.size,
            )
        except Exception as exc:
            LOGGER.error("Failed to process %s: %s", symbol, exc)
            raise


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
