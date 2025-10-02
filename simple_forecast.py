"""
Usage
-------
python simple_forecast.py --config ./config.yaml

"--config" is optional, by default it loads config.yaml

For each stock symbol X in config.yaml "stock_symbols" list, it loads stock price from <output_dir>/data/X.csv. Each line is a date, "avg_price" is the price of the stock at that date.

Sort by date, from oldest to newest. Load price from simple_forecast_history_start_date to simple_forecast_history_end_date as a Pytorch tensor P of shape (T,)

Fit a linear predictor {pred_k, pred_b}, such that:
denote t=torch.arange(T)

loss = torch.mean( ((pred_k * (t-T) + pred_k - P) / P)**2) is minimized.

After solving the optimal {pred_k, pred_b}, compute pred_std =torch.sqrt(torch.mean( ((pred_k * (t-T) + pred_k - P) / P)**2))

Write {pred_k, pred_b, pred_std} to <output_dir>/forecast/X.json:


Plot ``<output_dir>/forecast/X.png`` visualising price vs. fitted trend, blue line for historical price data, red line for predicted price. x-axis is the index of date starting from zero.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = "./config.yaml"
_EPS = 1e-6

plt.switch_backend("Agg")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a simple linear forecast for stock prices")
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
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, Mapping):
        raise TypeError("Configuration file must contain a mapping of keys")
    return cfg


def _normalize_symbol_list(raw: Any, field: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
        raise TypeError(f"Config field '{field}' must be a list of tickers")
    symbols: list[str] = []
    for item in raw:
        symbol = str(item).strip().upper()
        if not symbol or symbol in symbols:
            continue
        symbols.append(symbol)
    return symbols


def gather_symbols(cfg: Mapping[str, Any]) -> list[str]:
    symbols = _normalize_symbol_list(cfg.get("stock_symbols"), "stock_symbols")
    if not symbols:
        raise ValueError("Config must define at least one ticker via 'stock_symbols'")
    return symbols


def resolve_paths(cfg: Mapping[str, Any]) -> tuple[Path, Path]:
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


def _parse_date(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s, format="%Y-%m-%d", utc=True)
    except Exception as err:
        raise ValueError(f"Invalid date '{value}', expected YYYY-MM-DD") from err


def load_price_series(
    data_dir: Path,
    symbol: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> torch.Tensor:
    csv_path = data_dir / f"{symbol}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found for symbol '{symbol}': {csv_path}")

    df = pd.read_csv(csv_path, usecols=["date", "avg_price"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    if start_date is not None:
        df = df[df["date"] >= start_date]
    if end_date is not None:
        df = df[df["date"] <= end_date]

    prices = pd.to_numeric(df["avg_price"], errors="coerce")
    prices = prices.dropna()

    if prices.empty:
        raise ValueError(f"No valid 'avg_price' rows found for {symbol} in requested date range")

    return torch.from_numpy(prices.to_numpy(dtype="float32"))


def fit_trend(prices: torch.Tensor) -> tuple[float, float, float, torch.Tensor]:
    if prices.ndim != 1:
        raise ValueError("Price tensor must be one-dimensional")
    T = prices.numel()
    if T == 0:
        raise ValueError("Price tensor must contain at least one element")

    dtype = prices.dtype
    t = torch.arange(T, dtype=dtype, device=prices.device)
    x = t - float(T)

    safe_prices = torch.where(prices.abs() < _EPS, torch.full_like(prices, _EPS), prices)
    inv_var = 1.0 / (safe_prices * safe_prices)

    wx = inv_var * x
    sum_w = inv_var.sum()
    sum_wx = wx.sum()
    sum_wx2 = (wx * x).sum()
    sum_wy = (inv_var * prices).sum()
    sum_wxy = (wx * prices).sum()

    det = sum_wx2 * sum_w - sum_wx * sum_wx

    if T < 2 or torch.abs(det) < _EPS:
        pred_k = 0.0
        pred_b = float(prices[-1])
    else:
        pred_k = float((sum_wxy * sum_w - sum_wy * sum_wx) / det)
        pred_b = float((sum_wx2 * sum_wy - sum_wx * sum_wxy) / det)

    fitted = torch.tensor(pred_k, dtype=dtype) * x + torch.tensor(pred_b, dtype=dtype)
    rel_error = (fitted - prices) / torch.where(prices.abs() < _EPS, torch.full_like(prices, _EPS), prices)
    pred_std = float(torch.sqrt(torch.mean(rel_error * rel_error)))

    return pred_k, pred_b, pred_std, fitted


def write_forecast_json(forecast_dir: Path, symbol: str, pred_k: float, pred_b: float, pred_std: float) -> None:
    payload = {"pred_k": float(pred_k), "pred_b": float(pred_b), "pred_std": float(pred_std)}
    json_path = forecast_dir / f"{symbol}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def plot_forecast(
    forecast_dir: Path,
    symbol: str,
    prices: torch.Tensor,
    fitted: torch.Tensor,
) -> None:
    png_path = forecast_dir / f"{symbol}.png"

    t = range(prices.numel())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, prices.cpu().numpy(), color="tab:blue", label="history")
    ax.plot(t, fitted.cpu().numpy(), color="tab:red", label="trend")
    ax.set_title(f"{symbol} forecast")
    ax.set_xlabel("time index")
    ax.set_ylabel("avg_price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    cfg = load_config(args.config)
    symbols = gather_symbols(cfg)
    data_dir, forecast_dir = resolve_paths(cfg)

    start_date = _parse_date(cfg.get("simple_forecast_history_start_date"))
    end_date = _parse_date(cfg.get("simple_forecast_history_end_date"))

    LOGGER.info("Generating forecast for %d symbols", len(symbols))

    for symbol in symbols:
        LOGGER.info("Processing symbol %s", symbol)
        prices = load_price_series(data_dir, symbol, start_date, end_date)
        pred_k, pred_b, pred_std, fitted = fit_trend(prices)
        write_forecast_json(forecast_dir, symbol, pred_k, pred_b, pred_std)
        plot_forecast(forecast_dir, symbol, prices, fitted)
        LOGGER.info(
            "Finished %s: pred_k=%.6f pred_b=%.6f pred_std=%.6f",
            symbol,
            pred_k,
            pred_b,
            pred_std,
        )

    LOGGER.info("Forecast generation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
