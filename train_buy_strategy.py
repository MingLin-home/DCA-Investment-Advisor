"""
Usage:
python train_buy_strategy.py --config config.yaml

"--config" is optional. it loads config.yaml by default.

Use pytorch torch for tensor operations.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import torch
import yaml

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = "./config.yaml"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search optimal DCA buy strategy")
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


def resolve_output_paths(cfg: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    output_root = cfg.get("output_dir")
    if output_root is None or str(output_root).strip() == "":
        raise ValueError("Config must define a non-empty 'output_dir'")

    root_path = Path(str(output_root)).expanduser().resolve()
    data_dir = root_path / "data"
    forecast_dir = root_path / "forecast"
    train_dir = root_path / "train_buy_strategy"

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not forecast_dir.is_dir():
        raise FileNotFoundError(f"Forecast directory not found: {forecast_dir}")
    train_dir.mkdir(parents=True, exist_ok=True)

    return data_dir, forecast_dir, train_dir


def load_forecast_params(forecast_dir: Path, symbol: str) -> tuple[float, float, float]:
    json_path = forecast_dir / f"{symbol}.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"Forecast JSON not found for symbol {symbol}: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        pred_k = float(payload["pred_k"])
        pred_b = float(payload["pred_b"])
        pred_std = float(payload["pred_std"])
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise KeyError(f"Missing forecast key {exc.args[0]} in {json_path}") from exc
    return pred_k, pred_b, pred_std


def load_init_price(data_dir: Path, symbol: str) -> float:
    csv_path = data_dir / f"{symbol}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found for symbol '{symbol}': {csv_path}")
    df = pd.read_csv(csv_path, usecols=["avg_price"])
    series = df["avg_price"].dropna()
    if series.empty:
        raise ValueError(f"No non-null 'avg_price' rows found in {csv_path}")
    return float(series.iloc[-1])


def gen_price_trajectory(
    init_price: float | torch.Tensor,
    simulate_time_interval: int,
    pred_k: float,
    pred_b: float,
    pred_std: float,
    T: int,
    batch_size: int | None = None,
) -> torch.Tensor:
    """
    Generate price trajectory for M stocks from time t=0 to t=T-1.
    init_price: tensor of shape (1,)
    pred_k, pred_b, pred_std: float numbers
    T: simulation_T in the config.yaml

    Return:
    price_trajectory: tensor of shape (batch_size, T)
      price_trajectory[..., 0] = init_price ;
      price_trajectory[..., t] = (pred_k * simulate_time_interval * t + pred_b) / (1 + N(0, pred_std**2)) for t>=1
      where the noise term is sampled from a Gaussian with mean 0 and std pred_std.
    """
    if T <= 0:
        raise ValueError("T must be a positive integer")
    if batch_size is None:
        batch_size = 1
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    init_tensor = (
        init_price.clone().detach() if isinstance(init_price, torch.Tensor) else torch.tensor([init_price], dtype=torch.float32)
    )
    if init_tensor.ndim == 0:
        init_tensor = init_tensor.unsqueeze(0)
    if init_tensor.numel() != 1:
        raise ValueError("init_price must contain exactly one value")
    init_tensor = init_tensor.to(dtype=torch.float32)
    device = init_tensor.device

    trajectory = torch.empty((batch_size, T), dtype=torch.float32, device=device)
    trajectory[:, 0] = init_tensor[0]

    if T > 1:
        steps = torch.arange(1, T, dtype=torch.float32, device=device)
        trend = pred_k * simulate_time_interval * steps + pred_b
        trend = trend.unsqueeze(0).expand(batch_size, -1)
        std = float(abs(pred_std))
        if std > 0.0:
            noise = torch.randn((batch_size, T - 1), dtype=torch.float32, device=device) * std
        else:
            noise = torch.zeros((batch_size, T - 1), dtype=torch.float32, device=device)

        denom = 1.0 + noise
        if torch.any(denom == 0):
            eps = torch.finfo(denom.dtype).eps
            denom = denom.masked_fill(denom == 0, eps)

        trajectory[:, 1:] = trend / denom

    return trajectory


def get_final_asset_value(
    price_trajectory: torch.Tensor,
    buy_strategy_a: float,
    buy_strategy_b: float,
    buy_strategy_c: float,
    pred_k: float,
    pred_b: float,
    pred_std: float,
) -> dict[str, torch.Tensor]:
    """
    Return the final asset value when applying buying strategy {a,b,c} on price_trajectory.

    price_trajectory: tensor of shape (B,T) where B is the batch size, T is the time length
    buy_strategy_a, buy_strategy_b: float numbers of buying strategy

    for t=0,1, ... , T-1, set predicted_price=k*t+b . Set r = price_trajectory[:,t] / predicted_price

    Assume that at t=0, we have 1 dollar as our total budget.
    At time t, we spend p_t of our current remaining cash to buy the stock.
    That is, at time t, we have remaining cash C_t = (1-p_0) * (1-p_1) * ... * (1 - p_{t-1}).
    We spend C_t * p_t to buy stock, therefore, we will buy num_shares_t = C_t * p_t / price_trajectory[:,t]

    We use the following rules to determine p_t:
    p_t is a piecewise linear function in r. x-axis is the value of r, y-axis is the value of p_t:
    - When 0 <= r <= buy_strategy_a , p_t is a linear segment from (0,1) to (buy_strategy_a, buy_strategy_b)
    - When buy_strategy_a <= r <= (buy_strategy_a + buy_strategy_c), p_t is a linear segment from (buy_strategy_a, buy_strategy_b) to (buy_strategy_a + buy_strategy_c, 0)
    - For other r values, p_t takes 0.

    Return:
    the total asset value we have at time t=T-1. The total asset value is the cash we have at hand and the stock_shares * stock_price at time T-1.
    Return this dict:
    {
        "stock_share": stock_share, # tensor of shape (B,) shares of stock we have at time T-1
        "stock_price": stock_price, # tensor of shape (B,) price of stock we have at time T-1
        "stock_value": stock_value, # tensor of shape (B,) total value of stock we have have at time T-1
        "cash": cash, # tensor of shape (B,) the cash we have at hand at time T-1
        "total_asset_value": total_asset_value , # cash + stock_value  at time T-1
        "buy_fraction": buy_fraction, # tensor of shape (B,T) fractions spent each step
        "cash_spent": cash_spent, # tensor of shape (B,T) dollars spent each step
        "shares_acquired": shares_acquired, # tensor of shape (B,T) shares bought each step
    }
    """
    if not isinstance(price_trajectory, torch.Tensor):
        raise TypeError("price_trajectory must be a torch.Tensor")
    if price_trajectory.ndim != 2:
        raise ValueError("price_trajectory must have shape (batch, T)")

    price = price_trajectory.detach().clone().to(dtype=torch.float32)
    batch_size, T = price.shape
    if T == 0:
        raise ValueError("price_trajectory must have at least one timestep")

    device = price.device
    dtype = price.dtype
    eps_value = torch.finfo(dtype).eps

    time_index = torch.arange(T, dtype=dtype, device=device)
    predicted_price = pred_k * time_index + pred_b
    predicted_price = predicted_price.unsqueeze(0).expand(batch_size, -1)
    safe_predicted = torch.clamp(predicted_price, min=eps_value)
    r = price / safe_predicted

    buy_fraction = torch.zeros_like(price)
    a_tensor = torch.tensor(float(buy_strategy_a), dtype=dtype, device=device)
    b_tensor = torch.tensor(float(buy_strategy_b), dtype=dtype, device=device)
    c_tensor = torch.tensor(float(buy_strategy_c), dtype=dtype, device=device)

    denom_a_value = 1.0 if abs(float(buy_strategy_a)) < eps_value else float(buy_strategy_a)
    denom_c_value = 1.0 if abs(float(buy_strategy_c)) < eps_value else float(buy_strategy_c)
    denom_a = torch.tensor(denom_a_value, dtype=dtype, device=device)
    denom_c = torch.tensor(denom_c_value, dtype=dtype, device=device)

    mask_segment1 = (r >= 0) & (r <= a_tensor)
    ratio_segment1 = torch.where(mask_segment1, r / denom_a, torch.zeros_like(r))
    segment1 = 1.0 + (b_tensor - 1.0) * ratio_segment1
    buy_fraction = torch.where(mask_segment1, segment1, buy_fraction)

    upper_bound = a_tensor + c_tensor
    mask_segment2 = (r >= a_tensor) & (r <= upper_bound)
    ratio_segment2 = torch.where(mask_segment2, (r - a_tensor) / denom_c, torch.zeros_like(r))
    segment2 = b_tensor * (1.0 - ratio_segment2)
    buy_fraction = torch.where(mask_segment2, segment2, buy_fraction)

    buy_fraction = torch.clamp(buy_fraction, min=0.0, max=1.0)

    cash = torch.ones(batch_size, dtype=dtype, device=device)
    stock_share = torch.zeros(batch_size, dtype=dtype, device=device)
    cash_spent = torch.zeros_like(price)
    shares_acquired = torch.zeros_like(price)

    for t in range(T):
        fraction_t = buy_fraction[:, t]
        spend_t = cash * fraction_t
        if torch.any(spend_t < 0):
            min_spent = spend_t.amin().item()
            raise RuntimeError(f"Negative spend encountered at step {t}: {min_spent:.6f}")
        cash_spent[:, t] = spend_t

        price_t = torch.clamp(price[:, t], min=eps_value)
        shares_t = spend_t / price_t
        shares_acquired[:, t] = shares_t

        stock_share = stock_share + shares_t
        cash = cash - spend_t
        if torch.any(cash < 0):
            min_cash = cash.amin().item()
            raise RuntimeError(f"Negative cash balance encountered at step {t}: {min_cash:.6f}")

    stock_price = price[:, -1]
    stock_value = stock_share * stock_price
    total_asset_value = cash + stock_value

    return {
        "stock_share": stock_share,
        "stock_price": stock_price,
        "stock_value": stock_value,
        "cash": cash,
        "total_asset_value": total_asset_value,
        "buy_fraction": buy_fraction,
        "cash_spent": cash_spent,
        "shares_acquired": shares_acquired,
    }


def find_optimal_buy_strategy(cfg: Mapping[str, Any]) -> None:
    """
    Find optimal buy strategy for stock.

    cfg: configuration dict, by default loaded from ./config.yaml

    For each stock X in config.yaml "stock_symbols":
    - Its current price can be read from <output_dir>/data/X.csv file. The last line (date) "avg_price" is its current price. Use this as init_price.
    - Its forecasting parameter pred_k/b/std is stored in <output_dir>/forecast/X.json

    Do the following:
    - Use gen_price_trajectory to get price_trajectory of shape (batch_size, T)
    - split buy_strategy_a/b/c into grid_size intervals. a/b/c from/to value can be read from cfg. Say grid_size=10, there are 10*10*10 combinations of a/b/c
    - For each {a,b,c} triplet:
        + apply {a,b,c} on price_trajectory, get get_final_asset_value as Y
        + compute the mean value of Y and std of Y
        + compute buy_strategy_score = mean(Y) - buy_strategy_alpha * std(Y)
    - write these variables to  <output_dir>/train_buy_strategy/X.csv, each variable is a column. Sort by buy_strategy_score from large to small:
        + {a,b,c}
        + mean and std of Y,
        + buy_strategy_score
        + mean value of number of stock shares, stock values, cashes we have at time T-1.
        + At t=0, how much money we need to spend to buy the stock
        + At t=0, how much shares of stock we should buy
        + At t=0, the price of the stock

    Finally, print the first row of  <output_dir>/train_buy_strategy/X.csv in a human-friendly way.
    """
    train_cfg = cfg.get("train_buy_strategy")
    if not isinstance(train_cfg, Mapping):
        raise ValueError("Config must contain a 'train_buy_strategy' mapping")

    data_dir, forecast_dir, train_dir = resolve_output_paths(cfg)
    symbols = gather_symbols(cfg)

    simulation_T = int(train_cfg.get("simulation_T", 0))
    simulate_time_interval = int(train_cfg.get("simulate_time_interval", 1))
    batch_size = int(train_cfg.get("simulation_batch_size", 1))
    grid_size = int(train_cfg.get("grid_size", 0))
    alpha = float(train_cfg.get("buy_strategy_alpha", 0.0))
    device = torch.device(str(train_cfg.get("device", "cpu")))

    if simulation_T <= 0:
        raise ValueError("'simulation_T' must be a positive integer")
    if batch_size <= 0:
        raise ValueError("'simulation_batch_size' must be a positive integer")
    if grid_size <= 0:
        raise ValueError("'grid_size' must be a positive integer")

    a_from = float(train_cfg.get("buy_strategy_a_from", 0.0))
    a_to = float(train_cfg.get("buy_strategy_a_to", a_from))
    b_from = float(train_cfg.get("buy_strategy_b_from", 0.0))
    b_to = float(train_cfg.get("buy_strategy_b_to", b_from))
    c_from = float(train_cfg.get("buy_strategy_c_from", 0.0))
    c_to = float(train_cfg.get("buy_strategy_c_to", c_from))

    a_values = torch.linspace(a_from, a_to, grid_size).tolist()
    b_values = torch.linspace(b_from, b_to, grid_size).tolist()
    c_values = torch.linspace(c_from, c_to, grid_size).tolist()

    for symbol in symbols:
        init_price = load_init_price(data_dir, symbol)
        pred_k, pred_b, pred_std = load_forecast_params(forecast_dir, symbol)

        init_tensor = torch.tensor([init_price], dtype=torch.float32, device=device)
        with torch.no_grad():
            price_trajectory = gen_price_trajectory(
                init_tensor,
                simulate_time_interval,
                pred_k,
                pred_b,
                pred_std,
                simulation_T,
                batch_size=batch_size,
            )

        records: list[dict[str, float]] = []

        for a in a_values:
            for b in b_values:
                for c in c_values:
                    with torch.no_grad():
                        metrics = get_final_asset_value(
                            price_trajectory,
                            float(a),
                            float(b),
                            float(c),
                            pred_k,
                            pred_b,
                            pred_std,
                        )

                    total_asset = metrics["total_asset_value"]
                    mean_total = torch.mean(total_asset).item()
                    std_total = torch.std(total_asset, unbiased=False).item()
                    score = mean_total - alpha * std_total

                    record = {
                        "buy_strategy_a": float(a),
                        "buy_strategy_b": float(b),
                        "buy_strategy_c": float(c),
                        "mean_total_asset_value": mean_total,
                        "std_total_asset_value": std_total,
                        "buy_strategy_score": score,
                        "mean_stock_shares": metrics["stock_share"].mean().item(),
                        "mean_stock_value": metrics["stock_value"].mean().item(),
                        "mean_cash": metrics["cash"].mean().item(),
                        "initial_cash_spent": metrics["cash_spent"][:, 0].mean().item(),
                        "initial_stock_shares": metrics["shares_acquired"][:, 0].mean().item(),
                        "initial_stock_price": price_trajectory[:, 0].mean().item(),
                    }
                    records.append(record)

        df = pd.DataFrame.from_records(records)
        df.sort_values("buy_strategy_score", ascending=False, inplace=True)

        output_csv = train_dir / f"{symbol}.csv"
        df.to_csv(output_csv, index=False)

        best = df.iloc[0]
        print(
            f"{symbol}: best a={best['buy_strategy_a']:.4f}, b={best['buy_strategy_b']:.4f}, c={best['buy_strategy_c']:.4f} "
            f"-> total={best['mean_total_asset_value']:.4f}, std={best['std_total_asset_value']:.4f}, "
            f"score={best['buy_strategy_score']:.4f}, initial spend={best['initial_cash_spent']:.4f}, "
            f"initial shares={best['initial_stock_shares']:.4f}, price={best['initial_stock_price']:.4f}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    cfg = load_config(args.config)
    find_optimal_buy_strategy(cfg)


if __name__ == "__main__":
    main()
