"""Entry point for training a dollar-cost-averaging buy strategy."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import torch
import yaml

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train buy strategy using settings from a YAML config file.",
    )
    parser.add_argument(
        "--config",
        default="./config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    config_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("Configuration file must contain a top-level mapping of settings.")

    return cfg


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def get_batch_gain(batched_termination_asset_value: torch.Tensor, alpha: float):
    """
    batched_termination_asset_value: tensor of shape (simulation_batch_size,)
    alpha: a float number > 0
    """
    std = batched_termination_asset_value.std(dim=0)  # shape (1,)
    mean = batched_termination_asset_value.mean(dim=0)  # shape (1,) 
    y = mean - alpha * std
    
    return y

def get_termination_asset_value(
    price_trajectory: torch.Tensor,
    buy_strategy: torch.Tensor,
    overbuy_penalty_factor: float | int = 1.0,
    sell_penalty_factor: float | int = 1.0,
): 
    """
    overbuy_penalty_factor: non-negative scalar scaling the overspending penalty.
    sell_penalty_factor: non-negative scalar scaling the penalty for selling.

    price_trajectory: tensor of shape (M, T) or (B, M, T).
      price_trajectory[i, t] (or price_trajectory[b, i, t]) is the price of the i-th stock at time t.
    buy_strategy: tensor of shape (M, T) or (B, M, T) matching price_trajectory.

    Return:
    termination_asset_value: total money I have at time t=T-1 for each trajectory, computed by
      Y = torch.sum(torch.sum(buy_strategy / price_trajectory, dim=-1) * price_trajectory[..., -1], dim=-1)
        + 1 - torch.sum(buy_strategy, dim=(-2, -1))
    price_trajectory is guaranteed to be positive.
    overbuy_penalty = torch.clamp(
        torch.sum(buy_strategy, dim=(-2, -1)) - 1,
        min=0,
    ) * overbuy_penalty_factor
    sell_penalty = (-buy_strategy.clamp(max=0)).sum(dim=[-1, -2]) * sell_penalty_factor
    termination_asset_value = Y - overbuy_penalty - sell_penalty

    """
    tensors = {
        "price_trajectory": price_trajectory,
        "buy_strategy": buy_strategy,
    }

    for name, tensor in tensors.items():
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if price_trajectory.ndim not in (2, 3):
        raise ValueError("price_trajectory must have shape (M, T) or (B, M, T)")
    if buy_strategy.ndim != price_trajectory.ndim:
        raise ValueError("buy_strategy must have the same number of dimensions as price_trajectory")
    if price_trajectory.shape != buy_strategy.shape:
        raise ValueError("price_trajectory and buy_strategy must have the same shape")

    original_ndim = price_trajectory.ndim
    if original_ndim == 2:
        price_trajectory = price_trajectory.unsqueeze(0)
        buy_strategy = buy_strategy.unsqueeze(0)

    batch_size, num_stocks, num_steps = price_trajectory.shape
    if num_stocks == 0 or num_steps == 0:
        raise ValueError("price_trajectory must have positive dimensions")

    device = price_trajectory.device
    dtype = (
        price_trajectory.dtype if torch.is_floating_point(price_trajectory) else torch.float32
    )

    if not torch.is_floating_point(buy_strategy):
        buy_strategy = buy_strategy.to(dtype=torch.float32)

    price_trajectory = price_trajectory.to(device=device, dtype=dtype)
    buy_strategy = buy_strategy.to(device=device, dtype=dtype)

    if torch.any(price_trajectory <= 0):
        raise ValueError("price_trajectory must contain strictly positive prices")

    shares_held = buy_strategy / price_trajectory
    total_shares = shares_held.sum(dim=-1)
    final_prices = price_trajectory[..., -1]

    spent = buy_strategy.sum(dim=(-2, -1))
    ones = spent.new_ones(spent.shape)
    portfolio_value = (total_shares * final_prices).sum(dim=-1)
    cash_remaining = ones - spent
    raw_value = portfolio_value + cash_remaining

    overbuy_penalty_factor_tensor = ones.new_tensor(float(overbuy_penalty_factor))
    sell_penalty_factor_tensor = ones.new_tensor(float(sell_penalty_factor))
    overbuy_penalty = torch.clamp(spent - ones, min=0) * overbuy_penalty_factor_tensor
    sell_penalty = (
        -buy_strategy.clamp(max=0)
    ).sum(dim=[-1, -2]) * sell_penalty_factor_tensor
    termination_asset_value = raw_value - overbuy_penalty - sell_penalty

    if original_ndim == 2:
        return termination_asset_value.squeeze(0)
    return termination_asset_value

def get_buy_strategy(price_trajectory: torch.Tensor, buy_W: torch.Tensor, buy_b: torch.Tensor):
    """
    compute how much money I should spend on each stock.
    price_trajectory: tensor of shape (M, T) or (B, M, T).
    buy_W : tensor of shape (M, M). Model parameters to compute the buying strategy.
    buy_b: tensor of shape (M,). Model parameters to compute the buying strategy.
    
    Return:
    buy_strategy: tensor of shape (M, T) or (B, M, T) matching price_trajectory.
    
    buy_strategy[:,t] := 1 / ( 1 + exp(-z)) where z=buy_W * price_trajectory[:, t] + buy_b
    """
    tensors = {
        "price_trajectory": price_trajectory,
        "buy_W": buy_W,
        "buy_b": buy_b,
    }

    for name, tensor in tensors.items():
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if price_trajectory.ndim not in (2, 3):
        raise ValueError("price_trajectory must have shape (M, T) or (B, M, T)")
    if buy_W.ndim != 2:
        raise ValueError("buy_W must be a 2D tensor of shape (M, M)")
    if buy_b.ndim != 1:
        raise ValueError("buy_b must be a 1D tensor of shape (M,)")

    if price_trajectory.ndim == 2:
        num_stocks, num_steps = price_trajectory.shape
    else:
        _, num_stocks, num_steps = price_trajectory.shape
    if num_stocks == 0 or num_steps == 0:
        raise ValueError("price_trajectory must have positive dimensions")
    if buy_W.shape != (num_stocks, num_stocks):
        raise ValueError("buy_W must have shape (M, M) matching price_trajectory")
    if buy_b.shape[0] != num_stocks:
        raise ValueError("buy_b length must match number of stocks (M)")

    device = price_trajectory.device
    dtype = price_trajectory.dtype if torch.is_floating_point(price_trajectory) else torch.float32

    price_trajectory = price_trajectory.to(device=device, dtype=dtype)
    buy_W = buy_W.to(device=device, dtype=dtype)
    buy_b = buy_b.to(device=device, dtype=dtype)

    if price_trajectory.ndim == 2:
        z = buy_W @ price_trajectory + buy_b.unsqueeze(1)
    else:
        z = torch.einsum("ij,bjt->bit", buy_W, price_trajectory) + buy_b.view(1, -1, 1)

    buy_strategy = z
    return buy_strategy

def gen_price_trajectory(
    init_price: torch.Tensor,
    simulate_time_interval: int,
    k_vector: torch.Tensor,
    b_vector: torch.Tensor,
    std_vector: torch.Tensor,
    T: int,
    batch_size: int | None = None,
):
    """
    Generate price trajectory for M stocks until time T.
    Supports optional batching across multiple simulations.
    
    init_price: tensor of shape (M,)
    k_vector, b_vector, std_vector: tensor of shape (M,)
    
    Return:
    price_trajectory: tensor of shape (M, T) when batch_size is None, otherwise (batch_size, M, T).
      price_trajectory[..., 0] = init_price and price_trajectory[..., t] = k * simulate_time_interval * t + b_vector + N(0, std_vector)
      where the noise term is sampled from a Gaussian with mean 0 and per-stock std of std_vector[i].

    """
    if not isinstance(simulate_time_interval, int):
        raise TypeError("simulate_time_interval must be an integer")
    if not isinstance(T, int):
        raise TypeError("T must be an integer")
    if T <= 0:
        raise ValueError("T must be a positive integer")
    if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
        raise ValueError("batch_size must be a positive integer when provided")

    if not torch.is_tensor(init_price):
        raise TypeError("init_price must be a torch.Tensor")
    if init_price.ndim != 1:
        raise ValueError("init_price must be a 1D tensor of shape (M,)")

    expected_len = init_price.shape[0]
    vectors = {
        "k_vector": k_vector,
        "b_vector": b_vector,
        "std_vector": std_vector,
    }

    for name, vec in vectors.items():
        if not torch.is_tensor(vec):
            raise TypeError(f"{name} must be a torch.Tensor")
        if vec.ndim != 1:
            raise ValueError(f"{name} must be a 1D tensor of shape (M,)")
        if vec.shape[0] != expected_len:
            raise ValueError(f"{name} must have the same length as init_price")

    device = init_price.device
    dtype = init_price.dtype if torch.is_floating_point(init_price) else torch.float32

    init_price = init_price.to(device=device, dtype=dtype)
    k_vector = k_vector.to(device=device, dtype=dtype)
    b_vector = b_vector.to(device=device, dtype=dtype)
    std_vector = std_vector.to(device=device, dtype=dtype)

    if torch.any(std_vector < 0):
        raise ValueError("std_vector elements must be non-negative")

    if batch_size is None:
        trajectory = torch.empty((expected_len, T), device=device, dtype=dtype)
        trajectory[:, 0] = init_price
    else:
        trajectory = torch.empty((batch_size, expected_len, T), device=device, dtype=dtype)
        trajectory[:, :, 0] = init_price

    if T > 1:
        time_points = torch.arange(1, T, device=device, dtype=dtype)
        trend = (
            k_vector.unsqueeze(1) * (simulate_time_interval * time_points)
            + b_vector.unsqueeze(1)
        )
        if batch_size is None:
            noise = torch.randn((expected_len, T - 1), device=device, dtype=dtype)
            noise = noise * std_vector.unsqueeze(1)
            trajectory[:, 1:] = trend + noise
        else:
            noise = torch.randn((batch_size, expected_len, T - 1), device=device, dtype=dtype)
            noise = noise * std_vector.unsqueeze(0).unsqueeze(-1)
            trajectory[:, :, 1:] = trend.unsqueeze(0) + noise

    trajectory = torch.clamp(trajectory, min=0.01)
    return trajectory


def run_training(cfg: Dict[str, Any]) -> Dict[str, Any]:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.get("use_cuda", True)
        else "cpu"
    )

    seed = cfg.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))

    stock_symbols = cfg.get("stock_symbols")
    if isinstance(stock_symbols, str):
        stock_symbols = [stock_symbols]
    if not isinstance(stock_symbols, Iterable) or not stock_symbols:
        raise ValueError("Configuration must define a non-empty 'stock_symbols' list")

    symbols = tuple(str(sym) for sym in stock_symbols)

    output_dir = Path(cfg.get("output_dir", "./outputs"))
    data_dir = output_dir / "data"
    forecast_dir = output_dir / "forcast"

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Stock data directory does not exist: {data_dir}")
    if not forecast_dir.is_dir():
        raise FileNotFoundError(f"Forecast directory does not exist: {forecast_dir}")

    end_date_cfg = cfg.get("stock_end_date")
    if not end_date_cfg:
        raise ValueError("Configuration must define 'stock_end_date'")
    end_date = pd.to_datetime(end_date_cfg).date()

    init_prices = []
    k_values = []
    b_values = []
    std_values = []

    for symbol in symbols:
        csv_path = data_dir / f"{symbol}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing price file for symbol '{symbol}': {csv_path}")

        df = pd.read_csv(csv_path)
        if "date" not in df.columns or "avg_price" not in df.columns:
            raise ValueError(
                f"Price file '{csv_path}' must contain 'date' and 'avg_price' columns"
            )

        df["date"] = pd.to_datetime(df["date"]).dt.date
        price_row = df.loc[df["date"] == end_date, "avg_price"]
        if price_row.empty:
            raise ValueError(
                f"Could not find price for symbol '{symbol}' on {end_date} in {csv_path}"
            )

        init_prices.append(float(price_row.iloc[-1]))

        forecast_path = forecast_dir / f"{symbol}.json"
        if not forecast_path.is_file():
            raise FileNotFoundError(
                f"Missing forecast file for symbol '{symbol}': {forecast_path}"
            )

        with forecast_path.open("r", encoding="utf-8") as f:
            forecast_data = json.load(f)

        try:
            k_values.append(float(forecast_data["pred_k"]))
            b_values.append(float(forecast_data["pred_b"]))
            std_values.append(float(forecast_data["pred_std"]))
        except KeyError as exc:
            raise KeyError(
                f"Forecast file '{forecast_path}' missing required field: {exc}"
            ) from exc

    init_price = torch.tensor(init_prices, dtype=torch.float32, device=device)
    k_vector = torch.tensor(k_values, dtype=torch.float32, device=device)
    b_vector = torch.tensor(b_values, dtype=torch.float32, device=device)
    std_vector = torch.tensor(std_values, dtype=torch.float32, device=device)

    num_stocks = init_price.shape[0]
    if num_stocks == 0:
        raise ValueError("At least one stock is required for training")

    train_cfg = cfg.get("train_buy_strategy") or {}
    if not isinstance(train_cfg, dict):
        raise ValueError("'train_buy_strategy' section must be a mapping if provided")

    def get_train_setting(key: str, default: Any) -> Any:
        """Prefer nested train_buy_strategy overrides, fall back to top level/default."""
        if key in train_cfg:
            return train_cfg[key]
        return cfg.get(key, default)

    simulation_batch_size = int(cfg.get("simulation_batch_size", 128))
    if simulation_batch_size <= 0:
        raise ValueError("'simulation_batch_size' must be a positive integer")

    simulation_T = int(cfg.get("simulation_T", 6))
    if simulation_T <= 0:
        raise ValueError("'simulation_T' must be a positive integer")

    simulate_time_interval = int(cfg.get("simulate_time_interval", 1))
    simulate_time_interval = max(simulate_time_interval, 1)

    lr = float(get_train_setting("lr", 1e-3))
    weight_decay = float(get_train_setting("weight_decay", 0.0))
    alpha = float(cfg.get("batch_gain_alpha", 0.1))
    if alpha <= 0:
        raise ValueError("'batch_gain_alpha' must be a positive float")

    overbuy_penalty_cfg = get_train_setting("overbuy_penalty_factor", None)
    sell_penalty_cfg = get_train_setting("sell_penalty_factor", None)
    legacy_penalty_cfg = get_train_setting("penalty_factor", None)

    if overbuy_penalty_cfg is None:
        overbuy_penalty_cfg = legacy_penalty_cfg if legacy_penalty_cfg is not None else 1.0
    if sell_penalty_cfg is None:
        sell_penalty_cfg = legacy_penalty_cfg if legacy_penalty_cfg is not None else 1.0

    overbuy_penalty_factor = float(overbuy_penalty_cfg)
    sell_penalty_factor = float(sell_penalty_cfg)

    if overbuy_penalty_factor < 0:
        raise ValueError("'overbuy_penalty_factor' must be a non-negative float")
    if sell_penalty_factor < 0:
        raise ValueError("'sell_penalty_factor' must be a non-negative float")

    max_num_iters = int(get_train_setting("max_num_iters", 1000))
    if max_num_iters <= 0:
        raise ValueError("'max_num_iters' must be a positive integer")

    log_interval = int(get_train_setting("log_interval", max(1, max_num_iters // 10)))
    log_interval = max(log_interval, 1)

    init_std = 1e-6 / math.sqrt(float(num_stocks))
    buy_W = torch.normal(
        mean=0.0,
        std=init_std,
        size=(num_stocks, num_stocks),
        device=device,
        dtype=torch.float32,
    )
    buy_W.requires_grad_()
    buy_b = torch.normal(
        mean=0.0,
        std=init_std,
        size=(num_stocks,),
        device=device,
        dtype=torch.float32,
    )
    buy_b.requires_grad_()

    optimizer_name = str(get_train_setting("optimizer", "sgd")).strip().lower()
    if optimizer_name == "sgd":
        optimizer_cls = torch.optim.SGD
    elif optimizer_name == "adam":
        optimizer_cls = torch.optim.Adam
    else:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. Expected one of: sgd, adam."
        )

    optimizer = optimizer_cls([
        buy_W,
        buy_b,
    ], lr=lr, weight_decay=weight_decay)

    for iteration in range(1, max_num_iters + 1):
        price_traj = gen_price_trajectory(
            init_price,
            simulate_time_interval,
            k_vector,
            b_vector,
            std_vector,
            simulation_T,
            batch_size=simulation_batch_size,
        )
        buy_strategy = get_buy_strategy(price_traj, buy_W, buy_b)
        batched_termination_asset_value = get_termination_asset_value(
            price_traj,
            buy_strategy,
            overbuy_penalty_factor,
            sell_penalty_factor,
        )
        batch_gain = get_batch_gain(batched_termination_asset_value, alpha)

        loss = -batch_gain
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iteration % log_interval == 0 or iteration == 1 or iteration == max_num_iters:
            with torch.no_grad():
                mean_val = batched_termination_asset_value.mean().item()
                std_val = batched_termination_asset_value.std(unbiased=False).item()
                print(
                    f"iter={iteration:6d} batch_gain={batch_gain.item(): .6f} "
                    f"mean={mean_val: .6f} std={std_val: .6f}"
                )

    buy_W_cpu = buy_W.detach().cpu()
    buy_b_cpu = buy_b.detach().cpu()
    init_price_cpu = init_price.detach().cpu()
    k_vector_cpu = k_vector.detach().cpu()
    b_vector_cpu = b_vector.detach().cpu()
    std_vector_cpu = std_vector.detach().cpu()

    with torch.no_grad():
        deterministic_price_traj = gen_price_trajectory(
            init_price,
            simulate_time_interval,
            k_vector,
            b_vector,
            torch.zeros_like(std_vector),
            simulation_T,
        )
        buy_strategy_now = get_buy_strategy(deterministic_price_traj, buy_W, buy_b)

        shares_held = buy_strategy_now / deterministic_price_traj
        total_shares = shares_held.sum(dim=1)
        final_prices = deterministic_price_traj[:, -1]
        termination_stock_value = float((total_shares * final_prices).sum().item())
        cash_spent = float(buy_strategy_now.sum().item())
        termination_cash_value = float(1.0 - cash_spent)
        overbuy_penalty_value = max(cash_spent - 1.0, 0.0) * overbuy_penalty_factor
        sell_penalty_value = (
            -buy_strategy_now.clamp(max=0)
        ).sum().item() * sell_penalty_factor
        termination_total_asset_value = (
            termination_stock_value
            + termination_cash_value
            - overbuy_penalty_value
            - sell_penalty_value
        )

        per_stock_action = {
            symbol: float(buy_strategy_now[idx, 0].item())
            for idx, symbol in enumerate(symbols)
        }

        evaluation_batch_size = simulation_batch_size
        stochastic_price_traj = gen_price_trajectory(
            init_price,
            simulate_time_interval,
            k_vector,
            b_vector,
            std_vector,
            simulation_T,
            batch_size=evaluation_batch_size,
        )
        stochastic_buy_strategy = get_buy_strategy(stochastic_price_traj, buy_W, buy_b)
        stochastic_termination_asset = get_termination_asset_value(
            stochastic_price_traj,
            stochastic_buy_strategy,
            overbuy_penalty_factor,
            sell_penalty_factor,
        )

        stochastic_buy_strategy_cpu = stochastic_buy_strategy.detach().cpu()

        termination_asset_mean = float(stochastic_termination_asset.mean().item())
        termination_asset_std = float(
            stochastic_termination_asset.std(unbiased=False).item()
        )

        shares_per_stock = (stochastic_buy_strategy / stochastic_price_traj).sum(dim=-1)
        shares_mean = shares_per_stock.mean(dim=0)
        shares_std = shares_per_stock.std(dim=0, unbiased=False)

        final_prices_stochastic = stochastic_price_traj[..., -1]
        stock_values = shares_per_stock * final_prices_stochastic
        stock_value_mean = stock_values.mean(dim=0)
        stock_value_std = stock_values.std(dim=0, unbiased=False)

        spent_per_simulation = stochastic_buy_strategy.sum(dim=(-2, -1))
        cash_per_simulation = 1.0 - spent_per_simulation
        cash_mean = float(cash_per_simulation.mean().item())
        cash_std = float(cash_per_simulation.std(unbiased=False).item())

        termination_metrics = {
            "termination_asset": {
                "mean": termination_asset_mean,
                "std": termination_asset_std,
            },
            "cash": {
                "mean": cash_mean,
                "std": cash_std,
            },
            "stocks": {
                symbol: {
                    "shares_mean": float(shares_mean[idx].item()),
                    "shares_std": float(shares_std[idx].item()),
                    "value_mean": float(stock_value_mean[idx].item()),
                    "value_std": float(stock_value_std[idx].item()),
                }
                for idx, symbol in enumerate(symbols)
            },
        }

    return {
        "buy_W": buy_W_cpu.numpy(),
        "buy_b": buy_b_cpu.numpy(),
        "init_price": init_price_cpu.numpy(),
        "k_vector": k_vector_cpu.numpy(),
        "b_vector": b_vector_cpu.numpy(),
        "std_vector": std_vector_cpu.numpy(),
        "symbols": symbols,
        "stochastic_buy_strategy": stochastic_buy_strategy_cpu.numpy(),
        "buy_action": {
            **per_stock_action,
            "termination_stock_value": float(termination_stock_value),
            "termination_cash_value": float(termination_cash_value),
            "termination_total_asset_value": float(termination_total_asset_value),
        },
        "termination_metrics": termination_metrics,
    }

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    print(f"Loaded configuration with {len(cfg)} top-level keys from '{args.config}'.")

    training_artifacts = run_training(cfg)

    output_dir = Path(cfg.get("output_dir", "./outputs"))
    ensure_directory(output_dir)

    buy_model_path = output_dir / "buy_model.npz"
    np.savez(
        buy_model_path,
        buy_W=training_artifacts["buy_W"],
        buy_b=training_artifacts["buy_b"],
    )
    print(f"Saved buy parameters to '{buy_model_path}'.")

    buy_action_path = output_dir / "buy_action_now.json"
    with buy_action_path.open("w", encoding="utf-8") as f:
        json.dump(training_artifacts["buy_action"], f, indent=2)
    print(f"Saved buy action recommendation to '{buy_action_path}'.")

    termination_asset_path = output_dir / "termination_asset.csv"
    metrics = training_artifacts["termination_metrics"]
    rows = [
        {
            "item": "termination_asset",
            "symbol": "",
            "mean": metrics["termination_asset"]["mean"],
            "std": metrics["termination_asset"]["std"],
        },
        {
            "item": "cash",
            "symbol": "",
            "mean": metrics["cash"]["mean"],
            "std": metrics["cash"]["std"],
        },
    ]

    for symbol, stock_metrics in metrics["stocks"].items():
        rows.append(
            {
                "item": "shares",
                "symbol": symbol,
                "mean": stock_metrics["shares_mean"],
                "std": stock_metrics["shares_std"],
            }
        )
        rows.append(
            {
                "item": "value",
                "symbol": symbol,
                "mean": stock_metrics["value_mean"],
                "std": stock_metrics["value_std"],
            }
        )

    pd.DataFrame.from_records(rows).to_csv(termination_asset_path, index=False)
    print(f"Saved termination asset metrics to '{termination_asset_path}'.")

    stochastic_buy_path = output_dir / "stochastic_buy_strategy.csv"
    stochastic_buy = training_artifacts["stochastic_buy_strategy"]
    symbols = training_artifacts["symbols"]
    batch_size, num_stocks, num_steps = stochastic_buy.shape
    flattened = stochastic_buy.reshape(batch_size, num_stocks * num_steps)
    columns = [
        f"{symbol}_t{step}"
        for symbol in symbols
        for step in range(num_steps)
    ]
    pd.DataFrame(flattened, columns=columns).to_csv(stochastic_buy_path, index=False)
    print(f"Saved stochastic buy strategy samples to '{stochastic_buy_path}'.")


if __name__ == "__main__":
    main()
