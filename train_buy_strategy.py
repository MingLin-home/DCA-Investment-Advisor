"""Entry point for training a dollar-cost-averaging buy strategy."""
from __future__ import annotations

import argparse
import json
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

def get_termination_asset_value(price_trajectory: torch.Tensor, buy_strategy: torch.Tensor):
    """
    price_trajectory: tensor of shape (M,T). price_trajectory[i,t] is the price of i-th stock at time t.
    buy_strategy: tensor of shape (M,T), spend buy_strategy[i,t] to buy i-th stock at time t.
    
    Return:
    termination_asset_value: total money I have at time t=T-1, computed by Y= torch.sum(torch.sum(buy_strategy / price_trajectory, dim=1) * price_trajectory[:, -1]) + 1 - torch.sum(buy_strategy)
    price_trajectory is guaranteed to be positive.
    penalty = torch.clamp(torch.sum(buy_strategy) - 1, min=0) * 1000
    termination_asset_value = Y - penalty

    """
    tensors = {
        "price_trajectory": price_trajectory,
        "buy_strategy": buy_strategy,
    }

    for name, tensor in tensors.items():
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if price_trajectory.ndim != 2:
        raise ValueError("price_trajectory must be a 2D tensor of shape (M, T)")
    if buy_strategy.ndim != 2:
        raise ValueError("buy_strategy must be a 2D tensor of shape (M, T)")
    if price_trajectory.shape != buy_strategy.shape:
        raise ValueError("price_trajectory and buy_strategy must have the same shape")

    num_stocks, num_steps = price_trajectory.shape
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
    total_shares = shares_held.sum(dim=1)
    final_prices = price_trajectory[:, -1]

    spent = buy_strategy.sum()
    one = spent.new_tensor(1.0)
    portfolio_value = (total_shares * final_prices).sum()
    cash_remaining = one - spent
    raw_value = portfolio_value + cash_remaining

    penalty = torch.clamp(spent - one, min=0) * 1000
    termination_asset_value = raw_value - penalty

    return termination_asset_value

def get_buy_strategy(price_trajectory: torch.Tensor, buy_W: torch.Tensor, buy_b: torch.Tensor):
    """
    compute how much money I should spend on each stock.
    price_trajectory: tensor of shape (M, T). price_trajectory[i,t] is the price of the i-th stock at time t.
    buy_W : tensor of shape (M, M). Model parameters to compute the buying strategy.
    buy_b: tensor of shape (M,). Model parameters to compute the buying strategy.
    
    Return:
    buy_strategy: tensor of shape (M,T). buy_strategy[i,t] is the dollar spent to buy the i-th stock at time t.
    
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

    if price_trajectory.ndim != 2:
        raise ValueError("price_trajectory must be a 2D tensor of shape (M, T)")
    if buy_W.ndim != 2:
        raise ValueError("buy_W must be a 2D tensor of shape (M, M)")
    if buy_b.ndim != 1:
        raise ValueError("buy_b must be a 1D tensor of shape (M,)")

    num_stocks, num_steps = price_trajectory.shape
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

    z = buy_W @ price_trajectory + buy_b.unsqueeze(1)
    buy_strategy = torch.sigmoid(z)
    return buy_strategy

def gen_price_trajectory(init_price: torch.Tensor, simulate_time_interval: int, k_vector: torch.Tensor, b_vector: torch.Tensor, std_vector: torch.Tensor, T: int):
    """
    Generate price trajectory for M stocks until time T.
    
    init_price: tensor of shape (M,)
    k_vector, b_vector, std_vector: tensor of shape (M,)
    
    Return:
    price_trajectory: tensor of shape (M,T) , price_trajectory[:, 0]=init_price , price_trajectory[:, t] = k * simulate_time_interval * t + b_vector + N(0, std_vector) where N(0, std_vector) is a vector sampled from random gaussian, mean 0 and std of std_vector[i] for the i-th stock.

    """
    if not isinstance(simulate_time_interval, int):
        raise TypeError("simulate_time_interval must be an integer")
    if not isinstance(T, int):
        raise TypeError("T must be an integer")
    if T <= 0:
        raise ValueError("T must be a positive integer")

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

    trajectory = torch.empty((expected_len, T), device=device, dtype=dtype)
    trajectory[:, 0] = init_price

    if T > 1:
        time_points = torch.arange(1, T, device=device, dtype=dtype)
        trend = (
            k_vector.unsqueeze(1) * (simulate_time_interval * time_points)
            + b_vector.unsqueeze(1)
        )
        noise = torch.randn((expected_len, T - 1), device=device, dtype=dtype)
        noise = noise * std_vector.unsqueeze(1)
        trajectory[:, 1:] = trend + noise

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

    simulation_batch_size = int(cfg.get("simulation_batch_size", 128))
    if simulation_batch_size <= 0:
        raise ValueError("'simulation_batch_size' must be a positive integer")

    simulation_T = int(cfg.get("simulation_T", 6))
    if simulation_T <= 0:
        raise ValueError("'simulation_T' must be a positive integer")

    simulate_time_interval = int(cfg.get("simulate_time_interval", 1))
    simulate_time_interval = max(simulate_time_interval, 1)

    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    alpha = float(cfg.get("batch_gain_alpha", 0.1))
    if alpha <= 0:
        raise ValueError("'batch_gain_alpha' must be a positive float")

    max_num_iters = int(cfg.get("max_num_iters", 1000))
    if max_num_iters <= 0:
        raise ValueError("'max_num_iters' must be a positive integer")

    log_interval = int(cfg.get("log_interval", max(1, max_num_iters // 10)))
    log_interval = max(log_interval, 1)

    buy_W = torch.zeros((num_stocks, num_stocks), dtype=torch.float32, device=device, requires_grad=True)
    buy_b = torch.zeros(num_stocks, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.SGD([
        buy_W,
        buy_b,
    ], lr=lr, weight_decay=weight_decay)

    for iteration in range(1, max_num_iters + 1):
        termination_values = []
        for _ in range(simulation_batch_size):
            price_traj = gen_price_trajectory(
                init_price,
                simulate_time_interval,
                k_vector,
                b_vector,
                std_vector,
                simulation_T,
            )
            buy_strategy = get_buy_strategy(price_traj, buy_W, buy_b)
            termination_value = get_termination_asset_value(price_traj, buy_strategy)
            termination_values.append(termination_value)

        batched_termination_asset_value = torch.stack(termination_values)
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
        penalty = max(cash_spent - 1.0, 0.0) * 1000.0
        termination_total_asset_value = termination_stock_value + termination_cash_value - penalty

        per_stock_action = {
            symbol: float(buy_strategy_now[idx, 0].item())
            for idx, symbol in enumerate(symbols)
        }

    return {
        "buy_W": buy_W_cpu.numpy(),
        "buy_b": buy_b_cpu.numpy(),
        "init_price": init_price_cpu.numpy(),
        "k_vector": k_vector_cpu.numpy(),
        "b_vector": b_vector_cpu.numpy(),
        "std_vector": std_vector_cpu.numpy(),
        "symbols": symbols,
        "buy_action": {
            **per_stock_action,
            "termination_stock_value": float(termination_stock_value),
            "termination_cash_value": float(termination_cash_value),
            "termination_total_asset_value": float(termination_total_asset_value),
        },
    }

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    print(f"Loaded configuration with {len(cfg)} top-level keys from '{args.config}'.")

    training_artifacts = run_training(cfg)

    output_dir = Path(cfg.get("output_dir", "./outputs"))
    ensure_directory(output_dir)

    buy_traj_path = output_dir / "buy_trajectory.npz"
    np.savez(
        buy_traj_path,
        buy_W=training_artifacts["buy_W"],
        buy_b=training_artifacts["buy_b"],
    )
    print(f"Saved buy parameters to '{buy_traj_path}'.")

    buy_action_path = output_dir / "buy_action_now.json"
    with buy_action_path.open("w", encoding="utf-8") as f:
        json.dump(training_artifacts["buy_action"], f, indent=2)
    print(f"Saved buy action recommendation to '{buy_action_path}'.")


if __name__ == "__main__":
    main()
