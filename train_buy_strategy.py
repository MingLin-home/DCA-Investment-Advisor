"""Entry point for training a dollar-cost-averaging buy strategy."""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import yaml
import torch

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

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    # The cfg variable is intentionally kept within main for further use.
    print(f"Loaded configuration with {len(cfg)} top-level keys from '{args.config}'.")


if __name__ == "__main__":
    main()
