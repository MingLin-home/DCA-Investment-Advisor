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

    return trajectory

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    # The cfg variable is intentionally kept within main for further use.
    print(f"Loaded configuration with {len(cfg)} top-level keys from '{args.config}'.")


if __name__ == "__main__":
    main()
