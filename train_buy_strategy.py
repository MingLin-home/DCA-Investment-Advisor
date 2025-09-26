"""Entry point for training a dollar-cost-averaging buy strategy."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def checkpoint_iteration_from_path(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("checkpoint_iter_"):
        return None
    suffix = stem.replace("checkpoint_iter_", "", 1)
    return int(suffix) if suffix.isdigit() else None


def load_latest_checkpoint(
    checkpoint_dir: Path,
    device: torch.device,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_num_iters: int,
) -> tuple[int, bool, Optional[Dict[str, Any]]]:
    if not checkpoint_dir.is_dir():
        return 0, False, None

    latest: tuple[int, Path] | None = None
    for path in checkpoint_dir.glob("checkpoint_iter_*.pt"):
        iteration = checkpoint_iteration_from_path(path)
        if iteration is None:
            continue
        if latest is None or iteration > latest[0]:
            latest = (iteration, path)

    if latest is None:
        return 0, False, None

    iteration, checkpoint_path = latest
    checkpoint = torch.load(checkpoint_path, map_location=device)

    checkpoint_metadata = checkpoint.get("model_metadata")
    if isinstance(checkpoint_metadata, dict):
        checkpoint_model_name = checkpoint_metadata.get("name")
        if checkpoint_model_name is not None:
            checkpoint_model_name = str(checkpoint_model_name).strip().lower()
            current_model_name = infer_model_name(model)
            if checkpoint_model_name != current_model_name:
                raise ValueError(
                    "Checkpoint model type does not match the configured model type."
                )

    model_state = checkpoint.get("model_state")
    optimizer_state = checkpoint.get("optimizer_state")

    if optimizer_state is None:
        raise ValueError(f"Checkpoint '{checkpoint_path}' is missing optimizer state.")

    if model_state is not None:
        model.load_state_dict(model_state)
    else:
        # Legacy checkpoint support for linear weights.
        buy_W_state = checkpoint.get("buy_W")
        buy_b_state = checkpoint.get("buy_b")
        if buy_W_state is None or buy_b_state is None:
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' is missing model parameters."
            )
        if not isinstance(model, LinearBuyStrategyModel):
            raise ValueError(
                "Legacy checkpoint with linear weights cannot be loaded into a non-linear model."
            )
        with torch.no_grad():
            model.linear.weight.data.copy_(buy_W_state.to(device))
            model.linear.bias.data.copy_(buy_b_state.to(device))

    model.to(device)
    optimizer.load_state_dict(optimizer_state)

    stored_max_iters = checkpoint.get("max_num_iters")
    if stored_max_iters is not None and stored_max_iters != max_num_iters:
        print(
            f"Warning: checkpoint '{checkpoint_path}' max_num_iters={stored_max_iters} "
            f"differs from current setting {max_num_iters}."
        )

    print(f"Loaded checkpoint from iteration {iteration} ({checkpoint_path}).")
    metadata_return = checkpoint_metadata if isinstance(checkpoint_metadata, dict) else None
    return iteration, iteration >= max_num_iters, metadata_return


def save_checkpoint(
    checkpoint_dir: Path,
    iteration: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_num_iters: int,
    model_metadata: Dict[str, Any],
    keep: int = 3,
) -> None:
    ensure_directory(checkpoint_dir)
    checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iteration:06d}.pt"
    checkpoint = {
        "iteration": iteration,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_metadata": model_metadata,
        "optimizer_state": optimizer.state_dict(),
        "max_num_iters": max_num_iters,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}.")

    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("checkpoint_iter_*.pt"):
        iter_idx = checkpoint_iteration_from_path(path)
        if iter_idx is not None:
            checkpoints.append((iter_idx, path))

    checkpoints.sort(key=lambda item: item[0], reverse=True)
    for _, old_path in checkpoints[keep:]:
        try:
            old_path.unlink(missing_ok=True)
            print(f"Removed old checkpoint {old_path}.")
        except FileNotFoundError:
            continue


def normalize_symbol_list(raw: Any, field_name: str) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)):
        raw = [raw]
    if not isinstance(raw, Iterable):
        raise TypeError(
            f"Config field '{field_name}' must be an iterable collection of symbols"
        )

    symbols: list[str] = []
    for item in raw:
        symbol = str(item).strip().upper()
        if not symbol or symbol in symbols:
            continue
        symbols.append(symbol)
    return symbols


class LinearBuyStrategyModel(nn.Module):
    """Linear mapping from current prices to buy allocations."""

    def __init__(self, input_dim: int, init_std: float) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        self.linear = nn.Linear(input_dim, input_dim)
        with torch.no_grad():
            nn.init.normal_(self.linear.weight, mean=0.0, std=init_std)
            nn.init.normal_(self.linear.bias, mean=0.0, std=init_std)

    def forward(self, price: torch.Tensor) -> torch.Tensor:
        return self.linear(price)


class ResidualMLPBuyStrategyModel(nn.Module):
    """Residual MLP where each block includes a skip connection."""

    class ResidualBlock(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.activation = nn.GELU()
            self.residual_proj: nn.Module | None = None
            if in_dim != out_dim:
                self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)

            self._reset_parameters()

        def _reset_parameters(self) -> None:
            nn.init.kaiming_normal_(self.linear.weight, nonlinearity="linear")
            nn.init.zeros_(self.linear.bias)
            if self.residual_proj is not None:
                nn.init.xavier_normal_(self.residual_proj.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x if self.residual_proj is None else self.residual_proj(x)
            out = self.linear(x)
            out = self.activation(out)
            return out + residual

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be a positive integer")
        if num_layers < 0:
            raise ValueError("mlp_layers must be a non-negative integer")

        blocks: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            block = ResidualMLPBuyStrategyModel.ResidualBlock(in_dim, hidden_dim)
            blocks.append(block)
            in_dim = hidden_dim

        self.blocks = nn.ModuleList(blocks)
        self.output_layer = nn.Linear(in_dim, input_dim)
        self._reset_output_parameters()

    def _reset_output_parameters(self) -> None:
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, price: torch.Tensor) -> torch.Tensor:
        hidden = price
        for block in self.blocks:
            hidden = block(hidden)
        delta = self.output_layer(hidden)
        return delta


def infer_model_name(model: nn.Module) -> str:
    if isinstance(model, LinearBuyStrategyModel):
        return "linear"
    if isinstance(model, ResidualMLPBuyStrategyModel):
        return "mlp"
    return model.__class__.__name__.lower()


def prepare_model_export(
    state_dict: Dict[str, torch.Tensor],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert a model state dict and metadata to numpy-friendly payload."""
    state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
    export: Dict[str, Any] = {}

    key_list: list[str] = []
    for idx, (key, tensor) in enumerate(state_dict.items()):
        key_list.append(key)
        export[f"param_{idx}"] = tensor.numpy()

    export["model_type"] = np.array(str(metadata.get("name", "")))
    export["state_keys"] = np.array(key_list)

    config = metadata.get("config")
    if config is not None:
        export["model_config_json"] = np.array(json.dumps(config))

    # Provide linear-specific shortcuts for backward compatibility when possible.
    if metadata.get("name") == "linear":
        weight = state_dict.get("linear.weight")
        bias = state_dict.get("linear.bias")
        if weight is not None and bias is not None:
            export["buy_W"] = weight.numpy()
            export["buy_b"] = bias.numpy()

    return export

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
        (-1*buy_strategy).clamp(min=0)
    ).sum(dim=[-1, -2]) * sell_penalty_factor_tensor
    termination_asset_value = raw_value - overbuy_penalty - sell_penalty

    if original_ndim == 2:
        return termination_asset_value.squeeze(0)
    return termination_asset_value

def get_buy_strategy(price_trajectory: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Compute per-stock spending decisions for each point in the trajectory."""
    if not torch.is_tensor(price_trajectory):
        raise TypeError("price_trajectory must be a torch.Tensor")
    if price_trajectory.ndim not in (2, 3):
        raise ValueError("price_trajectory must have shape (M, T) or (B, M, T)")
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")

    if price_trajectory.ndim == 2:
        num_stocks, num_steps = price_trajectory.shape
        batch_dims = ()
    else:
        batch_dims = price_trajectory.shape[:-2]
        num_stocks, num_steps = price_trajectory.shape[-2:]

    if num_stocks == 0 or num_steps == 0:
        raise ValueError("price_trajectory must have positive dimensions")

    device = price_trajectory.device
    dtype = price_trajectory.dtype if torch.is_floating_point(price_trajectory) else torch.float32
    price_trajectory = price_trajectory.to(device=device, dtype=dtype)

    if price_trajectory.ndim == 2:
        inputs = price_trajectory.transpose(0, 1).reshape(-1, num_stocks)
        outputs = model(inputs)
        if outputs.shape[-1] != num_stocks:
            raise ValueError("Model output dimension does not match number of stocks")
        buy_strategy = outputs.view(num_steps, num_stocks).transpose(0, 1)
    else:
        permuted = price_trajectory.permute(*range(len(batch_dims)), -1, -2)
        flattened = permuted.reshape(-1, num_stocks)
        outputs = model(flattened)
        if outputs.shape[-1] != num_stocks:
            raise ValueError("Model output dimension does not match number of stocks")
        reshaped = outputs.view(*batch_dims, num_steps, num_stocks)
        buy_strategy = reshaped.permute(*range(len(batch_dims)), -1, -2)

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
    device_cfg = cfg.get("device")
    if device_cfg is not None:
        device_name = str(device_cfg).strip().lower()
        if device_name == "cpu":
            device = torch.device("cpu")
        elif device_name == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA requested via 'device' config but no CUDA-enabled device is available."
                )
            device = torch.device("cuda")
        else:
            raise ValueError("'device' config must be either 'cpu' or 'cuda'")
    else:
        use_cuda_cfg = cfg.get("use_cuda")
        use_cuda = True if use_cuda_cfg is None else bool(use_cuda_cfg)
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and use_cuda
            else "cpu"
        )

    seed = cfg.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))

    stock_symbols = normalize_symbol_list(cfg.get("stock_symbols"), "stock_symbols")
    etf_symbols = normalize_symbol_list(cfg.get("etf_symbols"), "etf_symbols")

    merged_symbols: list[str] = []
    for collection in (stock_symbols, etf_symbols):
        for symbol in collection:
            if symbol not in merged_symbols:
                merged_symbols.append(symbol)

    if not merged_symbols:
        raise ValueError(
            "Configuration must define at least one symbol via 'stock_symbols' or 'etf_symbols'"
        )

    symbols = tuple(merged_symbols)

    output_dir_cfg = cfg.get("output_dir")
    if output_dir_cfg is None:
        output_dir_cfg = "./outputs"
    base_output_dir = Path(output_dir_cfg)
    data_dir = base_output_dir / "data"
    forecast_dir = base_output_dir / "forecast"
    train_output_dir = base_output_dir / "train_buy_strategy"
    checkpoint_dir = train_output_dir / "checkpoints"

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
        if key in train_cfg and train_cfg[key] is not None:
            return train_cfg[key]
        value = cfg.get(key)
        if value is None:
            return default
        return value

    simulation_batch_size = int(get_train_setting("simulation_batch_size", 128))
    if simulation_batch_size <= 0:
        raise ValueError("'simulation_batch_size' must be a positive integer")

    simulation_T = int(get_train_setting("simulation_T", None))

    optimizer_name = str(get_train_setting("optimizer", "sgd")).strip().lower()
    lr_cfg = get_train_setting("lr", None)
    weight_decay_cfg = get_train_setting("weight_decay", 0.0)

    if optimizer_name == "sgd":
        if lr_cfg is None:
            raise ValueError("'lr' must be provided when optimizer is 'sgd'")
        lr = float(lr_cfg)
        weight_decay = float(weight_decay_cfg)
    elif optimizer_name == "adam":
        lr = float(lr_cfg) if lr_cfg is not None else torch.optim.Adam.defaults["lr"]
        weight_decay = float(weight_decay_cfg)
    else:
        lr = None
        weight_decay = None
    gradient_norm_clip_cfg = get_train_setting("gradient_norm_clip", None)
    if gradient_norm_clip_cfg is None:
        gradient_norm_clip = None
    else:
        gradient_norm_clip = float(gradient_norm_clip_cfg)
        if gradient_norm_clip <= 0:
            raise ValueError("'gradient_norm_clip' must be a positive float")

    alpha = float(get_train_setting("batch_gain_alpha",None))
    simulate_time_interval = int(get_train_setting("simulate_time_interval",None))

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

    max_num_iters = int(get_train_setting("max_num_iters", None))
    if max_num_iters <= 0:
        raise ValueError("'max_num_iters' must be a positive integer")

    log_interval_cfg = get_train_setting("log_interval", None)
    if log_interval_cfg is None:
        log_interval_iters = max(1, max_num_iters // 1000)
    else:
        try:
            log_interval_value = float(log_interval_cfg)
        except (TypeError, ValueError) as exc:
            raise ValueError("'log_interval' must be a numeric value") from exc

        if log_interval_value <= 0:
            raise ValueError("'log_interval' must be positive")

        if log_interval_value >= 1:
            rounded = round(log_interval_value)
            if not math.isclose(log_interval_value, rounded, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("'log_interval' must be an integer when it is >= 1")
            log_interval_iters = int(rounded)
        else:
            log_interval_iters = int(round(log_interval_value * max_num_iters))
            log_interval_iters = max(1, log_interval_iters)

    checkpoint_interval = max(1, int(math.ceil(max_num_iters * 0.05)))

    init_std = 1e-6 / math.sqrt(float(num_stocks))
    model_name_cfg = get_train_setting("buy_strategy_model", "linear")
    model_name = str(model_name_cfg).strip().lower()

    if model_name == "linear":
        model: nn.Module = LinearBuyStrategyModel(num_stocks, init_std)
        model_config: Dict[str, Any] = {"init_std": init_std}
    elif model_name == "mlp":
        hidden_dim_cfg = get_train_setting("mlp_hidden_dim", None)
        layers_cfg = get_train_setting("mlp_layers", None)
        if hidden_dim_cfg is None or layers_cfg is None:
            raise ValueError(
                "'mlp_hidden_dim' and 'mlp_layers' must be provided when using the mlp model"
            )
        hidden_dim = int(hidden_dim_cfg)
        num_layers = int(layers_cfg)
        model = ResidualMLPBuyStrategyModel(num_stocks, hidden_dim, num_layers)
        model_config = {
            "mlp_hidden_dim": hidden_dim,
            "mlp_layers": num_layers,
        }
    else:
        raise ValueError(
            f"Unsupported buy strategy model '{model_name}'. Expected one of: linear, mlp."
        )

    model = model.to(device=device, dtype=torch.float32)
    model_metadata = {
        "name": model_name,
        "config": model_config,
    }

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. Expected one of: sgd, adam."
        )

    resume_iteration, final_checkpoint_loaded, checkpoint_metadata = load_latest_checkpoint(
        checkpoint_dir,
        device,
        model,
        optimizer,
        max_num_iters,
    )

    if checkpoint_metadata:
        merged_metadata = dict(checkpoint_metadata)
        merged_metadata.update(model_metadata)
        model_metadata = merged_metadata

    if resume_iteration > 0 and resume_iteration < max_num_iters:
        print(f"Resuming training from iteration {resume_iteration}.")
    if final_checkpoint_loaded:
        print("Final checkpoint detected; skipping training loop and exporting results.")

    training_start_time = time.time()
    model.train()

    for iteration in range(resume_iteration + 1, max_num_iters + 1):
        price_traj = gen_price_trajectory(
            init_price,
            simulate_time_interval,
            k_vector,
            b_vector,
            std_vector,
            simulation_T,
            batch_size=simulation_batch_size,
        )
        buy_strategy = get_buy_strategy(price_traj, model)
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
        if gradient_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_norm_clip
            )
        optimizer.step()

        iterations_completed = iteration - resume_iteration
        should_log = (
            iteration == resume_iteration + 1
            or iteration == max_num_iters
            or (iteration % log_interval_iters == 0)
        )

        if should_log:
            with torch.no_grad():
                mean_val = batched_termination_asset_value.mean().item()
                std_val = batched_termination_asset_value.std(unbiased=False).item()
                elapsed_time = max(time.time() - training_start_time, 0.0)
                if iterations_completed > 0 and elapsed_time > 0:
                    remaining_iters = max_num_iters - iteration
                    eta_seconds = (
                        elapsed_time / iterations_completed
                    ) * remaining_iters if remaining_iters > 0 else 0.0
                else:
                    eta_seconds = float("inf")
                eta_display = (
                    format_seconds(eta_seconds)
                    if math.isfinite(eta_seconds)
                    else "--:--:--"
                )
                print(
                    f"iter={iteration:6d} batch_gain={batch_gain.item(): .6f} "
                    f"mean={mean_val: .6f} std={std_val: .6f} eta={eta_display}"
                )

        if iteration % checkpoint_interval == 0 or iteration == max_num_iters:
            save_checkpoint(
                checkpoint_dir,
                iteration,
                model,
                optimizer,
                max_num_iters,
                model_metadata,
            )

    model_state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    init_price_cpu = init_price.detach().cpu()
    k_vector_cpu = k_vector.detach().cpu()
    b_vector_cpu = b_vector.detach().cpu()
    std_vector_cpu = std_vector.detach().cpu()

    model.eval()

    with torch.no_grad():
        deterministic_price_traj = gen_price_trajectory(
            init_price,
            simulate_time_interval,
            k_vector,
            b_vector,
            torch.zeros_like(std_vector),
            simulation_T,
        )
        buy_strategy_now = get_buy_strategy(deterministic_price_traj, model)

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
        stochastic_buy_strategy = get_buy_strategy(stochastic_price_traj, model)
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

    model_export = prepare_model_export(model_state_cpu, model_metadata)

    return {
        "model_state_dict": model_state_cpu,
        "model_metadata": model_metadata,
        "model_export": model_export,
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

    output_dir_cfg = cfg.get("output_dir")
    if output_dir_cfg is None:
        output_dir_cfg = "./outputs"
    base_output_dir = Path(output_dir_cfg)
    train_output_dir = base_output_dir / "train_buy_strategy"
    ensure_directory(train_output_dir)

    buy_model_path = train_output_dir / "buy_model.npz"
    np.savez(
        buy_model_path,
        **training_artifacts["model_export"],
    )
    print(f"Saved buy parameters to '{buy_model_path}'.")

    buy_action_path = train_output_dir / "buy_action_now.json"
    with buy_action_path.open("w", encoding="utf-8") as f:
        json.dump(training_artifacts["buy_action"], f, indent=2)
    print(f"Saved buy action recommendation to '{buy_action_path}'.")

    termination_asset_path = train_output_dir / "termination_asset.csv"
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

    stochastic_buy_path = train_output_dir / "stochastic_buy_strategy.csv"
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
