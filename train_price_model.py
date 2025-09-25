"""Training script for linear trend predictor."""
import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, IterableDataset

from dataset_loader import SingleStockDataset, extract_price_trend


class MultiStockBatchDataset(IterableDataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Iterable dataset that yields full mini-batches sampled across stocks."""

    def __init__(
        self,
        stock_datasets: Sequence[SingleStockDataset],
        batch_size: int,
        history_window_size: int,
    ) -> None:
        if not stock_datasets:
            raise ValueError("stock_datasets must be a non-empty sequence")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self._datasets: Tuple[SingleStockDataset, ...] = tuple(stock_datasets)
        self.batch_size = int(batch_size)
        self.history_window_size = int(history_window_size)
        self._price_indices: Tuple[int, ...] = tuple(
            self._resolve_price_index(ds) for ds in self._datasets
        )

    @staticmethod
    def _resolve_price_index(dataset: SingleStockDataset) -> int:
        if "avg_price" not in dataset.col_index:
            raise KeyError(
                "Each dataset must contain 'avg_price' in its column index mapping"
            )
        return int(dataset.col_index["avg_price"])

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        base_seed = torch.initial_seed() % (2**32)
        np.random.seed(base_seed)
        rng = np.random.default_rng(base_seed)

        while True:
            xs: List[torch.Tensor] = []
            ys: List[torch.Tensor] = []

            for _ in range(self.batch_size):
                ds_idx = int(rng.integers(0, len(self._datasets)))
                dataset = self._datasets[ds_idx]
                price_idx = self._price_indices[ds_idx]
                sample = dataset.get_sample()

                sample_data = sample.get("sample_data")
                future_price = sample.get("future_price")
                if sample_data is None or future_price is None:
                    raise ValueError("get_sample() must return 'sample_data' and 'future_price'")

                x_np = np.asarray(sample_data, dtype=np.float32)
                if x_np.ndim != 2 or x_np.shape[0] != self.history_window_size:
                    raise ValueError(
                        "sample_data must be 2D with shape [history_window_size, num_columns]"
                    )
                if not (0 <= price_idx < x_np.shape[1]):
                    raise ValueError(
                        f"avg_price index {price_idx} out of bounds for sample_data with shape {x_np.shape}"
                    )

                x_series = x_np[:, price_idx]
                trend = extract_price_trend(np.asarray(future_price, dtype=np.float32))
                y_vec = np.array(
                    [trend["k"], trend["b"], trend["std"]],
                    dtype=np.float32,
                )

                xs.append(torch.tensor(x_series, dtype=torch.float32))
                ys.append(torch.tensor(y_vec, dtype=torch.float32))

            x_batch = torch.stack(xs, dim=0)
            y_batch = torch.stack(ys, dim=0)
            yield x_batch, y_batch


class LinearTrendModel(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(input_dim, 3, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


def save_model_npz(model: LinearTrendModel, path: str) -> None:
    """Persist model parameters as NumPy arrays."""
    weight_np = model.weight.detach().cpu().numpy()
    bias_np = model.bias.detach().cpu().numpy().reshape(1, 3)
    np.savez(path, W=weight_np, b=bias_np)


def load_model_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load model parameters stored via ``save_model_npz``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    with np.load(path) as data:
        weight_np = data["W"].astype(np.float32, copy=False)
        bias_np = data["b"].astype(np.float32, copy=False)

    if bias_np.ndim == 2:
        bias_np = bias_np.reshape(-1)
    elif bias_np.ndim != 1:
        raise ValueError(
            f"Unexpected bias array shape {bias_np.shape}; expected (3,) or (1, 3)"
        )

    return weight_np, bias_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear trend model")
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration file must define a mapping")
    return cfg


def ensure_positive_int(cfg: Dict[str, Any], key: str) -> int:
    value = cfg.get(key)
    if not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"Configuration key '{key}' must be a positive integer")
    return int(value)


def ensure_float(cfg: Dict[str, Any], key: str) -> float:
    value = cfg.get(key)
    if value is None:
        raise ValueError(f"Configuration key '{key}' is required")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Configuration key '{key}' must be a float") from exc


def build_datasets(stock_symbols: Sequence[str]) -> List[SingleStockDataset]:
    datasets: List[SingleStockDataset] = []
    for symbol in stock_symbols:
        datasets.append(SingleStockDataset(symbol))
    return datasets


def aggregate_history(x: torch.Tensor, agg_window_size: int) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("Input tensor for aggregation must be 2D [batch, history]")
    if x.size(1) % agg_window_size != 0:
        raise ValueError(
            "history_window_size must be divisible by agg_windows_size to compute aggregation"
        )
    new_len = x.size(1) // agg_window_size
    x_view = x.contiguous().view(x.size(0), new_len, agg_window_size)
    x_mean = x_view.mean(dim=2)
    x_std = x_view.std(dim=2, unbiased=False)
    return torch.cat((x_mean, x_std), dim=1)


def create_optimizer(model: LinearTrendModel, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    lr = ensure_float(cfg, "lr")
    weight_decay = ensure_float(cfg, "weight_decay")
    optimizer_name = str(cfg.get("optimizer", "sgd")).strip().lower()

    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'")


def set_model_parameters_from_numpy(
    model: LinearTrendModel, weight_np: np.ndarray, bias_np: np.ndarray
) -> None:
    """Copy NumPy weights/bias into a ``LinearTrendModel`` instance."""

    if weight_np.shape != tuple(model.weight.shape):
        raise ValueError(
            f"Weight shape mismatch: expected {tuple(model.weight.shape)}, got {tuple(weight_np.shape)}"
        )
    if bias_np.shape != tuple(model.bias.shape):
        raise ValueError(
            f"Bias shape mismatch: expected {tuple(model.bias.shape)}, got {tuple(bias_np.shape)}"
        )

    weight_tensor = torch.from_numpy(weight_np).to(model.weight.data.device)
    bias_tensor = torch.from_numpy(bias_np).to(model.bias.data.device)
    model.weight.data.copy_(weight_tensor)
    model.bias.data.copy_(bias_tensor)


def forecast_prices(
    cfg: Dict[str, Any],
    datasets: Sequence[SingleStockDataset],
    weight_np: np.ndarray,
    bias_np: np.ndarray,
    stock_end_date: Optional[str] = None,
) -> None:
    """Generate price trend forecasts for each stock symbol."""

    history_window_size = ensure_positive_int(cfg, "history_window_size")
    agg_window_size = ensure_positive_int(cfg, "agg_windows_size")
    if history_window_size % agg_window_size != 0:
        raise ValueError("history_window_size must be divisible by agg_window_size")

    cfg_output_dir = cfg.get("output_dir", "./outputs")
    output_dir = os.path.abspath(os.path.expanduser(str(cfg_output_dir)))

    if stock_end_date is None:
        stock_end_date = cfg.get("stock_end_date")

    if stock_end_date is None:
        raise ValueError("config must define non-empty 'stock_end_date'")

    stock_end_date = str(stock_end_date).strip()
    if not stock_end_date:
        raise ValueError("config must define non-empty 'stock_end_date'")

    forecast_dir = os.path.join(output_dir, "forecast")
    os.makedirs(forecast_dir, exist_ok=True)

    for dataset in datasets:
        symbol = dataset.stock_symbol
        if "avg_price" not in dataset.col_index:
            raise KeyError(
                f"Dataset for {symbol} is missing 'avg_price' column"
            )

        price_idx = int(dataset.col_index["avg_price"])

        end_ts = dataset.get_timestamp_from_date(stock_end_date)
        end_idx = dataset.get_index_from_timestamp(end_ts)

        start_idx = end_idx - history_window_size + 1
        if start_idx < 0:
            raise ValueError(
                f"Not enough history for symbol {symbol}: need {history_window_size} points"
            )

        normalized_series = dataset.data[start_idx : end_idx + 1, price_idx]
        if normalized_series.shape[0] != history_window_size:
            raise ValueError(
                f"Unexpected history length {normalized_series.shape[0]} for {symbol};"
                f" expected {history_window_size}"
            )

        x_tensor = torch.from_numpy(normalized_series.astype(np.float32, copy=False)).unsqueeze(0)
        x_agg = aggregate_history(x_tensor, agg_window_size)
        x_agg_np = x_agg.numpy()

        preds = x_agg_np @ weight_np + bias_np
        preds_flat = preds.reshape(-1)
        if preds_flat.shape[0] != 3:
            raise ValueError(
                f"Model output for {symbol} has unexpected shape {preds_flat.shape}"
            )

        price_mean = float(dataset.mean["avg_price"])
        price_std = float(dataset.std["avg_price"])

        pred_k = float(preds_flat[0] * price_std)
        pred_b = float(preds_flat[1] * price_std + price_mean)
        pred_std = float(preds_flat[2] * price_std)

        forecast_path = os.path.join(forecast_dir, f"{symbol}.json")
        output_payload = {
            "pred_k": pred_k,
            "pred_b": pred_b,
            "pred_std": pred_std,
        }

        with open(forecast_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    stock_symbols = cfg.get("stock_symbols")
    if not isinstance(stock_symbols, list) or not stock_symbols:
        raise ValueError("config must provide a non-empty list under 'stock_symbols'")

    history_window_size = ensure_positive_int(cfg, "history_window_size")
    agg_window_size = ensure_positive_int(cfg, "agg_windows_size")
    if history_window_size % agg_window_size != 0:
        raise ValueError(
            "history_window_size must be divisible by agg_windows_size"
        )
    aggregated_length = history_window_size // agg_window_size
    feature_dim = aggregated_length * 2

    batch_size = ensure_positive_int(cfg, "train_batch_size")
    num_workers = int(cfg.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    max_num_iters = ensure_positive_int(cfg, "max_num_iters")
    loss_avg_decay = ensure_float(cfg, "loss_avg_decay")
    if not (0.0 < loss_avg_decay < 1.0):
        raise ValueError("'loss_avg_decay' must be in the interval (0, 1)")

    maybe_seed = cfg.get("seed")
    if maybe_seed is not None:
        try:
            seed = int(maybe_seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("If provided, 'seed' must be an integer") from exc

    output_dir = cfg.get("output_dir", "./outputs")
    output_dir = os.path.abspath(os.path.expanduser(str(output_dir)))
    save_dir = os.path.join(output_dir, "train_price_model")
    os.makedirs(save_dir, exist_ok=True)

    final_path = os.path.join(save_dir, "model.npz")

    datasets = build_datasets(stock_symbols)

    model = LinearTrendModel(feature_dim)
    weight_np: Optional[np.ndarray] = None
    bias_np: Optional[np.ndarray] = None

    if os.path.isfile(final_path):
        weight_np, bias_np = load_model_npz(final_path)
        set_model_parameters_from_numpy(model, weight_np, bias_np)
        print(f"Loaded existing model parameters from {final_path}; skipping training.")
    else:
        dataset = MultiStockBatchDataset(
            stock_datasets=datasets,
            batch_size=batch_size,
            history_window_size=history_window_size,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=False,
        )

        optimizer = create_optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=max_num_iters,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        next_print_fraction = 0.001
        next_save_fraction = 0.01
        fraction_eps = 1e-12
        checkpoint_paths: List[str] = []

        model.train()
        data_iter = iter(dataloader)
        loss_moving_avg: Optional[float] = None
        for step in range(max_num_iters):
            x_batch, y_batch = next(data_iter)
            x_batch = x_batch.to(device=device, dtype=torch.float32)
            y_batch = y_batch.to(device=device, dtype=torch.float32)

            x_agg = aggregate_history(x_batch, agg_window_size)
            preds = model(x_agg)
            loss = F.mse_loss(preds, y_batch)
            naive_loss = torch.mean(y_batch**2)
            relative_loss = loss / (naive_loss + 1e-8)

            loss_value = float(loss.item())
            relative_loss_value = float(relative_loss.item())
            if loss_moving_avg is None:
                loss_moving_avg = loss_value
            else:
                loss_moving_avg = (
                    loss_avg_decay * loss_moving_avg
                    + (1.0 - loss_avg_decay) * loss_value
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step_idx = step + 1
            progress = step_idx / max_num_iters

            should_print = step == 0 or step_idx == max_num_iters
            # Advance logging thresholds so we emit once per 0.1% of progress even if a step covers multiple thresholds.
            while progress + fraction_eps >= next_print_fraction and next_print_fraction <= 1.0:
                should_print = True
                next_print_fraction += 0.001

            if should_print:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"step={step_idx:06d}, loss={loss_value:.6f}, "
                    f"loss_avg={loss_moving_avg:.6f}, "
                    f"relative_loss={relative_loss_value:.6f}, lr={current_lr:.6e}",
                    flush=True,
                )

            should_save = step_idx == max_num_iters
            # Advance saving thresholds so checkpoints align with every 1% of progress.
            while progress + fraction_eps >= next_save_fraction and next_save_fraction <= 1.0:
                should_save = True
                next_save_fraction += 0.01

            if should_save:
                checkpoint_path = os.path.join(save_dir, f"model_step_{step_idx:06d}.npz")
                save_model_npz(model, checkpoint_path)
                checkpoint_paths.append(checkpoint_path)
                if len(checkpoint_paths) > 5:
                    oldest_path = checkpoint_paths.pop(0)
                    try:
                        os.remove(oldest_path)
                    except OSError as exc:
                        print(
                            f"Warning: failed to remove old checkpoint {oldest_path}: {exc}",
                            flush=True,
                        )

        model.to("cpu")
        final_path = os.path.join(save_dir, "model.npz")
        save_model_npz(model, final_path)
        print(f"Model parameters saved to {final_path}")
        weight_np, bias_np = load_model_npz(final_path)
        set_model_parameters_from_numpy(model, weight_np, bias_np)

    if weight_np is None or bias_np is None:
        weight_np, bias_np = load_model_npz(final_path)

    forecast_prices(
        cfg=cfg,
        datasets=datasets,
        weight_np=weight_np,
        bias_np=bias_np,
    )


if __name__ == "__main__":
    main()
