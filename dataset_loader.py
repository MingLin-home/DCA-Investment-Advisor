import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
"""
This module uses NumPy arrays exclusively. PyTorch tensors are not used.
"""


class SingleStockDataset:
    def __init__(
        self,
        stock_symbol: str,
        cfg: Dict[str, Any],
    ) -> None:
        """
        Loads stock CSV data into a NumPy array of shape (num_rows, num_columns).

        - CSV path: <cfg['output_dir']>/data/<stock_symbol>.csv
        - Columns are read from cfg['columns'].
        - self.data[:, 0] = avg_price, etc.
        """

        self.stock_symbol: str = stock_symbol

        if cfg is None:
            raise ValueError("cfg must be provided for SingleStockDataset")
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a mapping of configuration values")

        # Make a shallow copy so internal mutations don't affect caller state.
        self.cfg: Dict[str, Any] = dict(cfg)

        output_dir = self.cfg.get("output_dir")
        if output_dir is None or str(output_dir).strip() == "":
            raise ValueError("config must define a non-empty 'output_dir'")
        self._output_root = os.path.abspath(os.path.expanduser(str(output_dir)))

        # Read columns from provided config mapping.
        cfg_columns = self.cfg.get("columns")
        if not isinstance(cfg_columns, list) or not cfg_columns or not all(isinstance(c, str) for c in cfg_columns):
            raise ValueError("config must define 'columns' as a non-empty list of strings")
        self.columns: List[str] = list(cfg_columns)

        # Load CSV
        csv_path = os.path.join(self._output_root, "data", f"{stock_symbol}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Validate requested columns exist
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing columns in CSV {csv_path}: {missing}. Available: {list(df.columns)}"
            )

        # Keep only requested columns in the specified order
        df = df[self.columns]

        # Convert to NumPy array (float32); timestamps as float for homogeneity
        data_raw = df.to_numpy(dtype=np.float32)
        if data_raw.size == 0:
            raise ValueError(f"CSV {csv_path} contains no rows to load")

        # Column index mapping for convenience
        self.col_index = {name: i for i, name in enumerate(self.columns)}

        # Identify columns that contain at least one finite datapoint
        num_columns = data_raw.shape[1]
        valid_counts = np.sum(~np.isnan(data_raw), axis=0)
        valid_mask = valid_counts > 0

        # Pre-compute per-column normalization statistics using NaN-aware ops.
        # Fallback to mean=0/std=1 for columns that are entirely NaN so we avoid
        # RuntimeWarnings from numpy and keep normalization stable.
        mean_vec = np.zeros(num_columns, dtype=np.float32)
        std_vec = np.ones(num_columns, dtype=np.float32)
        if np.any(valid_mask):
            valid_data = data_raw[:, valid_mask]
            mean_valid = np.nanmean(valid_data, axis=0).astype(np.float32)
            std_valid = np.nanstd(valid_data, axis=0, ddof=0).astype(np.float32)
            std_valid = np.maximum(std_valid, 1e-8).astype(np.float32)
            mean_vec[valid_mask] = mean_valid
            std_vec[valid_mask] = std_valid

        # Store vector stats for multi-column normalization
        self._mean_vec: np.ndarray = mean_vec
        self._std_vec: np.ndarray = std_vec

        # Store per-column stats as dict of scalars for single-column ops
        self.mean = {col: float(mean_vec[i]) for col, i in self.col_index.items()}
        self.std = {col: float(std_vec[i]) for col, i in self.col_index.items()}

        # Replace NaNs with column means (or zero for empty columns) before normalizing
        data_centered = np.where(np.isnan(data_raw), mean_vec, data_raw).astype(np.float32)

        # Normalize data immediately and keep both raw and normalized versions
        self._data_raw = data_raw
        self.data = (data_centered - mean_vec) / std_vec

        # Keep a copy of timestamps for date range sampling and direct lookup
        df_all = pd.read_csv(csv_path, usecols=["timestamp"])  # raises if column missing
        self._timestamps: np.ndarray = df_all["timestamp"].to_numpy(dtype=np.int64)
        # Ensure we know whether timestamps are sorted ascending (expected)
        self._timestamps_sorted: bool = bool(np.all(self._timestamps[:-1] <= self._timestamps[1:]))
        # Build mapping from timestamp -> first index for O(1) exact lookups
        self._ts_to_index = {}
        for i, ts in enumerate(self._timestamps.tolist()):
            self._ts_to_index.setdefault(int(ts), i)

        # Default sampling range is the full dataset (Python slice semantics: [from_idx:to_idx))
        self._from_idx: int = 0
        self._to_idx: int = self.data.shape[0]

        # Optionally narrow sampling range using sampling_* keys from config
        self._apply_sampling_range_from_config()

    # --- Utilities ---
    @staticmethod
    def _parse_date_to_ts(date_str: str) -> int:
        """Parse YYYY-MM-DD to epoch seconds at 00:00:00 UTC."""
        if date_str is None:
            raise ValueError("date string must not be None")
        s = str(date_str).strip()
        if s == "":
            raise ValueError("date string must not be empty")
        try:
            ts = pd.to_datetime(s, format="%Y-%m-%d", utc=True)
        except Exception as e:
            raise ValueError(f"Invalid date format '{date_str}', expected YYYY-MM-DD") from e
        return int(ts.timestamp())

    def __len__(self) -> int:
        """Return the total number of rows loaded in the dataset."""
        return int(self.data.shape[0])

    # --- Config helpers ---
    def _apply_sampling_range_from_config(self) -> None:
        """If config defines sampling_start_date/end_date, apply date range.

        Supported values per key:
        - 'YYYY-MM-DD' (ISO) or None/empty to ignore that bound.
        """
        cfg = getattr(self, "cfg", None)
        if not cfg:
            # No config available; nothing to apply
            return

        raw_start = cfg.get("sampling_start_date")
        raw_end = cfg.get("sampling_end_date")

        start_str = self._normalize_sampling_date(raw_start)
        end_str = self._normalize_sampling_date(raw_end)

        # Only apply if at least one bound is provided
        if start_str is not None or end_str is not None:
            self.set_sample_range_by_date(start_str, end_str)

    @staticmethod
    def _normalize_sampling_date(val: Optional[str]) -> Optional[str]:
        """Normalize date to 'YYYY-MM-DD' or return None if missing/empty.

        Accepts only ISO date string 'YYYY-MM-DD'. Whitespace is ignored.
        """
        if val is None:
            return None
        s = str(val).strip()
        if s == "":
            return None
        # Expect ISO-like date; validate via pandas with strict format
        try:
            dt = pd.to_datetime(s, format="%Y-%m-%d", utc=True)
            return dt.strftime("%Y-%m-%d")
        except Exception as e:
            raise ValueError(f"Unrecognized date format: {val}") from e

    def set_sampling_range_by_index(self, from_idx: int, to_idx: int) -> None:
        """
        Restrict sampling to data[from_idx:to_idx].

        - Uses Python slice semantics: from_idx inclusive, to_idx exclusive.
        - Validates bounds within [0, len(self.data)].
        """
        n = len(self)
        if not (isinstance(from_idx, int) and isinstance(to_idx, int)):
            raise TypeError("from and to must be integers")
        if not (0 <= from_idx <= to_idx <= n):
            raise ValueError(
                f"Invalid sampling range: from={from_idx}, to={to_idx}, dataset_size={n}"
            )
        self._from_idx = from_idx
        self._to_idx = to_idx

    # --- Date-based sampling ---
    def set_sample_range_by_date(self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> None:
        """
        Set sampling range by date boundaries (YYYY-MM-DD).

        - If from_date is None/empty, use earliest date in dataset.
        - If to_date is None/empty, use latest date in dataset.
        - Range is inclusive for both ends (samples may include rows on `to_date`).
        - Uses midnight UTC boundaries matching the CSV `timestamp` semantics.
        """

        # Determine numeric timestamp range [from_ts, to_ts]
        from_ts = None if (from_date is None or str(from_date).strip() == "") else self._parse_date_to_ts(from_date)
        to_ts = None if (to_date is None or str(to_date).strip() == "") else self._parse_date_to_ts(to_date)

        # Fallback to dataset bounds when missing
        if from_ts is None:
            from_ts = int(self._timestamps.min())
        if to_ts is None:
            to_ts = int(self._timestamps.max())

        if from_ts > to_ts:
            raise ValueError(f"from date must be <= to date (got {from_date} > {to_date})")

        # Convert to index range [start_idx, end_idx) using timestamps
        if self._timestamps_sorted:
            # left inclusive for start, right exclusive beyond end day
            start_idx = int(np.searchsorted(self._timestamps, from_ts, side="left"))
            end_idx = int(np.searchsorted(self._timestamps, to_ts, side="right"))
        else:
            mask = (self._timestamps >= from_ts) & (self._timestamps <= to_ts)
            if not np.any(mask):
                # No data in range; set empty range that will trigger error on sampling
                start_idx, end_idx = 0, 0
            else:
                idxs = np.nonzero(mask)[0]
                start_idx, end_idx = int(idxs.min()), int(idxs.max()) + 1

        # Clamp within dataset bounds and apply
        start_idx = max(0, min(start_idx, len(self)))
        end_idx = max(start_idx, min(end_idx, len(self)))
        self.set_sampling_range_by_index(start_idx, end_idx)

    # --- Direct lookup helpers ---
    def get_timestamp_from_date(self, date_str: str) -> int:
        """Return epoch-seconds timestamp for given date (YYYY-MM-DD, UTC midnight)."""
        return self._parse_date_to_ts(date_str)
    

    # --- Normalization helpers ---
    def normalize(self, data: np.ndarray, column_name: Optional[str] = None) -> np.ndarray:
        """Normalize data using dataset mean/std.

        - If column_name is None: expects shape (..., num_columns) and broadcasts
          vector stats across leading dims.
        - If column_name is provided: expects shape [N, M] where N is batch size
          and M is number of time points (days). Uses scalar stats for that column.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")

        if column_name is None:
            m = self._mean_vec.astype(data.dtype, copy=False)
            s = self._std_vec.astype(data.dtype, copy=False)
            return (data - m) / s
        else:
            if column_name not in self.col_index:
                raise KeyError(f"Unknown column: {column_name}. Available: {self.columns}")
            if data.ndim != 2:
                raise ValueError(
                    f"When column_name is provided, data must be 2D [N, M]; got shape {tuple(data.shape)}"
                )
            m = np.asarray(self.mean[column_name], dtype=data.dtype)
            s = np.asarray(self.std[column_name], dtype=data.dtype)
            return (data - m) / s

    def denormalize(self, data: np.ndarray, column_name: Optional[str] = None) -> np.ndarray:
        """Invert normalization: x = data * std + mean.

        - If column_name is None: accepts shape (..., num_columns) and broadcasts
          vector stats across leading dims.
        - If column_name is provided: expects shape [N, M] where N is batch size
          and M is number of time points (days). Uses scalar stats for that column.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")

        if column_name is None:
            m = self._mean_vec.astype(data.dtype, copy=False)
            s = self._std_vec.astype(data.dtype, copy=False)
            return data * s + m
        else:
            if column_name not in self.col_index:
                raise KeyError(f"Unknown column: {column_name}. Available: {self.columns}")
            if data.ndim != 2:
                raise ValueError(
                    f"When column_name is provided, data must be 2D [N, M]; got shape {tuple(data.shape)}"
                )
            m = np.asarray(self.mean[column_name], dtype=data.dtype)
            s = np.asarray(self.std[column_name], dtype=data.dtype)
            return data * s + m

    # --- Timestamp index lookup ---
    def get_index_from_timestamp(self, ts: int) -> int:
        """
        Return the largest index i such that timestamp[i] <= ts.

        - If no such index exists (i.e., ts is earlier than the first timestamp),
          raises ValueError.
        - Works whether the dataset timestamps are sorted or not, though sorted
          arrays are handled via binary search for efficiency.
        """
        try:
            ts_int = int(ts)
        except Exception as e:
            raise ValueError(f"Invalid timestamp value: {ts}") from e

        if self._timestamps.size == 0:
            raise ValueError("Dataset has no timestamps")

        if self._timestamps_sorted:
            # Rightmost insertion point minus one gives last index <= ts
            idx = int(np.searchsorted(self._timestamps, ts_int, side="right") - 1)
            if idx < 0:
                raise ValueError(f"No index found with timestamp <= {ts_int}")
            return idx
        else:
            # Fallback for unsorted timestamps
            mask = self._timestamps <= ts_int
            if not np.any(mask):
                raise ValueError(f"No index found with timestamp <= {ts_int}")
            return int(np.nonzero(mask)[0].max())

    # --- Accessors respecting normalization and sampling range ---
    def get_data_at_index(self, index: int) -> np.ndarray:
        """Return normalized feature vector [num_columns] at dataset row `index`.

        Valid indices are 0 <= index < len(self).
        """
        try:
            idx = int(index)
        except Exception as e:
            raise ValueError(f"Invalid index value: {index}") from e

        n = len(self)
        if not (0 <= idx < n):
            raise ValueError(f"Index {idx} out of bounds for dataset of size {n}")
        return self.data[idx]

    # --- Sampling API ---
    def get_sample(self) -> dict:
        """
        Randomly sample an index i such that:
        i ∈ [sampling_start_index + history_window_size, sampling_end_index - future_window_size)

        Returns a dict with:
        - sample_data: np.float32 array of shape [history_window_size, num_columns]
        - future_price: np.float32 array of shape [future_window_size]
        """
        # Load window sizes from cached config (set in __init__)
        cfg = getattr(self, "cfg", {}) or {}
        try:
            history_window_size = int(cfg.get("history_window_size"))
            future_window_size = int(cfg.get("future_window_size"))
        except Exception as e:
            raise ValueError("history_window_size and future_window_size must be set in config") from e

        # Compute valid sampling range considering previously applied date sampling
        start_index = int(self._from_idx) + history_window_size
        end_index_exclusive = int(self._to_idx) - future_window_size

        if start_index >= end_index_exclusive:
            raise ValueError(
                "Invalid sampling window: not enough data between sampling_start_date + "
                f"history_window_size ({start_index}) and sampling_end_date - future_window_size ({end_index_exclusive})."
            )

        # Randomly choose i in [start_index, end_index_exclusive)
        i = int(np.random.randint(start_index, end_index_exclusive))

        # Determine which column stores the price: always use 'avg_price'
        if "avg_price" not in self.col_index:
            raise KeyError(
                f"'avg_price' column not found in dataset columns {self.columns}"
            )
        price_idx = int(self.col_index["avg_price"])

        # Slice normalized data
        sample_np = self.data[i - history_window_size : i]
        future_np = self.data[i : i + future_window_size, price_idx]

        # Ensure NumPy float32 arrays (no torch tensors)
        sample_data = sample_np.astype(np.float32, copy=False)
        future_price = future_np.astype(np.float32, copy=False)

        return {
            "sample_data": sample_data,
            "future_price": future_price,
        }


def extract_price_trend(price_array: np.ndarray) -> dict:
    """Estimate linear trend p_t ≈ k*t + b for a 1D price series.

    - Input: `price_array` as a NumPy array of shape (N,).
    - Uses least squares regression over t = 0..N-1 to find k, b that minimize
      sum_t (k*t + b - p_t)^2.
    - Returns a dict with keys: 'k', 'b', 'std', where 'std' is the population
      standard deviation of residuals (p_t - (k*t + b)).
    """
    p = np.asarray(price_array, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"price_array must be 1D (N,), got shape {tuple(p.shape)}")
    N = int(p.shape[0])
    if N == 0:
        raise ValueError("price_array must be non-empty")
    if N == 1:
        k = 0.0
        b = float(p[0])
        price_std = 0.0
        return {"k": k, "b": b, "std": price_std}

    t = np.arange(N, dtype=np.float64)
    t_mean = t.mean()
    p_mean = p.mean()

    # Compute slope and intercept via closed-form least squares
    t_centered = t - t_mean
    p_centered = p - p_mean
    denom = np.sum(t_centered * t_centered)
    if denom == 0.0:
        # Fallback (shouldn't happen for N>=2 with distinct t)
        k = 0.0
    else:
        k = float(np.sum(t_centered * p_centered) / denom)
    b = float(p_mean - k * t_mean)

    residuals = p - (k * t + b)
    price_std = float(np.std(residuals, ddof=0))

    return {
        "k": k,
        "b": b,
        "std": price_std,
    }
