import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml


class SingleStockDataset:
    def __init__(
        self,
        stock_symbol: str,
        time_window: int,
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Loads stock CSV data into a tensor of shape (num_rows, num_columns).

        - CSV path: ./outputs/data/<stock_symbol>.csv
        - Default columns: ["avg_price", "timestamp", "ESP", "ROE"]
        - self.data[:, 0] = avg_price, etc.
        """

        if columns is None:
            columns = ["avg_price", "timestamp", "ESP", "ROE"]

        if not isinstance(columns, (list, tuple)) or len(columns) == 0:
            raise ValueError("columns must be a non-empty sequence of column names")

        if not isinstance(time_window, int) or time_window <= 0:
            raise ValueError("time_window must be a positive integer")

        self.stock_symbol: str = stock_symbol
        self.time_window: int = time_window
        self.columns: List[str] = list(columns)

        # Load CSV
        csv_path = os.path.join("outputs", "data", f"{stock_symbol}.csv")
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

        # Convert to tensor (float32); timestamps as float for tensor homogeneity
        self.data = torch.tensor(df.to_numpy(dtype=float), dtype=torch.float32)

        # Pre-compute per-column normalization statistics
        # Use population std (unbiased=False) and guard against zeros
        self.mean: torch.Tensor = self.data.mean(dim=0)
        std = self.data.std(dim=0, unbiased=False)
        self.std: torch.Tensor = torch.clamp(std, min=1e-8)

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

        # Optionally narrow sampling range from config.yaml using sampling_* keys
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
        """If config.yaml defines sampling_start_date/end_date, apply date range.

        Supported values per key:
        - 'YYYY-MM-DD' (ISO) or None/empty to ignore that bound.
        """
        cfg_path = os.path.join("config.yaml")
        if not os.path.isfile(cfg_path):
            return
        try:
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            # If config can't be read, skip applying sampling range
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



    def get_data_at_index(self, index: int) -> torch.Tensor:
        """Return feature vector [num_columns] at dataset row `index`.

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

    # --- Normalization helpers ---
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data using dataset mean/std per column.

        Accepts shape (..., num_columns) and broadcasts stats across leading dims.
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor")
        m = self.mean.to(device=data.device, dtype=data.dtype)
        s = self.std.to(device=data.device, dtype=data.dtype)
        return (data - m) / s

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Invert normalization: x = data * std + mean.

        Accepts shape (..., num_columns) and broadcasts stats across leading dims.
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor")
        m = self.mean.to(device=data.device, dtype=data.dtype)
        s = self.std.to(device=data.device, dtype=data.dtype)
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
