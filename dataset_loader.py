import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch


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

    def set_sampling_range(self, from_idx: int, to_idx: int) -> None:
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

    def get_sample(self) -> torch.Tensor:
        """
        Randomly and uniformly sample a contiguous window of length `time_window`
        from self.data[self._from_idx:self._to_idx].

        Returns: torch.Tensor with shape (time_window, num_columns).
        Raises: ValueError if time_window >= (to - from).
        """
        window = self.time_window
        start_min = self._from_idx
        start_max_exclusive = self._to_idx - window  # inclusive start_max = start_max_exclusive

        if window >= (self._to_idx - self._from_idx):
            raise ValueError(
                f"time_window ({window}) must be < (to - from) ({self._to_idx - self._from_idx})"
            )

        # Uniformly choose start index in [start_min, start_max] inclusive
        start_max = start_max_exclusive
        start_idx = int(torch.randint(low=start_min, high=start_max + 1, size=(1,)).item())
        end_idx = start_idx + window
        return self.data[start_idx:end_idx]

    # --- Date-based sampling ---
    def set_sample_date_range(self, from_date: Optional[str] = None, to_date: Optional[str] = None) -> None:
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
        self.set_sampling_range(start_idx, end_idx)

    # --- Direct lookup helpers ---
    def get_timestamp_from_date(self, date_str: str) -> int:
        """Return epoch-seconds timestamp for given date (YYYY-MM-DD, UTC midnight)."""
        return self._parse_date_to_ts(date_str)

    def get_data_at_timestamp(self, ts: int) -> torch.Tensor:
        """Return feature vector [num_columns] at exact timestamp `ts`.

        Raises ValueError if timestamp is not present in the dataset.
        """
        try:
            ts_int = int(ts)
        except Exception as e:
            raise ValueError(f"Invalid timestamp value: {ts}") from e

        idx = self._ts_to_index.get(ts_int)
        if idx is None:
            raise ValueError(f"Timestamp {ts_int} not found in dataset")
        return self.data[idx]

    def get_data_at_date(self, date_str: str) -> torch.Tensor:
        """Return feature vector [num_columns] at given date (YYYY-MM-DD)."""
        ts = self.get_timestamp_from_date(date_str)
        return self.get_data_at_timestamp(ts)

# Backwards-compatible alias
Dataset = SingleStockDataset
