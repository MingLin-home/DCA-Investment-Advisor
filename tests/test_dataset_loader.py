import os
import sys
import math
import pandas as pd
import numpy as np
import torch

# Ensure project root is on sys.path to import dataset_loader.py when run directly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset_loader import Dataset, SingleStockDataset


STOCK_SYMBOL = os.environ.get("TEST_STOCK_SYMBOL", "INTC")
CSV_PATH = os.path.join("outputs", "data", f"{STOCK_SYMBOL}.csv")


def _load_csv(symbol=STOCK_SYMBOL):
    assert os.path.isfile(CSV_PATH), f"CSV not found at {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)
    print("Loaded CSV:")
    print(f"- path: {CSV_PATH}")
    print(f"- shape: {df.shape}")
    print(f"- columns: {list(df.columns)}")
    print(df.head(3).to_string(index=False))
    return df


def test_init_and_len():
    df = _load_csv()
    ds = Dataset(STOCK_SYMBOL, time_window=16)
    print("Dataset initialized with default columns")
    print(f"- columns: {ds.columns}")
    print(f"- tensor shape: {tuple(ds.data.shape)}")
    assert len(ds) == len(df), "__len__ should equal the number of CSV rows"
    assert ds.data.shape[1] == 4, "Default columns should be 4"


def test_set_sampling_range_and_get_sample():
    torch.manual_seed(0)
    ds = Dataset(STOCK_SYMBOL, time_window=8)
    n = len(ds)
    start, end = 10, min(200, n)
    ds.set_sampling_range(start, end)
    print(f"Sampling range set to indices [{start}, {end}) out of {n}")
    x = ds.get_sample()
    print(f"Sample shape: {tuple(x.shape)}")
    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == ds.time_window and x.shape[1] == len(ds.columns)
    # Ensure the sampled window lies fully in the requested range by checking timestamp bounds
    ts_col = ds.columns.index("timestamp")
    ts_vals = x[:, ts_col].numpy()
    assert np.all(np.isfinite(ts_vals))
    all_ts = pd.read_csv(CSV_PATH, usecols=["timestamp"])  # full timestamps
    from_ts = all_ts.iloc[start][0]
    to_ts = all_ts.iloc[end - 1][0]
    assert ts_vals.min() >= from_ts and ts_vals.max() <= to_ts


def test_set_sample_date_range_and_get_sample():
    ds = Dataset(STOCK_SYMBOL, time_window=8)
    df = _load_csv()
    # choose a middle slice by dates
    mid_idx = len(df) // 2
    # pick a window around the middle within bounds
    a = max(0, mid_idx - 30)
    b = min(len(df) - 1, mid_idx + 30)
    from_date = str(df.iloc[a]["date"]) if "date" in df.columns else pd.to_datetime(df.iloc[a]["timestamp"], unit="s", utc=True).strftime("%Y-%m-%d")
    to_date = str(df.iloc[b]["date"]) if "date" in df.columns else pd.to_datetime(df.iloc[b]["timestamp"], unit="s", utc=True).strftime("%Y-%m-%d")
    ds.set_sample_date_range(from_date, to_date)
    print(f"Date sampling range set to [{from_date}, {to_date}]")
    x = ds.get_sample()
    print(f"Sample (date-range) shape: {tuple(x.shape)}")
    ts_col = ds.columns.index("timestamp")
    ts_vals = x[:, ts_col].numpy()
    from_ts = ds.get_timestamp_from_date(from_date)
    to_ts = ds.get_timestamp_from_date(to_date)
    print(f"- Expected ts in [{from_ts}, {to_ts}] inclusive")
    print(f"- Actual ts window: [{ts_vals.min()}, {ts_vals.max()}]")
    assert ts_vals.min() >= from_ts and ts_vals.max() <= to_ts


def test_get_timestamp_and_data_accessors():
    ds = Dataset(STOCK_SYMBOL, time_window=4)
    df = _load_csv()
    # take three sample rows: first, middle, last
    idxs = [0, len(df) // 2, len(df) - 1]
    for i in idxs:
        row = df.iloc[i]
        date_str = row["date"] if "date" in df.columns else pd.to_datetime(row["timestamp"], unit="s", utc=True).strftime("%Y-%m-%d")
        ts_expected = int(pd.to_datetime(date_str, format="%Y-%m-%d", utc=True).timestamp())
        ts = ds.get_timestamp_from_date(date_str)
        print(f"Row {i}: date={date_str}, expected_ts={ts_expected}, got_ts={ts}")
        assert ts == ts_expected

        # data at timestamp/date should match selected columns
        data_ts = ds.get_data_at_timestamp(ts)
        data_dt = ds.get_data_at_date(date_str)
        print(f"- data_at_timestamp: {data_ts.tolist()}")
        print(f"- data_at_date:      {data_dt.tolist()}")
        assert torch.allclose(data_ts, data_dt)

        # Construct expected vector in the same order as ds.columns
        expected_vals = []
        for col in ds.columns:
            # some columns may be NaN; dataset stores as float
            expected_vals.append(float(row[col]))
        expected = torch.tensor(expected_vals, dtype=torch.float32)
        print(f"- expected:           {expected.tolist()}")
        assert torch.allclose(data_ts, expected, equal_nan=True)


def test_custom_columns_order_and_values():
    df = _load_csv()
    cols = ["timestamp", "avg_price", "ROE"]
    ds = SingleStockDataset(STOCK_SYMBOL, time_window=5, columns=cols)
    print(f"Custom columns: {ds.columns}")
    assert ds.data.shape[1] == len(cols)
    # Verify a specific row matches CSV values
    i = min(42, len(df) - 1)
    ts = int(df.iloc[i]["timestamp"])
    v = ds.get_data_at_timestamp(ts)
    expected = torch.tensor([float(df.iloc[i][c]) for c in cols], dtype=torch.float32)
    print(f"Row {i} values (tensor vs expected):\n- tensor:   {v.tolist()}\n- expected: {expected.tolist()}")
    assert torch.allclose(v, expected, equal_nan=True)


def test_error_conditions():
    ds = Dataset(STOCK_SYMBOL, time_window=10)
    n = len(ds)
    # Invalid sampling range
    try:
        ds.set_sampling_range(-1, 5)
        assert False, "Expected ValueError for negative from_idx"
    except (TypeError, ValueError):
        print("Caught expected error for invalid set_sampling_range(-1, 5)")

    # Range too small for time_window
    small_from, small_to = 0, 10  # size == time_window -> should raise
    ds.set_sampling_range(small_from, small_to)
    try:
        _ = ds.get_sample()
        assert False, "Expected ValueError when time_window >= (to - from)"
    except ValueError:
        print("Caught expected ValueError for get_sample with too-small range")

    # Date range with from > to
    df = _load_csv()
    if len(df) >= 2:
        a_date = str(df.iloc[0]["date"]) if "date" in df.columns else pd.to_datetime(df.iloc[0]["timestamp"], unit="s", utc=True).strftime("%Y-%m-%d")
        b_date = str(df.iloc[-1]["date"]) if "date" in df.columns else pd.to_datetime(df.iloc[-1]["timestamp"], unit="s", utc=True).strftime("%Y-%m-%d")
        try:
            ds.set_sample_date_range(b_date, a_date)
            assert False, "Expected ValueError for from_date > to_date"
        except ValueError:
            print("Caught expected ValueError for set_sample_date_range with reversed dates")

    # Missing timestamp lookup
    # Take a real timestamp and offset by +1 to guarantee missing (timestamps are midnight UTC)
    any_ts = int(pd.read_csv(CSV_PATH, usecols=["timestamp"]).iloc[0][0])
    missing_ts = any_ts + 1
    try:
        _ = ds.get_data_at_timestamp(missing_ts)
        assert False, "Expected ValueError for missing timestamp"
    except ValueError:
        print(f"Caught expected ValueError for get_data_at_timestamp({missing_ts}) not in dataset")


if __name__ == "__main__":
    # Allow running directly for verbose diagnostics without pytest
    print("Running dataset_loader smoke tests...")
    test_init_and_len()
    test_set_sampling_range_and_get_sample()
    test_set_sample_date_range_and_get_sample()
    test_get_timestamp_and_data_accessors()
    test_custom_columns_order_and_values()
    test_error_conditions()
    print("All tests executed.")
