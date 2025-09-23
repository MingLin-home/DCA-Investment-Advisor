# DCA-Investment-Advisor
An auto-regressive transformer model for Dollar-Cost Averaging (DCA) investment.


## Installation

```
conda activate ./venv
pip install -r requirements.txt
```

## Configuration

`config.yaml` stores configuration variables.
- "stock_symbols": a list of strings of the stock symbols we want to track.

## Data Download

Download historical daily data for symbols in `config.yaml` and write one CSV per symbol:

```
python download_data.py \
  --config config.yaml \
  --output outputs/data
```

Flags are optional; defaults are `--config config.yaml` and `--output outputs`.
The script saves data to `<output>/data/{SYMBOL}.csv` and raw frames to `<output>/raw_data/{SYMBOL}.raw.csv`.

Example:

```
python download_data.py --output ./outputs
```

Creates:
- `./outputs/data/QQQ.csv`
- `./outputs/raw_data/QQQ.raw.csv`

Notes:
- `stock_start_date` and `stock_end_date` are in `YYYY-MM-DD` format; `stock_end_date` may be `today`.
- CSV columns: `stock_symbol`, `date`, `avg_price`, `timestamp`, `ESP`, `FCF`, `PBR`, `ROE`.
- `avg_price` is computed as (High + Low + Close) / 3 for each day.
- Fundamental fields are latest available values repeated across days (may be missing for some symbols).

### Using Alpha Vantage

An alternative script uses the Alpha Vantage API instead of yfinance. Set your API key in the environment variable `alpha_vantage_api_key` and run:

```
export alpha_vantage_api_key=YOUR_KEY
python download_data_alpha_vantage.py --config config.yaml --output outputs
```

It writes CSVs with the same schema as above to `outputs/raw_data/{SYMBOL}.csv` and does not save a separate `.raw.csv` file. Respect free-tier rate limits when downloading many symbols.
