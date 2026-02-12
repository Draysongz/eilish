# Forex Scalp Bot (MT5 + Headway Demo)

A minimal, demo-first Forex scalping bot scaffold designed for Headway MT5 accounts. It uses a simple EMA crossover strategy with basic risk controls and can run in **dry-run** mode until you’re ready to trade.

## ✅ What this includes
- MT5 client wrapper (connect, fetch rates, place orders)
- EMA crossover scalping strategy
- Basic risk controls (spread cap, max positions, SL/TP)
- Optional AI filter (XGBoost) to validate signals
- Config-driven setup (`config.yaml` + `.env`)
- Unit tests for the strategy logic

## ⚠️ Important notes
- The MetaTrader5 Python package works best on **Windows** (or a Windows VPS). On macOS, MT5 can be unstable unless you run the terminal via Wine or a remote VPS.
- Always test on **demo** first. Move to live only after stable results.

## Setup
1. Create a virtual environment (recommended).
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

> To trade live, also install the MT5 package **on a machine with MT5 terminal installed**:

```bash
python -m pip install MetaTrader5
```

## Configure
1. Copy the environment template:

```bash
cp .env.example .env
```

2. Fill in your MT5 demo credentials in `.env`.
3. Adjust strategy and risk settings in `config.yaml`.

### AI filter setup (optional)
1. Place your historical trades data in `data/trades.csv`.
	- If you don't have one yet, you can generate it from MT5 history (Windows required):

```bash
python -m src.data_builder --symbol EURUSD --timeframe M1 --bars 5000 --horizon 15
```

This will create `data/trades.csv` using your current strategy and label trades as profitable or not.
2. Enable the filter in `config.yaml`:

```yaml
ai:
	enabled: true
	model_path: models/ai_filter.json
	train_data_path: data/trades.csv
	probability_threshold: 0.6
```

The model will train automatically if `models/ai_filter.json` does not exist.

## Run (demo + dry-run)
```bash
python -m src.main
```

Set `dry_run: false` in `config.yaml` when you are ready to place demo trades.

## Tests
```bash
python -m pytest -q
```

## Price data importers
You can build a price dataset from CSV, Dukascopy-exported CSV, or OANDA.

### CSV (any broker export)
```bash
python -m src.price_importer --provider csv --input data/raw_prices.csv --output data/price_history.csv
```

### Dukascopy CSV export
```bash
python -m src.price_importer --provider dukascopy --input data/dukascopy_export.csv --output data/price_history.csv
```

### OANDA (requires API key)
```bash
export OANDA_API_KEY=your_key
python -m src.price_importer --provider oanda --symbol EUR_USD --granularity M1 --start 2024-01-01 --end 2024-02-01
```

## Data merge pipeline
Merge multiple price CSVs into a single dataset:

```bash
python -m src.data_merge data/price1.csv data/price2.csv --output data/merged_prices.csv
```

## Model training notebook
Open `notebooks/model_training.ipynb` to experiment with features and train a model interactively.

## Quick contract (inputs/outputs)
**Inputs**
- MT5 demo credentials in `.env`
- Strategy + risk params in `config.yaml`
- Optional AI training data in `data/trades.csv`

**Outputs**
- Console logs for signals and order actions
- AI filter logs (probability + decision)
- MT5 orders (only if `dry_run` is off)

## Next steps (optional)
- Add trailing stop logic
- Add multi-timeframe filters
- Store trades in a local DB for analytics

---
If you want a different strategy, multi-symbol support, or live deployment on a VPS, I can extend this.