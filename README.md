# Forex Scalp Bot (MT5 + Headway Demo)

A minimal, demo-first Forex scalping bot scaffold designed for Headway MT5 accounts. It uses a simple EMA crossover strategy with basic risk controls and can run in **dry-run** mode until you’re ready to trade.

## ✅ What this includes
- MT5 client wrapper (connect, fetch rates, place orders)
- EMA crossover scalping strategy
- Basic risk controls (spread cap, max positions, SL/TP)
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

## Run (demo + dry-run)
```bash
python -m src.main
```

Set `dry_run: false` in `config.yaml` when you are ready to place demo trades.

## Tests
```bash
python -m pytest -q
```

## Quick contract (inputs/outputs)
**Inputs**
- MT5 demo credentials in `.env`
- Strategy + risk params in `config.yaml`

**Outputs**
- Console logs for signals and order actions
- MT5 orders (only if `dry_run` is off)

## Next steps (optional)
- Add trailing stop logic
- Add multi-timeframe filters
- Store trades in a local DB for analytics

---
If you want a different strategy, multi-symbol support, or live deployment on a VPS, I can extend this.