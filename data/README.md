# Trades dataset

Place your historical trades data in `data/trades.csv`.

If you don't have a file yet, you can generate one from MT5 history (Windows required):

```bash
python -m src.data_builder --symbol EURUSD --timeframe M1 --bars 5000 --horizon 15
```

## Expected columns
- `time` (ISO timestamp)
- `open`, `high`, `low`, `close`
- `label` (1 if trade would have been profitable, 0 otherwise)

Optional:
- `profit` (if `label` is missing, `profit > 0` will be used as the label)

The AI filter uses this file to train an XGBoost model when no saved model exists.

You can also import raw price data with `src/price_importer.py` and merge multiple files with `src/data_merge.py`.
