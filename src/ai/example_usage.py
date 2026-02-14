from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import pandas as pd

from src.ai.labels import label_signals
from src.ai.train import summary_table, train_walk_forward
from src.ai_filter import compute_feature_frame
from src.strategy import generate_signal


def build_signal_candidates(rates: pd.DataFrame, min_bars: int) -> pd.DataFrame:
    logger = logging.getLogger("ai_phase1")
    signals = []
    for idx in range(min_bars, len(rates)):
        if (idx - min_bars) % 5000 == 0:
            logger.info("Scanning bars: %d/%d", idx, len(rates))
        window = rates.iloc[: idx + 1]
        state = generate_signal(
            window,
            ema_fast=6,
            ema_slow=18,
            min_bars=min_bars,
            allow_short=True,
            use_rsi=True,
            rsi_period=14,
            rsi_overbought=68.0,
            rsi_oversold=32.0,
            use_atr=True,
            atr_period=14,
            atr_min_threshold=0.2,
        )
        if state.signal in ("buy", "sell"):
            signals.append({"index": idx, "direction": state.signal})
    return pd.DataFrame(signals)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("ai_phase1")

    price_path = Path("data/dukascopy_xauusd.csv")
    if not price_path.exists():
        raise FileNotFoundError("Expected data/dukascopy_xauusd.csv")

    logger.info("Loading price data: %s", price_path)
    rates = pd.read_csv(price_path)
    rates = rates.sort_values("time").reset_index(drop=True)
    logger.info("Loaded %d rows", len(rates))

    logger.info("Building strategy signal candidates")
    signal_candidates = build_signal_candidates(rates, min_bars=80)
    if signal_candidates.empty:
        raise RuntimeError("No signals found in the price history.")
    logger.info("Signals found: %d", len(signal_candidates))

    logger.info("Labeling signals with ATR-scaled hold window")
    labels = label_signals(
        rates,
        signal_indices=signal_candidates["index"].tolist(),
        directions=signal_candidates["direction"].tolist(),
        tp_pips=250,
        sl_pips=150,
        pip_size=0.01,
        base_bars=15,
    )
    logger.info("Labels produced: %d", len(labels))

    logger.info("Building feature frame")
    features = compute_feature_frame(rates).reset_index(drop=True)
    labeled = labels.merge(
        signal_candidates,
        left_on="index",
        right_on="index",
        how="left",
    )
    labeled = labeled.join(features, on="index")
    labeled = labeled.dropna(subset=["ema_distance", "atr", "rsi"])
    logger.info("Labeled samples after feature join: %d", len(labeled))

    thresholds = np.round(np.arange(0.5, 0.81, 0.02), 2)
    logger.info("Running walk-forward validation")
    results = train_walk_forward(
        labeled,
        feature_cols=[
            "ema_distance",
            "atr",
            "rsi",
            "body",
            "volatility_5",
            "hour",
            "session_asia",
            "session_europe",
            "session_us",
        ],
        time_col="time",
        label_col="label",
        weight_col="weight",
        tp_pips=250,
        sl_pips=150,
        thresholds=thresholds,
    )

    print(summary_table(results))


if __name__ == "__main__":
    main()
