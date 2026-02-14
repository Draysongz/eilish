from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class LabelResult:
    index: int
    direction: str
    label: int
    weight: float
    max_hold_bars: int


def compute_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def _max_hold_bars(
    atr: float,
    atr_mean: float,
    base_bars: int,
    min_bars: int = 5,
    max_bars: int = 300,
) -> int:
    if atr_mean <= 0 or pd.isna(atr_mean) or pd.isna(atr):
        return base_bars
    scaled = int(round(base_bars * (atr / atr_mean)))
    return max(min_bars, min(max_bars, scaled))


def _label_single(
    rates: pd.DataFrame,
    index: int,
    direction: str,
    tp_pips: float,
    sl_pips: float,
    pip_size: float,
    max_hold_bars: int,
) -> tuple[int, float]:
    entry_price = float(rates.iloc[index]["close"])
    future = rates.iloc[index + 1 : index + 1 + max_hold_bars]

    for _, row in future.iterrows():
        high = float(row["high"])
        low = float(row["low"])

        if direction == "buy":
            hit_tp = high - entry_price >= tp_pips * pip_size
            hit_sl = entry_price - low >= sl_pips * pip_size
        else:
            hit_tp = entry_price - low >= tp_pips * pip_size
            hit_sl = high - entry_price >= sl_pips * pip_size

        if hit_tp and hit_sl:
            return 0, 1.0
        if hit_tp:
            return 1, 1.0
        if hit_sl:
            return 0, 1.0

    return 0, 0.5


def label_signals(
    rates: pd.DataFrame,
    signal_indices: Iterable[int],
    directions: Iterable[str],
    tp_pips: float,
    sl_pips: float,
    pip_size: float,
    base_bars: int = 15,
    atr_period: int = 14,
    atr_mean_window: int = 200,
) -> pd.DataFrame:
    rates = rates.reset_index(drop=True)
    atr = compute_atr(rates, period=atr_period)
    atr_mean = atr.rolling(atr_mean_window, min_periods=atr_period).mean()

    results = []
    for idx, direction in zip(signal_indices, directions):
        if idx + 2 >= len(rates):
            continue
        max_hold = _max_hold_bars(
            float(atr.iloc[idx]),
            float(atr_mean.iloc[idx]),
            base_bars=base_bars,
        )
        label, weight = _label_single(
            rates,
            idx,
            direction,
            tp_pips,
            sl_pips,
            pip_size,
            max_hold,
        )
        results.append(
            LabelResult(
                index=idx,
                direction=direction,
                label=label,
                weight=weight,
                max_hold_bars=max_hold,
            )
        )

    return pd.DataFrame([r.__dict__ for r in results])
