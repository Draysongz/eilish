from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Signal = Literal["buy", "sell", "hold"]


@dataclass(frozen=True)
class StrategyState:
    ema_fast: float
    ema_slow: float
    signal: Signal


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def generate_signal(
    data: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
    min_bars: int,
    allow_short: bool,
) -> StrategyState:
    if data.empty or len(data) < max(min_bars, ema_slow + 2):
        return StrategyState(ema_fast=float("nan"), ema_slow=float("nan"), signal="hold")

    close = data["close"]
    fast = compute_ema(close, ema_fast)
    slow = compute_ema(close, ema_slow)

    prev_fast, curr_fast = fast.iloc[-2], fast.iloc[-1]
    prev_slow, curr_slow = slow.iloc[-2], slow.iloc[-1]

    if prev_fast <= prev_slow and curr_fast > curr_slow:
        return StrategyState(ema_fast=curr_fast, ema_slow=curr_slow, signal="buy")

    if allow_short and prev_fast >= prev_slow and curr_fast < curr_slow:
        return StrategyState(ema_fast=curr_fast, ema_slow=curr_slow, signal="sell")

    return StrategyState(ema_fast=curr_fast, ema_slow=curr_slow, signal="hold")
