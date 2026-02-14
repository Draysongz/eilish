from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import numpy as np

Signal = Literal["buy", "sell", "hold"]


@dataclass(frozen=True)
class StrategyState:
    ema_fast: float
    ema_slow: float
    signal: Signal
    rsi: float = 0.0
    atr: float = 0.0
    reason: str = ""


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    
    avg_gain = gains.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = data["high"]
    low = data["low"]
    close = data["close"]
    
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    
    return atr


def generate_signal(
    data: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
    min_bars: int,
    allow_short: bool,
    use_rsi: bool = True,
    rsi_period: int = 14,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    use_atr: bool = True,
    atr_period: int = 14,
    atr_min_threshold: float = 0.0001,
) -> StrategyState:
    """
    Generate trading signal with multiple indicator filters.
    
    Args:
        data: OHLC price data
        ema_fast: Fast EMA period
        ema_slow: Slow EMA period
        min_bars: Minimum bars required
        allow_short: Allow short/sell signals
        use_rsi: Enable RSI filter
        rsi_period: RSI calculation period
        rsi_overbought: RSI overbought level (reject buys above this)
        rsi_oversold: RSI oversold level (reject sells below this)
        use_atr: Enable ATR volatility filter
        atr_period: ATR calculation period
        atr_min_threshold: Minimum ATR to consider (filters low volatility)
        
    Returns:
        StrategyState with signal and indicator values
    """
    if data.empty or len(data) < max(min_bars, ema_slow + 2, rsi_period + 2, atr_period + 2):
        return StrategyState(
            ema_fast=float("nan"),
            ema_slow=float("nan"),
            signal="hold",
            reason="Insufficient data"
        )

    close = data["close"]
    fast = compute_ema(close, ema_fast)
    slow = compute_ema(close, ema_slow)

    prev_fast, curr_fast = fast.iloc[-2], fast.iloc[-1]
    prev_slow, curr_slow = slow.iloc[-2], slow.iloc[-1]
    
    # Calculate indicators
    rsi_values = compute_rsi(close, rsi_period) if use_rsi else pd.Series([50.0] * len(data))
    curr_rsi = float(rsi_values.iloc[-1]) if not pd.isna(rsi_values.iloc[-1]) else 50.0
    
    atr_values = compute_atr(data, atr_period) if use_atr else pd.Series([0.0] * len(data))
    curr_atr = float(atr_values.iloc[-1]) if not pd.isna(atr_values.iloc[-1]) else 0.0

    # Check for EMA crossover
    bullish_cross = prev_fast <= prev_slow and curr_fast > curr_slow
    bearish_cross = prev_fast >= prev_slow and curr_fast < curr_slow
    
    # Apply filters
    if bullish_cross:
        # Potential BUY signal - check filters
        if use_rsi and curr_rsi >= rsi_overbought:
            return StrategyState(
                ema_fast=curr_fast,
                ema_slow=curr_slow,
                signal="hold",
                rsi=curr_rsi,
                atr=curr_atr,
                reason=f"RSI overbought ({curr_rsi:.1f} >= {rsi_overbought})"
            )
        
        if use_atr and curr_atr < atr_min_threshold:
            return StrategyState(
                ema_fast=curr_fast,
                ema_slow=curr_slow,
                signal="hold",
                rsi=curr_rsi,
                atr=curr_atr,
                reason=f"Low volatility (ATR {curr_atr:.5f} < {atr_min_threshold})"
            )
        
        return StrategyState(
            ema_fast=curr_fast,
            ema_slow=curr_slow,
            signal="buy",
            rsi=curr_rsi,
            atr=curr_atr,
            reason="EMA bullish cross + filters passed"
        )

    if allow_short and bearish_cross:
        # Potential SELL signal - check filters
        if use_rsi and curr_rsi <= rsi_oversold:
            return StrategyState(
                ema_fast=curr_fast,
                ema_slow=curr_slow,
                signal="hold",
                rsi=curr_rsi,
                atr=curr_atr,
                reason=f"RSI oversold ({curr_rsi:.1f} <= {rsi_oversold})"
            )
        
        if use_atr and curr_atr < atr_min_threshold:
            return StrategyState(
                ema_fast=curr_fast,
                ema_slow=curr_slow,
                signal="hold",
                rsi=curr_rsi,
                atr=curr_atr,
                reason=f"Low volatility (ATR {curr_atr:.5f} < {atr_min_threshold})"
            )
        
        return StrategyState(
            ema_fast=curr_fast,
            ema_slow=curr_slow,
            signal="sell",
            rsi=curr_rsi,
            atr=curr_atr,
            reason="EMA bearish cross + filters passed"
        )

    return StrategyState(
        ema_fast=curr_fast,
        ema_slow=curr_slow,
        signal="hold",
        rsi=curr_rsi,
        atr=curr_atr,
        reason="No crossover"
    )
