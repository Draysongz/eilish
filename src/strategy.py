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


# ========== ENTRY REFINEMENT FILTERS ==========
# These filters help avoid late entries and reduce stop loss sizes

def check_expansion_candle_filter(
    data: pd.DataFrame,
    lookback: int = 10,
    multiplier: float = 1.5,
) -> tuple[bool, str]:
    """
    A) Expansion Candle Filter
    Skip trade if current candle body is too large compared to recent candles.
    
    Args:
        data: OHLC price data
        lookback: Number of candles to compute average body size
        multiplier: Current body must be < multiplier × average body
        
    Returns:
        (passed, reason) - True if filter passed, False if trade should be rejected
    """
    if len(data) < lookback + 1:
        return False, "Insufficient data for expansion filter"
    
    # Calculate candle body sizes
    bodies = (data["close"] - data["open"]).abs()
    
    # Average body size of last N candles (excluding current)
    avg_body = bodies.iloc[-(lookback + 1):-1].mean()
    
    # Current candle body
    current_body = bodies.iloc[-1]
    
    if avg_body == 0:
        return False, "Zero average body size"
    
    # Reject if current body is too large
    if current_body > multiplier * avg_body:
        return False, f"Expansion candle (body={current_body:.5f} > {multiplier}×avg={avg_body:.5f})"
    
    return True, "Expansion candle filter passed"


def check_distance_from_slow_ema_filter(
    data: pd.DataFrame,
    slow_ema: pd.Series,
    atr: float,
    multiplier: float = 1.2,
) -> tuple[bool, str]:
    """
    B) Distance From Slow EMA Filter
    Skip trade if price is too extended from slow EMA.
    
    Args:
        data: OHLC price data
        slow_ema: Slow EMA series
        atr: Current ATR value
        multiplier: Distance must be < multiplier × ATR
        
    Returns:
        (passed, reason) - True if filter passed, False if trade should be rejected
    """
    if len(data) == 0 or len(slow_ema) == 0:
        return False, "Insufficient data for distance filter"
    
    current_close = float(data["close"].iloc[-1])
    current_slow_ema = float(slow_ema.iloc[-1])
    
    distance = abs(current_close - current_slow_ema)
    threshold = multiplier * atr
    
    if distance > threshold:
        return False, f"Too far from EMA (dist={distance:.5f} > {multiplier}×ATR={threshold:.5f})"
    
    return True, "Distance from EMA filter passed"


def check_atr_spike_ceiling_filter(
    atr_series: pd.Series,
    lookback: int = 20,
    multiplier: float = 1.5,
) -> tuple[bool, str]:
    """
    C) ATR Spike Ceiling Filter
    Skip trade if volatility just spiked (avoid entering on panic moves).
    
    Args:
        atr_series: ATR values series
        lookback: Number of bars to compute average ATR
        multiplier: Current ATR must be < multiplier × average ATR
        
    Returns:
        (passed, reason) - True if filter passed, False if trade should be rejected
    """
    if len(atr_series) < lookback + 1:
        return False, "Insufficient data for ATR spike filter"
    
    # Average ATR of last N bars (excluding current)
    avg_atr = atr_series.iloc[-(lookback + 1):-1].mean()
    
    # Current ATR
    current_atr = float(atr_series.iloc[-1])
    
    if avg_atr == 0:
        return False, "Zero average ATR"
    
    # Reject if current ATR is too high (volatility spike)
    if current_atr > multiplier * avg_atr:
        return False, f"ATR spike (curr={current_atr:.5f} > {multiplier}×avg={avg_atr:.5f})"
    
    return True, "ATR spike filter passed"


def check_break_structure_confirmation(
    data: pd.DataFrame,
    signal: Signal,
    lookback: int = 5,
) -> tuple[bool, str]:
    """
    D) Break Structure Confirmation
    Only allow BUY if price breaks above recent highs.
    Only allow SELL if price breaks below recent lows.
    
    Args:
        data: OHLC price data
        signal: Trade signal ("buy" or "sell")
        lookback: Number of candles to find high/low
        
    Returns:
        (passed, reason) - True if filter passed, False if trade should be rejected
    """
    if signal == "hold":
        return True, "No signal to filter"
    
    if len(data) < lookback + 1:
        return False, "Insufficient data for structure filter"
    
    current_close = float(data["close"].iloc[-1])
    
    if signal == "buy":
        # For BUY: current close must break above highest high of last N candles
        highest_high = data["high"].iloc[-(lookback + 1):-1].max()
        if current_close <= highest_high:
            return False, f"BUY: No structure break (close={current_close:.5f} <= high={highest_high:.5f})"
        return True, "BUY structure break confirmed"
    
    elif signal == "sell":
        # For SELL: current close must break below lowest low of last N candles
        lowest_low = data["low"].iloc[-(lookback + 1):-1].min()
        if current_close >= lowest_low:
            return False, f"SELL: No structure break (close={current_close:.5f} >= low={lowest_low:.5f})"
        return True, "SELL structure break confirmed"
    
    return True, "No applicable signal"


# ========== MAIN SIGNAL GENERATION ==========

def generate_signal(
    data: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
    min_bars: int,
    allow_short: bool,
    entry_delay_bars: int = 0,
    use_rsi: bool = True,
    rsi_period: int = 14,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    use_atr: bool = True,
    atr_period: int = 14,
    atr_min_threshold: float = 0.0001,
    # New entry refinement filters
    use_expansion_candle_filter: bool = True,
    expansion_lookback: int = 10,
    expansion_multiplier: float = 1.5,
    use_distance_from_ema_filter: bool = True,
    distance_from_ema_multiplier: float = 1.2,
    use_atr_spike_filter: bool = True,
    atr_spike_lookback: int = 20,
    atr_spike_multiplier: float = 1.5,
    use_break_structure_filter: bool = True,
    break_structure_lookback: int = 5,
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
    required_bars = max(min_bars, ema_slow + entry_delay_bars + 2, rsi_period + 2, atr_period + 2)
    if data.empty or len(data) < required_bars:
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
    if entry_delay_bars > 0:
        idx = -1 - entry_delay_bars
        prev_idx = idx - 1
        if abs(prev_idx) > len(fast) or abs(idx) > len(fast):
            return StrategyState(
                ema_fast=curr_fast,
                ema_slow=curr_slow,
                signal="hold",
                rsi=curr_rsi,
                atr=curr_atr,
                reason="Insufficient data for delay"
            )
        bullish_cross = fast.iloc[prev_idx] <= slow.iloc[prev_idx] and fast.iloc[idx] > slow.iloc[idx] and curr_fast > curr_slow
        bearish_cross = fast.iloc[prev_idx] >= slow.iloc[prev_idx] and fast.iloc[idx] < slow.iloc[idx] and curr_fast < curr_slow
    else:
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
        
        # Apply new entry refinement filters
        if use_expansion_candle_filter:
            passed, reason = check_expansion_candle_filter(data, expansion_lookback, expansion_multiplier)
            if not passed:
                return StrategyState(
                    ema_fast=curr_fast,
                    ema_slow=curr_slow,
                    signal="hold",
                    rsi=curr_rsi,
                    atr=curr_atr,
                    reason=reason
                )
        
        if use_distance_from_ema_filter:
            passed, reason = check_distance_from_slow_ema_filter(data, slow, curr_atr, distance_from_ema_multiplier)
            if not passed:
                return StrategyState(
                    ema_fast=curr_fast,
                    ema_slow=curr_slow,
                    signal="hold",
                    rsi=curr_rsi,
                    atr=curr_atr,
                    reason=reason
                )
        
        if use_atr_spike_filter:
            passed, reason = check_atr_spike_ceiling_filter(atr_values, atr_spike_lookback, atr_spike_multiplier)
            if not passed:
                return StrategyState(
                    ema_fast=curr_fast,
                    ema_slow=curr_slow,
                    signal="hold",
                    rsi=curr_rsi,
                    atr=curr_atr,
                    reason=reason
                )
        
        if use_break_structure_filter:
            passed, reason = check_break_structure_confirmation(data, "buy", break_structure_lookback)
            if not passed:
                return StrategyState(
                    ema_fast=curr_fast,
                    ema_slow=curr_slow,
                    signal="hold",
                    rsi=curr_rsi,
                    atr=curr_atr,
                    reason=reason
                )
        
        return StrategyState(
            ema_fast=curr_fast,
            ema_slow=curr_slow,
            signal="buy",
            rsi=curr_rsi,
            atr=curr_atr,
            reason="EMA bullish cross + all filters passed"
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
        
        # Apply new entry refinement filters
        if use_expansion_candle_filter:
            passed, reason = check_expansion_candle_filter(data, expansion_lookback, expansion_multiplier)
            if not passed:
                return StrategyState(
                    ema_fast=curr_fast,
                    ema_slow=curr_slow,
                    signal="hold",
                    rsi=curr_rsi,
                    atr=curr_atr,
                    reason=reason
                )
        
        if use_distance_from_ema_filter:
            passed, reason = check_distance_from_slow_ema_filter(data, slow, curr_atr, distance_from_ema_multiplier)
            if not passed:
                return StrategyState(
                    ema_fast=curr_fast,
                    ema_slow=curr_slow,
                    signal="hold",
                    rsi=curr_rsi,
                    atr=curr_atr,
                    reason=reason
                )
        
        if use_atr_spike_filter:
            passed, reason = check_atr_spike_ceiling_filter(atr_values, atr_spike_lookback, atr_spike_multiplier)
            if not passed:
                return StrategyState(
                    ema_fast=curr_fast,
                    ema_slow=curr_slow,
                    signal="hold",
                    rsi=curr_rsi,
                    atr=curr_atr,
                    reason=reason
                )
        
        if use_break_structure_filter:
            passed, reason = check_break_structure_confirmation(data, "sell", break_structure_lookback)
            if not passed:
                return StrategyState(
                    ema_fast=curr_fast,
                    ema_slow=curr_slow,
                    signal="hold",
                    rsi=curr_rsi,
                    atr=curr_atr,
                    reason=reason
                )
        
        return StrategyState(
            ema_fast=curr_fast,
            ema_slow=curr_slow,
            signal="sell",
            rsi=curr_rsi,
            atr=curr_atr,
            reason="EMA bearish cross + all filters passed"
        )

    return StrategyState(
        ema_fast=curr_fast,
        ema_slow=curr_slow,
        signal="hold",
        rsi=curr_rsi,
        atr=curr_atr,
        reason="No crossover"
    )
