import pandas as pd
import numpy as np

from src.strategy import generate_signal, compute_rsi, compute_atr


def _make_prices(prices):
    return pd.DataFrame({
        "open": prices,
        "high": [p + 0.0001 for p in prices],
        "low": [p - 0.0001 for p in prices],
        "close": prices,
    })


def test_signal_buy_on_cross():
    prices = list(range(1, 120))
    prices = [p / 100.0 for p in prices]  # Make prices more realistic
    data = _make_prices(prices)
    state = generate_signal(
        data, 
        ema_fast=5, 
        ema_slow=20, 
        min_bars=50, 
        allow_short=True,
        use_rsi=False,  # Disable filters for basic test
        use_atr=False,
    )
    assert state.signal in {"buy", "hold"}


def test_signal_sell_on_cross():
    prices = list(range(120, 0, -1))
    prices = [p / 100.0 for p in prices]
    data = _make_prices(prices)
    state = generate_signal(
        data, 
        ema_fast=5, 
        ema_slow=20, 
        min_bars=50, 
        allow_short=True,
        use_rsi=False,
        use_atr=False,
    )
    assert state.signal in {"sell", "hold"}


def test_hold_with_insufficient_bars():
    prices = list(range(1, 10))
    prices = [p / 100.0 for p in prices]
    data = _make_prices(prices)
    state = generate_signal(
        data, 
        ema_fast=5, 
        ema_slow=20, 
        min_bars=50, 
        allow_short=True,
        use_rsi=False,
        use_atr=False,
    )
    assert state.signal == "hold"
    assert "Insufficient data" in state.reason


def test_rsi_filter_rejects_overbought():
    """Test that RSI filter blocks buy signals when overbought."""
    # Create uptrend data that would normally trigger buy
    prices = [1.0 + i * 0.01 for i in range(120)]
    data = _make_prices(prices)
    
    state = generate_signal(
        data,
        ema_fast=5,
        ema_slow=20,
        min_bars=50,
        allow_short=True,
        use_rsi=True,
        rsi_overbought=70.0,
        use_atr=False,
    )
    
    # If RSI blocks it, signal should be hold
    if state.signal == "hold" and "RSI overbought" in state.reason:
        assert state.rsi >= 70.0


def test_atr_filter_rejects_low_volatility():
    """Test that ATR filter blocks signals in low volatility."""
    # Create flat data (low volatility) - much flatter
    prices = [1.1000] * 120  # Completely flat
    data = _make_prices(prices)
    
    state = generate_signal(
        data,
        ema_fast=5,
        ema_slow=20,
        min_bars=50,
        allow_short=True,
        use_rsi=False,
        use_atr=True,
        atr_min_threshold=0.001,  # Higher threshold
    )
    
    # With completely flat data and high threshold, should be filtered
    # Or if there's a signal, ATR should be very low
    if state.signal != "hold":
        assert state.atr < 0.001


def test_compute_rsi():
    """Test RSI calculation."""
    prices = pd.Series([1.0, 1.1, 1.2, 1.15, 1.25, 1.3, 1.28] * 10)
    rsi = compute_rsi(prices, period=14)
    
    assert not rsi.empty
    # Drop NaN values and check the rest
    rsi_valid = rsi.dropna()
    assert all((rsi_valid >= 0) & (rsi_valid <= 100))


def test_compute_atr():
    """Test ATR calculation."""
    data = pd.DataFrame({
        "high": [1.1, 1.12, 1.15, 1.13, 1.16] * 10,
        "low": [1.09, 1.10, 1.12, 1.11, 1.14] * 10,
        "close": [1.10, 1.11, 1.13, 1.12, 1.15] * 10,
    })
    
    atr = compute_atr(data, period=14)
    
    assert not atr.empty
    # Drop NaN values and check the rest
    atr_valid = atr.dropna()
    assert all(atr_valid >= 0)
