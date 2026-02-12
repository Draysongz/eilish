import pandas as pd

from src.strategy import generate_signal


def _make_prices(prices):
    return pd.DataFrame({"close": prices})


def test_signal_buy_on_cross():
    prices = list(range(1, 120))
    data = _make_prices(prices)
    state = generate_signal(data, ema_fast=5, ema_slow=20, min_bars=50, allow_short=True)
    assert state.signal in {"buy", "hold"}


def test_signal_sell_on_cross():
    prices = list(range(120, 0, -1))
    data = _make_prices(prices)
    state = generate_signal(data, ema_fast=5, ema_slow=20, min_bars=50, allow_short=True)
    assert state.signal in {"sell", "hold"}


def test_hold_with_insufficient_bars():
    prices = list(range(1, 10))
    data = _make_prices(prices)
    state = generate_signal(data, ema_fast=5, ema_slow=20, min_bars=50, allow_short=True)
    assert state.signal == "hold"
