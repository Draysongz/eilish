import pandas as pd

from src.data_builder import label_trade_outcome


def _rates(prices):
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 0.0005 for p in prices],
            "low": [p - 0.0005 for p in prices],
            "close": prices,
        }
    )


def test_label_trade_outcome_hits_tp_first():
    prices = [1.0, 1.002, 1.004, 1.006]
    rates = _rates(prices)
    label = label_trade_outcome(
        rates=rates,
        index=0,
        direction="buy",
        tp_pips=5,
        sl_pips=5,
        pip_size=0.0001,
        horizon=3,
    )
    assert label == 1


def test_label_trade_outcome_hits_sl_first():
    prices = [1.0, 0.998, 0.996, 0.994]
    rates = _rates(prices)
    label = label_trade_outcome(
        rates=rates,
        index=0,
        direction="buy",
        tp_pips=5,
        sl_pips=5,
        pip_size=0.0001,
        horizon=3,
    )
    assert label == 0
