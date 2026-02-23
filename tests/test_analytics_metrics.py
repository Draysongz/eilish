from datetime import datetime, timedelta

from analytics.metrics import Trade, calculate_metrics


def _trade(symbol: str, profit: float, start: datetime, minutes: int = 1):
    return Trade(
        symbol=symbol,
        open_time=start,
        close_time=start + timedelta(minutes=minutes),
        profit=profit,
    )


def test_metrics_basic_counts():
    start = datetime(2024, 1, 1, 0, 0, 0)
    trades = [
        _trade("EURUSD", 1.0, start),
        _trade("EURUSD", -0.5, start + timedelta(minutes=2)),
        _trade("EURUSD", 2.0, start + timedelta(minutes=4)),
    ]
    metrics = calculate_metrics(trades)
    assert metrics.total_trades == 3
    assert metrics.wins == 2
    assert metrics.losses == 1
    assert metrics.rapid_reentry_count == 1
    assert metrics.consecutive_losses_max == 1
