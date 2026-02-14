"""Tests for trade tracking functionality."""
from pathlib import Path
import tempfile

from src.trade_tracker import TradeTracker, TradeRecord, PerformanceMetrics


def test_record_trade_creates_csv():
    """Test that recording a trade creates CSV file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "trades.csv"
        tracker = TradeTracker(csv_path=csv_path)
        
        trade = tracker.record_trade(
            symbol="EURUSD",
            action="buy",
            lot=0.01,
            entry_price=1.1000,
            sl_price=1.0950,
            tp_price=1.1050,
            ticket=12345,
        )
        
        assert csv_path.exists()
        assert trade.symbol == "EURUSD"
        assert trade.action == "buy"
        assert trade.ticket == 12345


def test_load_trades():
    """Test loading trades from CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "trades.csv"
        tracker = TradeTracker(csv_path=csv_path)
        
        tracker.record_trade("EURUSD", "buy", 0.01, 1.1000, 1.0950, 1.1050)
        tracker.record_trade("GBPUSD", "sell", 0.02, 1.2500, 1.2550, 1.2450)
        
        trades = tracker.load_trades()
        
        assert len(trades) == 2
        assert trades[0].symbol == "EURUSD"
        assert trades[1].symbol == "GBPUSD"


def test_calculate_metrics_no_trades():
    """Test metrics calculation with no closed trades."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "trades.csv"
        tracker = TradeTracker(csv_path=csv_path)
        
        metrics = tracker.calculate_metrics()
        
        assert metrics is None


def test_calculate_metrics_with_closed_trades():
    """Test metrics calculation with closed trades."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "trades.csv"
        tracker = TradeTracker(csv_path=csv_path)
        
        # Manually create some closed trades for testing
        import csv as csv_module
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv_module.DictWriter(f, fieldnames=list(TradeRecord.__annotations__.keys()))
            writer.writeheader()
            writer.writerow({
                "timestamp": "2024-01-01T10:00:00",
                "symbol": "EURUSD",
                "action": "buy",
                "lot": 0.01,
                "entry_price": 1.1000,
                "sl_price": 1.0950,
                "tp_price": 1.1050,
                "exit_price": 1.1020,
                "exit_time": "2024-01-01T10:05:00",
                "profit": 20.0,
                "status": "closed",
                "magic": "",
                "spread_pips": "",
                "ai_probability": "",
                "ticket": "",
            })
            writer.writerow({
                "timestamp": "2024-01-01T11:00:00",
                "symbol": "EURUSD",
                "action": "buy",
                "lot": 0.01,
                "entry_price": 1.1000,
                "sl_price": 1.0950,
                "tp_price": 1.1050,
                "exit_price": 1.0980,
                "exit_time": "2024-01-01T11:05:00",
                "profit": -20.0,
                "status": "closed",
                "magic": "",
                "spread_pips": "",
                "ai_probability": "",
                "ticket": "",
            })
        
        metrics = tracker.calculate_metrics()
        
        assert metrics is not None
        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 0.5
        assert metrics.net_profit == 0.0


def test_get_recent_trades():
    """Test getting recent trades."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "trades.csv"
        tracker = TradeTracker(csv_path=csv_path)
        
        for i in range(15):
            tracker.record_trade(
                symbol=f"PAIR{i}",
                action="buy",
                lot=0.01,
                entry_price=1.1000,
                sl_price=1.0950,
                tp_price=1.1050,
            )
        
        recent = tracker.get_recent_trades(count=5)
        
        assert len(recent) == 5
        assert recent[-1].symbol == "PAIR14"
