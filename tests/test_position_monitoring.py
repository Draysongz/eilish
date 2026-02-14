"""Test position monitoring functionality."""
import tempfile
from pathlib import Path

from src.trade_tracker import TradeTracker


def test_update_trade_exit():
    """Test updating a trade with exit information."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "trades.csv"
        tracker = TradeTracker(csv_path=csv_path)
        
        # Record a trade
        trade = tracker.record_trade(
            symbol="EURUSD",
            action="buy",
            lot=0.01,
            entry_price=1.1000,
            sl_price=1.0950,
            tp_price=1.1050,
            ticket=12345,
        )
        
        # Update with exit info
        tracker.update_trade_exit(
            ticket=12345,
            exit_price=1.1020,
            profit=20.0,
            status="closed",
        )
        
        # Load and verify
        trades = tracker.load_trades()
        assert len(trades) == 1
        assert trades[0].exit_price == 1.1020
        assert trades[0].profit == 20.0
        assert trades[0].status == "closed"


def test_get_open_trades():
    """Test getting only open trades."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "trades.csv"
        tracker = TradeTracker(csv_path=csv_path)
        
        # Record trades
        tracker.record_trade("EURUSD", "buy", 0.01, 1.1000, 1.0950, 1.1050, ticket=111)
        tracker.record_trade("GBPUSD", "sell", 0.02, 1.2500, 1.2550, 1.2450, ticket=222)
        
        # Close one trade
        tracker.update_trade_exit(ticket=111, exit_price=1.1020, profit=20.0)
        
        # Get open trades
        open_trades = tracker.get_open_trades()
        
        assert len(open_trades) == 1
        assert open_trades[0].ticket == 222
        assert open_trades[0].status == "open"
