"""Track trade history and calculate performance metrics."""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class TradeRecord:
    """Record of a single trade."""
    timestamp: str
    symbol: str
    action: str  # buy or sell
    lot: float
    entry_price: float
    sl_price: float
    tp_price: float
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    profit: Optional[float] = None
    status: str = "open"  # open, closed, cancelled
    magic: Optional[int] = None
    spread_pips: Optional[float] = None
    ai_probability: Optional[float] = None
    ticket: Optional[int] = None  # MT5 position/order ticket


@dataclass
class PerformanceMetrics:
    """Performance statistics for trading."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    total_loss: float
    net_profit: float
    average_win: float
    average_loss: float
    profit_factor: float
    largest_win: float
    largest_loss: float


class TradeTracker:
    """Track trades and calculate performance metrics."""
    
    def __init__(self, csv_path: Path = Path("logs/trades.csv")):
        """
        Initialize trade tracker.
        
        Args:
            csv_path: Path to CSV file for storing trade history
        """
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_exists()
        self.trades: List[TradeRecord] = []
    
    def _ensure_csv_exists(self) -> None:
        """Create CSV file with headers if it doesn't exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(TradeRecord.__annotations__.keys()))
                writer.writeheader()
    
    def record_trade(
        self,
        symbol: str,
        action: str,
        lot: float,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        magic: Optional[int] = None,
        spread_pips: Optional[float] = None,
        ai_probability: Optional[float] = None,
        ticket: Optional[int] = None,
    ) -> TradeRecord:
        """
        Record a new trade.
        
        Returns:
            TradeRecord instance
        """
        timestamp = datetime.now().isoformat()
        trade = TradeRecord(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            lot=lot,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            magic=magic,
            spread_pips=spread_pips,
            ai_probability=ai_probability,
            ticket=ticket,
        )
        
        self.trades.append(trade)
        self._write_trade(trade)
        return trade
    
    def _write_trade(self, trade: TradeRecord) -> None:
        """Append trade to CSV file."""
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(TradeRecord.__annotations__.keys()))
            writer.writerow(asdict(trade))
    
    def load_trades(self) -> List[TradeRecord]:
        """Load all trades from CSV file."""
        if not self.csv_path.exists():
            return []

        trades = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle legacy CSV rows without ticket header (extra value under None key)
                if None in row:
                    extra_values = row.pop(None)
                    if extra_values and not row.get("ticket"):
                        row["ticket"] = extra_values[-1]

                # Ensure all expected keys exist
                for key in TradeRecord.__annotations__.keys():
                    row.setdefault(key, "")

                # Convert string values to appropriate types
                row["lot"] = float(row["lot"])
                row["entry_price"] = float(row["entry_price"])
                row["sl_price"] = float(row["sl_price"])
                row["tp_price"] = float(row["tp_price"])
                row["exit_price"] = float(row["exit_price"]) if row["exit_price"] else None
                row["profit"] = float(row["profit"]) if row["profit"] else None
                row["magic"] = int(row["magic"]) if row["magic"] else None
                row["spread_pips"] = float(row["spread_pips"]) if row["spread_pips"] else None
                row["ai_probability"] = float(row["ai_probability"]) if row["ai_probability"] else None
                row["ticket"] = int(row["ticket"]) if row.get("ticket") and row["ticket"] else None
                trades.append(TradeRecord(**row))

        # If the CSV header is missing new fields, rewrite it with the full schema.
        expected_fields = list(TradeRecord.__annotations__.keys())
        if reader.fieldnames != expected_fields:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=expected_fields)
                writer.writeheader()
                for trade in trades:
                    writer.writerow(asdict(trade))

        return trades
    
    def update_trade_exit(
        self,
        ticket: int,
        exit_price: float,
        profit: float,
        status: str = "closed",
    ) -> None:
        """
        Update a trade with exit information.
        
        Args:
            ticket: Position ticket number
            exit_price: Exit price
            profit: Profit/loss amount
            status: Trade status (default: closed)
        """
        trades = self.load_trades()
        updated = False
        
        for trade in trades:
            if trade.ticket == ticket:
                trade.exit_price = exit_price
                trade.exit_time = datetime.now().isoformat()
                trade.profit = profit
                trade.status = status
                updated = True
                break
        
        if updated:
            # Rewrite entire CSV with updated data
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(TradeRecord.__annotations__.keys()))
                writer.writeheader()
                for trade in trades:
                    writer.writerow(asdict(trade))
    
    def get_open_trades(self) -> List[TradeRecord]:
        """Get all open trades."""
        all_trades = self.load_trades()
        return [t for t in all_trades if t.status == "open" and t.ticket is not None]
    
    def calculate_metrics(self) -> Optional[PerformanceMetrics]:
        """
        Calculate performance metrics from closed trades.
        
        Returns:
            PerformanceMetrics or None if no closed trades
        """
        all_trades = self.load_trades()
        closed_trades = [t for t in all_trades if t.status == "closed" and t.profit is not None]
        
        if not closed_trades:
            return None
        
        winning = [t for t in closed_trades if t.profit > 0]
        losing = [t for t in closed_trades if t.profit < 0]
        
        total_trades = len(closed_trades)
        winning_trades = len(winning)
        losing_trades = len(losing)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        total_profit = sum(t.profit for t in winning)
        total_loss = abs(sum(t.profit for t in losing))
        net_profit = total_profit - total_loss
        
        average_win = total_profit / winning_trades if winning_trades > 0 else 0.0
        average_loss = total_loss / losing_trades if losing_trades > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
        
        largest_win = max((t.profit for t in winning), default=0.0)
        largest_loss = min((t.profit for t in losing), default=0.0)
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_profit=total_profit,
            total_loss=total_loss,
            net_profit=net_profit,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            largest_win=largest_win,
            largest_loss=largest_loss,
        )
    
    def get_recent_trades(self, count: int = 10) -> List[TradeRecord]:
        """Get the most recent trades."""
        all_trades = self.load_trades()
        return all_trades[-count:] if len(all_trades) >= count else all_trades
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export all trades to a pandas DataFrame."""
        trades = self.load_trades()
        if not trades:
            return pd.DataFrame()
        return pd.DataFrame([asdict(t) for t in trades])
