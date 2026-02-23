from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import csv


@dataclass
class Trade:
    symbol: str
    open_time: datetime
    close_time: datetime
    profit: float


@dataclass
class PerformanceMetrics:
    total_trades: int
    wins: int
    losses: int
    winrate: float
    net_profit: float
    average_win: float
    average_loss: float
    reward_risk_ratio: float
    max_drawdown: float
    trade_frequency: float
    consecutive_losses_max: int
    profit_factor: float
    rapid_reentry_count: int


def _parse_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def load_production_trades(path: Path) -> List[Trade]:
    if not path.exists():
        return []
    trades: List[Trade] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") != "closed":
                continue
            profit_str = row.get("profit") or ""
            if not profit_str:
                continue
            profit = float(profit_str)
            symbol = row.get("symbol") or ""
            open_time = _parse_dt(row.get("timestamp") or "")
            close_time = _parse_dt(row.get("exit_time") or "") or open_time
            if open_time is None or close_time is None:
                continue
            trades.append(Trade(symbol=symbol, open_time=open_time, close_time=close_time, profit=profit))
    return trades


def load_shadow_trades(path: Path) -> List[Trade]:
    if not path.exists():
        return []
    trades: List[Trade] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            profit_str = row.get("profit") or ""
            if not profit_str:
                continue
            profit = float(profit_str)
            symbol = row.get("symbol") or ""
            open_time = _parse_dt(row.get("time_open") or "")
            close_time = _parse_dt(row.get("time_close") or "") or open_time
            if open_time is None or close_time is None:
                continue
            trades.append(Trade(symbol=symbol, open_time=open_time, close_time=close_time, profit=profit))
    return trades


def _equity_curve(trades: Iterable[Trade]) -> List[float]:
    equity = 0.0
    curve: List[float] = []
    for trade in trades:
        equity += trade.profit
        curve.append(equity)
    return curve


def _max_drawdown(equity_curve: Iterable[float]) -> float:
    peak = 0.0
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = value - peak
        if drawdown < max_dd:
            max_dd = drawdown
    return max_dd


def _trade_frequency(trades: List[Trade]) -> float:
    if len(trades) < 2:
        return 0.0
    trades_sorted = sorted(trades, key=lambda t: t.open_time)
    duration = (trades_sorted[-1].close_time - trades_sorted[0].open_time).total_seconds()
    if duration <= 0:
        return 0.0
    hours = duration / 3600
    return len(trades_sorted) / hours


def _consecutive_losses(trades: List[Trade]) -> int:
    max_streak = 0
    current = 0
    for trade in trades:
        if trade.profit < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _rapid_reentry_count(trades: List[Trade], window_seconds: int = 120) -> int:
    count = 0
    by_symbol: dict[str, List[Trade]] = {}
    for trade in sorted(trades, key=lambda t: t.open_time):
        by_symbol.setdefault(trade.symbol, []).append(trade)

    for symbol, symbol_trades in by_symbol.items():
        for idx, trade in enumerate(symbol_trades[:-1]):
            if trade.profit >= 0:
                continue
            loss_time = trade.close_time
            next_trade = symbol_trades[idx + 1]
            delta = (next_trade.open_time - loss_time).total_seconds()
            if 0 <= delta <= window_seconds:
                count += 1
    return count


def calculate_metrics(trades: List[Trade]) -> PerformanceMetrics:
    trades_sorted = sorted(trades, key=lambda t: t.close_time)
    total = len(trades_sorted)
    wins = len([t for t in trades_sorted if t.profit > 0])
    losses = len([t for t in trades_sorted if t.profit < 0])
    winrate = wins / total if total > 0 else 0.0
    net_profit = sum(t.profit for t in trades_sorted)
    win_profits = [t.profit for t in trades_sorted if t.profit > 0]
    loss_profits = [t.profit for t in trades_sorted if t.profit < 0]
    average_win = sum(win_profits) / len(win_profits) if win_profits else 0.0
    average_loss = sum(loss_profits) / len(loss_profits) if loss_profits else 0.0
    reward_risk_ratio = average_win / abs(average_loss) if average_loss < 0 else float("inf")
    equity = _equity_curve(trades_sorted)
    max_drawdown = _max_drawdown(equity)
    trade_frequency = _trade_frequency(trades_sorted)
    consecutive_losses_max = _consecutive_losses(trades_sorted)
    gross_win = sum(win_profits)
    gross_loss = abs(sum(loss_profits))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    rapid_reentry_count = _rapid_reentry_count(trades_sorted)

    return PerformanceMetrics(
        total_trades=total,
        wins=wins,
        losses=losses,
        winrate=winrate,
        net_profit=net_profit,
        average_win=average_win,
        average_loss=average_loss,
        reward_risk_ratio=reward_risk_ratio,
        max_drawdown=max_drawdown,
        trade_frequency=trade_frequency,
        consecutive_losses_max=consecutive_losses_max,
        profit_factor=profit_factor,
        rapid_reentry_count=rapid_reentry_count,
    )
