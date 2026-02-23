from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class PaperTrade:
    symbol: str
    direction: str  # BUY or SELL
    entry_price: float
    sl: float
    tp: float
    open_time: datetime
    close_time: Optional[datetime] = None
    result: Optional[str] = None  # TP / SL / MANUAL
    profit: Optional[float] = None


class PaperExecutor:
    def __init__(self) -> None:
        self._open_trades: List[PaperTrade] = []
        self._closed_trades: List[PaperTrade] = []

    @property
    def open_trades(self) -> List[PaperTrade]:
        return list(self._open_trades)

    def place_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
        open_time: Optional[datetime] = None,
    ) -> PaperTrade:
        trade = PaperTrade(
            symbol=symbol,
            direction=direction.upper(),
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            open_time=open_time or datetime.utcnow(),
        )
        self._open_trades.append(trade)
        print(
            f"[SHADOW] open {trade.symbol} {trade.direction} entry={trade.entry_price:.5f} sl={trade.sl:.5f} tp={trade.tp:.5f}"
        )
        return trade

    def close_trade(self, trade: PaperTrade, exit_price: float, result: str) -> PaperTrade:
        trade.close_time = datetime.utcnow()
        trade.result = result
        if trade.direction == "BUY":
            trade.profit = exit_price - trade.entry_price
        else:
            trade.profit = trade.entry_price - exit_price
        self._open_trades = [t for t in self._open_trades if t is not trade]
        self._closed_trades.append(trade)
        print(
            f"[SHADOW] close {trade.symbol} {trade.direction} result={result} exit={exit_price:.5f} profit={trade.profit:.5f}"
        )
        return trade

    def update_positions(self, symbol: str, current_price: float) -> List[PaperTrade]:
        closed: List[PaperTrade] = []
        for trade in list(self._open_trades):
            if trade.symbol != symbol:
                continue
            if trade.direction == "BUY":
                if current_price <= trade.sl:
                    closed.append(self.close_trade(trade, trade.sl, "SL"))
                elif current_price >= trade.tp:
                    closed.append(self.close_trade(trade, trade.tp, "TP"))
            else:
                if current_price >= trade.sl:
                    closed.append(self.close_trade(trade, trade.sl, "SL"))
                elif current_price <= trade.tp:
                    closed.append(self.close_trade(trade, trade.tp, "TP"))
        return closed
