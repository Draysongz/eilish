from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

from src.config import AppConfig, TradeSymbolConfig
from src.strategy import generate_signal, compute_atr
from strategy_shadow.paper_executor import PaperExecutor
from strategy_shadow.shadow_tracker import ShadowTracker


@dataclass
class ShadowState:
    last_signal: Optional[str] = None
    last_trade_signal: Optional[str] = None
    last_loss_time: Optional[datetime] = None


class ShadowDecisionEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.executor = PaperExecutor()
        self.tracker = ShadowTracker()
        self.state: Dict[str, ShadowState] = {}

        # Behavioral filters (default values)
        self.cooldown_seconds = 300
        self.atr_expansion_multiplier = 1.1

    def _get_state(self, symbol: str) -> ShadowState:
        if symbol not in self.state:
            self.state[symbol] = ShadowState()
        return self.state[symbol]

    def _pip_size(self, price: float) -> float:
        return 0.01 if price >= 100 else 0.0001

    def _atr_expansion_ok(self, rates: pd.DataFrame) -> bool:
        if rates.empty:
            return False
        atr_series = compute_atr(rates, self.config.strategy.atr_period)
        if len(atr_series) < 3:
            return False
        current_atr = float(atr_series.iloc[-1])
        previous_atr = float(atr_series.iloc[-2])
        if previous_atr <= 0:
            return False
        return current_atr >= previous_atr * self.atr_expansion_multiplier

    def _cooldown_active(self, state: ShadowState) -> bool:
        if state.last_loss_time is None:
            return False
        return datetime.utcnow() - state.last_loss_time < timedelta(seconds=self.cooldown_seconds)

    def _resolve_symbol_trade(self, symbol: str) -> TradeSymbolConfig:
        override = self.config.trade.per_symbol.get(symbol)
        if override is None:
            raise KeyError(f"Missing trade config for symbol {symbol}")
        return override

    def process(self, symbol: str, rates: pd.DataFrame) -> None:
        if rates.empty:
            return

        state = self._get_state(symbol)
        current_price = float(rates.iloc[-1]["close"])

        closed = self.executor.update_positions(symbol, current_price)
        for trade in closed:
            self.tracker.record_close(trade, current_price)
            if trade.profit is not None and trade.profit < 0:
                state.last_loss_time = datetime.utcnow()

        trade_cfg = self._resolve_symbol_trade(symbol)

        signal_state = generate_signal(
            rates,
            self.config.strategy.ema_fast,
            self.config.strategy.ema_slow,
            self.config.strategy.min_bars,
            self.config.strategy.allow_short,
            self.config.strategy.entry_delay_bars,
            self.config.strategy.use_rsi,
            self.config.strategy.rsi_period,
            self.config.strategy.rsi_overbought,
            self.config.strategy.rsi_oversold,
            self.config.strategy.use_atr,
            self.config.strategy.atr_period,
            self.config.strategy.atr_min_thresholds.get(symbol, self.config.strategy.atr_min_threshold),
        )

        if signal_state.signal == "hold":
            state.last_signal = "hold"
            state.last_trade_signal = None
            return

        if self._cooldown_active(state):
            print(f"[SHADOW] {symbol} cooldown active, skip trade")
            return

        if not self._atr_expansion_ok(rates):
            print(f"[SHADOW] {symbol} ATR expansion not met")
            return

        # One-trade-per-signal lock
        if signal_state.signal == state.last_trade_signal and state.last_signal == signal_state.signal:
            print(f"[SHADOW] {symbol} signal lock {signal_state.signal}")
            return

        state.last_signal = signal_state.signal

        open_positions = [t for t in self.executor.open_trades if t.symbol == symbol]
        if len(open_positions) >= self.config.trade.max_positions:
            print(f"[SHADOW] {symbol} max positions reached")
            return

        entry_price = current_price
        if trade_cfg.sl_mode == "atr" and signal_state.atr > 0:
            sl_distance = signal_state.atr * trade_cfg.atr_sl_multiplier
        else:
            sl_distance = trade_cfg.sl_pips * self._pip_size(entry_price)
        tp_distance = sl_distance * trade_cfg.rr_ratio if trade_cfg.rr_ratio > 0 else trade_cfg.tp_pips * self._pip_size(entry_price)

        if signal_state.signal == "buy":
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
            direction = "BUY"
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
            direction = "SELL"

        self.executor.place_trade(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
        )
        state.last_trade_signal = signal_state.signal
