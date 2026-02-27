from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover - handled gracefully when MT5 not installed
    mt5 = None


TIMEFRAMES: Dict[str, int] = {
    "M1": getattr(mt5, "TIMEFRAME_M1", 1),
    "M5": getattr(mt5, "TIMEFRAME_M5", 5),
    "M15": getattr(mt5, "TIMEFRAME_M15", 15),
    "M30": getattr(mt5, "TIMEFRAME_M30", 30),
    "H1": getattr(mt5, "TIMEFRAME_H1", 16385),
}


@dataclass
class MT5Credentials:
    login: int
    password: str
    server: str
    path: Optional[str] = None


class MT5Client:
    def __init__(self, credentials: MT5Credentials) -> None:
        self.credentials = credentials

    def _require_mt5(self):
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package is not installed or unavailable.")

    def initialize(self) -> None:
        self._require_mt5()
        path = self.credentials.path
        if path:
            mt5.initialize(path)
        else:
            mt5.initialize()
        authorized = mt5.login(self.credentials.login, self.credentials.password, self.credentials.server)
        if not authorized:
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

    def shutdown(self) -> None:
        if mt5 is None:
            return
        mt5.shutdown()

    def symbol_info(self, symbol: str):
        self._require_mt5()
        return mt5.symbol_info(symbol)

    def symbol_info_tick(self, symbol: str):
        self._require_mt5()
        return mt5.symbol_info_tick(symbol)

    @staticmethod
    def _pip_size(info) -> float:
        point = info.point or 0.00001
        return point * 10 if info.digits in (3, 5) else point

    def get_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        self._require_mt5()
        tf = TIMEFRAMES.get(timeframe)
        if tf is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None:
            raise RuntimeError(f"Failed to fetch rates for {symbol}: {mt5.last_error()}")
        frame = pd.DataFrame(rates)
        frame["time"] = pd.to_datetime(frame["time"], unit="s")
        return frame

    def get_spread_pips(self, symbol: str) -> float:
        self._require_mt5()
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if tick is None or info is None:
            raise RuntimeError(f"Failed to fetch symbol info for {symbol}")
        pip_size = self._pip_size(info)
        spread = (tick.ask - tick.bid) / pip_size
        return float(spread)

    def get_open_positions(self, symbol: Optional[str] = None, magic: Optional[int] = None):
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package is not installed or unavailable.")
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if positions is None:
            return []
        if magic is None:
            return list(positions)
        return [pos for pos in positions if getattr(pos, "magic", None) == magic]

    def get_position_by_ticket(self, ticket: int):
        """Get an open position by ticket number."""
        self._require_mt5()
        positions = mt5.positions_get(ticket=ticket)
        if positions:
            return positions[0]
        return None
    
    def get_deals_history(self, from_date=None, to_date=None):
        """Get deals history from MT5."""
        self._require_mt5()
        if from_date is None:
            from datetime import datetime, timedelta
            from_date = datetime.now() - timedelta(days=1)
        if to_date is None:
            from datetime import datetime
            to_date = datetime.now()
        
        deals = mt5.history_deals_get(from_date, to_date)
        return list(deals) if deals else []
    
    def get_position_by_ticket(self, ticket: int):
        """Get position by ticket number."""
        self._require_mt5()
        positions = mt5.positions_get(ticket=ticket)
        if positions and len(positions) > 0:
            return positions[0]
        return None

    def place_market_order(
        self,
        symbol: str,
        action: str,
        lot: float,
        sl_price: float,
        tp_price: float,
        magic: int,
    ) -> Optional[int]:
        """Place market order and return the position ticket."""
        self._require_mt5()
        order_type = mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).ask if action == "buy" else mt5.symbol_info_tick(symbol).bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "magic": magic,
            "comment": "ema-scalp",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Order failed: {result}")
        # Return the order/position ticket for tracking
        return result.order if hasattr(result, 'order') else None

    def close_position(self, ticket: int) -> None:
        """Close an open position by ticket at market."""
        self._require_mt5()
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            raise RuntimeError(f"Position not found for ticket {ticket}")
        position = positions[0]

        symbol = position.symbol
        volume = position.volume
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Tick info missing for {symbol}")
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "magic": getattr(position, "magic", None) or 0,
            "comment": "profit-filter",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Close failed: {result}")

    def modify_position(self, ticket: int, sl_price: float, tp_price: float) -> None:
        """Modify SL/TP of an open position."""
        self._require_mt5()
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            raise RuntimeError(f"Position not found for ticket {ticket}")
        position = positions[0]

        symbol = position.symbol
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": sl_price,
            "tp": tp_price,
            "magic": getattr(position, "magic", None) or 0,
            "comment": "modify-sltp",
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Modify failed: {result}")


def credentials_from_env(login_env: str, password_env: str, server_env: str, path_env: str | None = None) -> MT5Credentials:
    login = int(os.getenv(login_env, "0"))
    password = os.getenv(password_env, "")
    server = os.getenv(server_env, "")
    path = os.getenv(path_env) if path_env else None
    if not login or not password or not server:
        raise RuntimeError("Missing MT5 credentials in environment variables.")
    return MT5Credentials(login=login, password=password, server=server, path=path)
