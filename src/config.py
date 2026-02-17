from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class MT5Config:
    path_env: str
    login_env: str
    password_env: str
    server_env: str


@dataclass(frozen=True)
class TradeConfig:
    symbols: List[str]
    timeframe: str
    max_positions: int
    magic: int
    dry_run: bool
    polling_seconds: int
    per_symbol: Dict[str, "TradeSymbolConfig"]


@dataclass(frozen=True)
class TradeSymbolConfig:
    lot: float
    max_spread_pips: float
    sl_pips: float
    tp_pips: float


@dataclass(frozen=True)
class StrategyConfig:
    ema_fast: int
    ema_slow: int
    min_bars: int
    allow_short: bool
    use_rsi: bool
    rsi_period: int
    rsi_overbought: float
    rsi_oversold: float
    use_atr: bool
    atr_period: int
    atr_min_threshold: float
    atr_min_thresholds: Dict[str, float]


@dataclass(frozen=True)
class ProfitTakeFilterConfig:
    enabled: bool
    trigger_usd: float
    risk_threshold: float


@dataclass(frozen=True)
class AppConfig:
    mt5: MT5Config
    trade: TradeConfig
    strategy: StrategyConfig
    profit_take: ProfitTakeFilterConfig
    ai: "AIConfig"


@dataclass(frozen=True)
class AIConfig:
    enabled: bool
    model_path: str
    train_data_path: str
    probability_threshold: float


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _required(data: Dict[str, Any], key: str) -> Any:
    if key not in data:
        raise KeyError(f"Missing required config key: {key}")
    return data[key]


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    load_dotenv()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    mt5_raw = _required(raw, "mt5")
    trade_raw = _required(raw, "trade")
    strategy_raw = _required(raw, "strategy")
    profit_raw = raw.get("profit_take", {})
    ai_raw = raw.get("ai", {})

    mt5 = MT5Config(
        path_env=_required(mt5_raw, "path_env"),
        login_env=_required(mt5_raw, "login_env"),
        password_env=_required(mt5_raw, "password_env"),
        server_env=_required(mt5_raw, "server_env"),
    )

    per_symbol_raw = _required(trade_raw, "per_symbol") or {}
    per_symbol: Dict[str, TradeSymbolConfig] = {}
    for symbol, overrides in per_symbol_raw.items():
        if overrides is None:
            overrides = {}
        per_symbol[str(symbol)] = TradeSymbolConfig(
            lot=float(_required(overrides, "lot")),
            max_spread_pips=float(_required(overrides, "max_spread_pips")),
            sl_pips=float(_required(overrides, "sl_pips")),
            tp_pips=float(_required(overrides, "tp_pips")),
        )

    trade = TradeConfig(
        symbols=list(_required(trade_raw, "symbols")),
        timeframe=_required(trade_raw, "timeframe"),
        max_positions=int(_required(trade_raw, "max_positions")),
        magic=int(_required(trade_raw, "magic")),
        dry_run=bool(_required(trade_raw, "dry_run")),
        polling_seconds=int(_required(trade_raw, "polling_seconds")),
        per_symbol=per_symbol,
    )

    strategy = StrategyConfig(
        ema_fast=int(_required(strategy_raw, "ema_fast")),
        ema_slow=int(_required(strategy_raw, "ema_slow")),
        min_bars=int(_required(strategy_raw, "min_bars")),
        allow_short=bool(_required(strategy_raw, "allow_short")),
        use_rsi=bool(strategy_raw.get("use_rsi", True)),
        rsi_period=int(strategy_raw.get("rsi_period", 14)),
        rsi_overbought=float(strategy_raw.get("rsi_overbought", 70.0)),
        rsi_oversold=float(strategy_raw.get("rsi_oversold", 30.0)),
        use_atr=bool(strategy_raw.get("use_atr", True)),
        atr_period=int(strategy_raw.get("atr_period", 14)),
        atr_min_threshold=float(strategy_raw.get("atr_min_threshold", 0.00005)),
        atr_min_thresholds={
            str(k): float(v)
            for k, v in (strategy_raw.get("atr_min_thresholds", {}) or {}).items()
        },
    )

    profit_take = ProfitTakeFilterConfig(
        enabled=bool(profit_raw.get("enabled", False)),
        trigger_usd=float(profit_raw.get("trigger_usd", 1.0)),
        risk_threshold=float(profit_raw.get("risk_threshold", 0.6)),
    )

    ai = AIConfig(
        enabled=bool(ai_raw.get("enabled", False)),
        model_path=str(ai_raw.get("model_path", "models/ai_filter.json")),
        train_data_path=str(ai_raw.get("train_data_path", "data/trades.csv")),
        probability_threshold=float(ai_raw.get("probability_threshold", 0.6)),
    )

    return AppConfig(mt5=mt5, trade=trade, strategy=strategy, profit_take=profit_take, ai=ai)
