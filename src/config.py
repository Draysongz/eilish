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
    lot: float
    max_spread_pips: float
    sl_pips: float
    tp_pips: float
    max_positions: int
    magic: int
    dry_run: bool
    polling_seconds: int


@dataclass(frozen=True)
class StrategyConfig:
    ema_fast: int
    ema_slow: int
    min_bars: int
    allow_short: bool


@dataclass(frozen=True)
class AppConfig:
    mt5: MT5Config
    trade: TradeConfig
    strategy: StrategyConfig
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
    ai_raw = raw.get("ai", {})

    mt5 = MT5Config(
        path_env=_required(mt5_raw, "path_env"),
        login_env=_required(mt5_raw, "login_env"),
        password_env=_required(mt5_raw, "password_env"),
        server_env=_required(mt5_raw, "server_env"),
    )

    trade = TradeConfig(
        symbols=list(_required(trade_raw, "symbols")),
        timeframe=_required(trade_raw, "timeframe"),
        lot=float(_required(trade_raw, "lot")),
        max_spread_pips=float(_required(trade_raw, "max_spread_pips")),
        sl_pips=float(_required(trade_raw, "sl_pips")),
        tp_pips=float(_required(trade_raw, "tp_pips")),
        max_positions=int(_required(trade_raw, "max_positions")),
        magic=int(_required(trade_raw, "magic")),
        dry_run=bool(_required(trade_raw, "dry_run")),
        polling_seconds=int(_required(trade_raw, "polling_seconds")),
    )

    strategy = StrategyConfig(
        ema_fast=int(_required(strategy_raw, "ema_fast")),
        ema_slow=int(_required(strategy_raw, "ema_slow")),
        min_bars=int(_required(strategy_raw, "min_bars")),
        allow_short=bool(_required(strategy_raw, "allow_short")),
    )

    ai = AIConfig(
        enabled=bool(ai_raw.get("enabled", False)),
        model_path=str(ai_raw.get("model_path", "models/ai_filter.json")),
        train_data_path=str(ai_raw.get("train_data_path", "data/trades.csv")),
        probability_threshold=float(ai_raw.get("probability_threshold", 0.6)),
    )

    return AppConfig(mt5=mt5, trade=trade, strategy=strategy, ai=ai)
