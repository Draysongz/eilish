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
    sl_mode: str
    atr_sl_multiplier: float
    rr_ratio: float
    max_sl_cap_points: float = 400.0


@dataclass(frozen=True)
class StrategyConfig:
    ema_fast: int
    ema_slow: int
    min_bars: int
    allow_short: bool
    entry_delay_bars: int
    use_rsi: bool
    rsi_period: int
    rsi_overbought: float
    rsi_oversold: float
    use_atr: bool
    atr_period: int
    atr_min_threshold: float
    atr_min_thresholds: Dict[str, float]
    # Entry refinement filters
    use_expansion_candle_filter: bool = True
    expansion_lookback: int = 10
    expansion_multiplier: float = 1.5
    use_distance_from_ema_filter: bool = True
    distance_from_ema_multiplier: float = 1.2
    use_atr_spike_filter: bool = True
    atr_spike_lookback: int = 20
    atr_spike_multiplier: float = 1.5
    use_break_structure_filter: bool = True
    break_structure_lookback: int = 5


@dataclass(frozen=True)
class ProfitTakeFilterConfig:
    enabled: bool
    trigger_usd: float
    risk_threshold: float
    take_profit_usd_per_symbol: Dict[str, float]
    check_interval_seconds: int


@dataclass(frozen=True)
class BreakevenConfig:
    enabled: bool
    activation_ratio: float  # Move SL to BE when profit reaches this ratio of initial risk


@dataclass(frozen=True)
class AppConfig:
    mt5: MT5Config
    trade: TradeConfig
    strategy: StrategyConfig
    profit_take: ProfitTakeFilterConfig
    breakeven: BreakevenConfig
    ai: "AIConfig"
    shadow_testing: "ShadowTestingConfig"


@dataclass(frozen=True)
class AIConfig:
    enabled: bool
    model_path: str
    train_data_path: str
    probability_threshold: float


@dataclass(frozen=True)
class ShadowTestingConfig:
    enabled: bool


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
    shadow_raw = raw.get("shadow_testing", {})

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
            sl_mode=str(overrides.get("sl_mode", "fixed")),
            atr_sl_multiplier=float(overrides.get("atr_sl_multiplier", 2.0)),
            rr_ratio=float(overrides.get("rr_ratio", 1.0)),
            max_sl_cap_points=float(overrides.get("max_sl_cap_points", 400.0)),
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
        entry_delay_bars=int(strategy_raw.get("entry_delay_bars", 0)),
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
        # Entry refinement filters
        use_expansion_candle_filter=bool(strategy_raw.get("use_expansion_candle_filter", True)),
        expansion_lookback=int(strategy_raw.get("expansion_lookback", 10)),
        expansion_multiplier=float(strategy_raw.get("expansion_multiplier", 1.5)),
        use_distance_from_ema_filter=bool(strategy_raw.get("use_distance_from_ema_filter", True)),
        distance_from_ema_multiplier=float(strategy_raw.get("distance_from_ema_multiplier", 1.2)),
        use_atr_spike_filter=bool(strategy_raw.get("use_atr_spike_filter", True)),
        atr_spike_lookback=int(strategy_raw.get("atr_spike_lookback", 20)),
        atr_spike_multiplier=float(strategy_raw.get("atr_spike_multiplier", 1.5)),
        use_break_structure_filter=bool(strategy_raw.get("use_break_structure_filter", True)),
        break_structure_lookback=int(strategy_raw.get("break_structure_lookback", 5)),
    )

    profit_take = ProfitTakeFilterConfig(
        enabled=bool(profit_raw.get("enabled", False)),
        trigger_usd=float(profit_raw.get("trigger_usd", 1.0)),
        risk_threshold=float(profit_raw.get("risk_threshold", 0.6)),
        take_profit_usd_per_symbol={
            str(k): float(v)
            for k, v in (profit_raw.get("take_profit_usd_per_symbol", {}) or {}).items()
        },
        check_interval_seconds=int(profit_raw.get("check_interval_seconds", 1)),
    )

    ai = AIConfig(
        enabled=bool(ai_raw.get("enabled", False)),
        model_path=str(ai_raw.get("model_path", "models/ai_filter.json")),
        train_data_path=str(ai_raw.get("train_data_path", "data/trades.csv")),
        probability_threshold=float(ai_raw.get("probability_threshold", 0.6)),
    )

    shadow_testing = ShadowTestingConfig(
        enabled=bool(shadow_raw.get("enabled", False)),
    )

    breakeven_raw = raw.get("breakeven", {})
    breakeven = BreakevenConfig(
        enabled=bool(breakeven_raw.get("enabled", False)),
        activation_ratio=float(breakeven_raw.get("activation_ratio", 0.8)),
    )

    return AppConfig(
        mt5=mt5,
        trade=trade,
        strategy=strategy,
        profit_take=profit_take,
        breakeven=breakeven,
        ai=ai,
        shadow_testing=shadow_testing,
    )
