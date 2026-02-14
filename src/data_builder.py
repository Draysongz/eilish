from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import re

import pandas as pd

from src.config import AppConfig, load_config
from src.mt5_client import MT5Client, credentials_from_env
from src.price_importer import import_from_csv
from src.strategy import generate_signal


@dataclass
class BuildConfig:
    symbol: str
    timeframe: str
    bars: int
    horizon: int
    output_path: Path
    input_path: Optional[Path]
    session: str
    pip_size: Optional[float]


def _pip_size(symbol_info) -> float:
    point = symbol_info.point or 0.00001
    return point * 10 if symbol_info.digits in (3, 5) else point


def _pip_size_from_symbol(symbol: str) -> float:
    symbol = symbol.upper()
    if symbol.startswith("XAU") or symbol.startswith("XAG"):
        return 0.01
    if symbol.endswith("JPY"):
        return 0.01
    return 0.0001


def _parse_sessions(value: str) -> set[str]:
    normalized = value.strip().lower()
    if not normalized or normalized == "all":
        return set()
    tokens = re.split(r"[,+ ]+", normalized)
    sessions = {token for token in tokens if token}
    return sessions


def _filter_sessions(frame: pd.DataFrame, session: str) -> pd.DataFrame:
    sessions = _parse_sessions(session)
    if not sessions:
        return frame

    if "time" not in frame.columns:
        raise ValueError("Rates data must include a 'time' column for session filtering.")

    time_values = pd.to_datetime(frame["time"], errors="coerce", utc=True)
    hours = time_values.dt.hour

    mask = pd.Series(False, index=frame.index)
    if "london" in sessions:
        mask |= (hours >= 8) & (hours < 17)
    if "newyork" in sessions or "ny" in sessions:
        mask |= (hours >= 13) & (hours < 22)

    filtered = frame.loc[mask].copy()
    return filtered


def label_trade_outcome(
    rates: pd.DataFrame,
    index: int,
    direction: str,
    tp_pips: float,
    sl_pips: float,
    pip_size: float,
    horizon: int,
) -> int:
    entry_price = float(rates.iloc[index]["close"])
    future = rates.iloc[index + 1 : index + 1 + horizon]

    for _, row in future.iterrows():
        high = float(row["high"])
        low = float(row["low"])

        if direction == "buy":
            hit_tp = high - entry_price >= tp_pips * pip_size
            hit_sl = entry_price - low >= sl_pips * pip_size
        else:
            hit_tp = entry_price - low >= tp_pips * pip_size
            hit_sl = high - entry_price >= sl_pips * pip_size

        if hit_tp and hit_sl:
            return 0
        if hit_tp:
            return 1
        if hit_sl:
            return 0

    return 0


def build_training_data(app_config: AppConfig, build_config: BuildConfig) -> pd.DataFrame:
    if build_config.input_path:
        rates = import_from_csv(build_config.input_path)
        pip_size = build_config.pip_size or _pip_size_from_symbol(build_config.symbol)
    else:
        creds = credentials_from_env(
            app_config.mt5.login_env,
            app_config.mt5.password_env,
            app_config.mt5.server_env,
            app_config.mt5.path_env,
        )

        client = MT5Client(creds)
        client.initialize()

        try:
            rates = client.get_rates(build_config.symbol, build_config.timeframe, build_config.bars)
            symbol_info = client.symbol_info(build_config.symbol)
            if symbol_info is None:
                raise RuntimeError(f"Symbol info missing for {build_config.symbol}")
            pip_size = _pip_size(symbol_info)
        finally:
            client.shutdown()

    rates = rates.sort_values("time") if "time" in rates.columns else rates
    rates = _filter_sessions(rates, build_config.session)
    if rates.empty:
        raise ValueError("No rates available after applying the session filter.")

    rows = []
    start_index = app_config.strategy.min_bars
    end_index = len(rates) - build_config.horizon
    total_steps = max(0, end_index - start_index)
    start_time = time.time()

    for idx in range(start_index, end_index):
        if total_steps and (idx - start_index) % 1000 == 0:
            elapsed = time.time() - start_time
            processed = idx - start_index
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = total_steps - processed
            eta_sec = remaining / rate if rate > 0 else 0.0
            print(
                f"Progress: {processed}/{total_steps} bars "
                f"({processed / total_steps:.1%}) | "
                f"ETA {eta_sec/60:.1f} min"
            )
        window = rates.iloc[: idx + 1]
        state = generate_signal(
            window,
            app_config.strategy.ema_fast,
            app_config.strategy.ema_slow,
            app_config.strategy.min_bars,
            app_config.strategy.allow_short,
            app_config.strategy.use_rsi,
            app_config.strategy.rsi_period,
            app_config.strategy.rsi_overbought,
            app_config.strategy.rsi_oversold,
            app_config.strategy.use_atr,
            app_config.strategy.atr_period,
            app_config.strategy.atr_min_threshold,
        )
        if state.signal == "hold":
            continue

        label = label_trade_outcome(
            rates,
            idx,
            state.signal,
            app_config.trade.tp_pips,
            app_config.trade.sl_pips,
            pip_size,
            build_config.horizon,
        )

        row = rates.iloc[idx]
        rows.append(
            {
                "time": row["time"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "label": label,
            }
        )

    dataset = pd.DataFrame(rows)
    build_config.output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(build_config.output_path, index=False)
    return dataset


def _parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(
        description="Build AI training data from MT5 history or a price CSV."
    )
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M1")
    parser.add_argument("--bars", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--output", default="data/trades.csv")
    parser.add_argument("--input", help="Optional CSV file with price history.")
    parser.add_argument(
        "--session",
        default="all",
        help="Session filter: all, london, newyork, or london+newyork (UTC).",
    )
    parser.add_argument("--pip-size", type=float, help="Override pip size when using --input.")
    args = parser.parse_args()

    return BuildConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        bars=args.bars,
        horizon=args.horizon,
        output_path=Path(args.output),
        input_path=Path(args.input) if args.input else None,
        session=args.session,
        pip_size=args.pip_size,
    )


def main() -> None:
    app_config = load_config()
    build_config = _parse_args()
    dataset = build_training_data(app_config, build_config)
    print(f"Saved {len(dataset)} rows to {build_config.output_path}")


if __name__ == "__main__":
    main()
