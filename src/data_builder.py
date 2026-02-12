from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import AppConfig, load_config
from src.mt5_client import MT5Client, credentials_from_env
from src.strategy import generate_signal


@dataclass
class BuildConfig:
    symbol: str
    timeframe: str
    bars: int
    horizon: int
    output_path: Path


def _pip_size(symbol_info) -> float:
    point = symbol_info.point or 0.00001
    return point * 10 if symbol_info.digits in (3, 5) else point


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

        rows = []
        for idx in range(app_config.strategy.min_bars, len(rates) - build_config.horizon):
            window = rates.iloc[: idx + 1]
            state = generate_signal(
                window,
                app_config.strategy.ema_fast,
                app_config.strategy.ema_slow,
                app_config.strategy.min_bars,
                app_config.strategy.allow_short,
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
    finally:
        client.shutdown()


def _parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description="Build AI training data from MT5 history.")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="M1")
    parser.add_argument("--bars", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--output", default="data/trades.csv")
    args = parser.parse_args()

    return BuildConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        bars=args.bars,
        horizon=args.horizon,
        output_path=Path(args.output),
    )


def main() -> None:
    app_config = load_config()
    build_config = _parse_args()
    dataset = build_training_data(app_config, build_config)
    print(f"Saved {len(dataset)} rows to {build_config.output_path}")


if __name__ == "__main__":
    main()
