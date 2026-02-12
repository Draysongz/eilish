from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests


@dataclass
class ImportConfig:
    provider: str
    symbol: str
    granularity: str
    start: Optional[str]
    end: Optional[str]
    input_path: Optional[Path]
    output_path: Path


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "time": "time",
        "date": "time",
        "timestamp": "time",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    renamed = {}
    for col in frame.columns:
        key = col.strip().lower()
        if key in column_map:
            renamed[col] = column_map[key]
    frame = frame.rename(columns=renamed)

    required = {"time", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"])
    frame = frame.sort_values("time")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0

    return frame[["time", "open", "high", "low", "close", "volume"]]


def import_from_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return _standardize_columns(frame)


def import_from_dukascopy_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return _standardize_columns(frame)


def _oanda_endpoint() -> str:
    env = os.getenv("OANDA_ENV", "practice")
    if env == "live":
        return "https://api-fxtrade.oanda.com/v3"
    return "https://api-fxpractice.oanda.com/v3"


def _iter_oanda_candles(symbol: str, granularity: str, start: str, end: str) -> Iterable[dict]:
    token = os.getenv("OANDA_API_KEY")
    if not token:
        raise RuntimeError("Missing OANDA_API_KEY in environment.")

    base_url = _oanda_endpoint()
    headers = {"Authorization": f"Bearer {token}"}

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    cursor = start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=7), end_dt)
        params = {
            "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "granularity": granularity,
            "price": "M",
        }
        response = requests.get(
            f"{base_url}/instruments/{symbol}/candles",
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        candles = payload.get("candles", [])
        if not candles:
            cursor = chunk_end
            continue
        for candle in candles:
            if not candle.get("complete", False):
                continue
            yield candle
        cursor = chunk_end


def import_from_oanda(symbol: str, granularity: str, start: str, end: str) -> pd.DataFrame:
    rows = []
    for candle in _iter_oanda_candles(symbol, granularity, start, end):
        mid = candle.get("mid", {})
        rows.append(
            {
                "time": candle.get("time"),
                "open": float(mid.get("o")),
                "high": float(mid.get("h")),
                "low": float(mid.get("l")),
                "close": float(mid.get("c")),
                "volume": float(candle.get("volume", 0)),
            }
        )
    frame = pd.DataFrame(rows)
    return _standardize_columns(frame)


def run_import(config: ImportConfig) -> Path:
    if config.provider == "csv":
        if not config.input_path:
            raise ValueError("--input is required for csv provider")
        data = import_from_csv(config.input_path)
    elif config.provider == "dukascopy":
        if not config.input_path:
            raise ValueError("--input is required for dukascopy provider")
        data = import_from_dukascopy_csv(config.input_path)
    elif config.provider == "oanda":
        if not config.start or not config.end:
            raise ValueError("--start and --end are required for oanda provider")
        data = import_from_oanda(config.symbol, config.granularity, config.start, config.end)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(config.output_path, index=False)
    return config.output_path


def _parse_args() -> ImportConfig:
    parser = argparse.ArgumentParser(description="Import price data into normalized CSV format.")
    parser.add_argument("--provider", choices=["csv", "dukascopy", "oanda"], default="csv")
    parser.add_argument("--symbol", default="EUR_USD")
    parser.add_argument("--granularity", default="M1")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--input")
    parser.add_argument("--output", default="data/price_history.csv")
    args = parser.parse_args()

    return ImportConfig(
        provider=args.provider,
        symbol=args.symbol,
        granularity=args.granularity,
        start=args.start,
        end=args.end,
        input_path=Path(args.input) if args.input else None,
        output_path=Path(args.output),
    )


def main() -> None:
    config = _parse_args()
    path = run_import(config)
    print(f"Saved price data to {path}")


if __name__ == "__main__":
    main()
