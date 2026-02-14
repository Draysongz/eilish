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


def _dukascopy_interval(granularity: str):
    try:
        import dukascopy_python as dukascopy
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "dukascopy-python is not installed. Install it with 'pip install -r requirements.txt'."
        ) from exc

    interval_map = {
        "S1": dukascopy.INTERVAL_SEC_1,
        "S10": dukascopy.INTERVAL_SEC_10,
        "S30": dukascopy.INTERVAL_SEC_30,
        "M1": dukascopy.INTERVAL_MIN_1,
        "M5": dukascopy.INTERVAL_MIN_5,
        "M10": dukascopy.INTERVAL_MIN_10,
        "M15": dukascopy.INTERVAL_MIN_15,
        "M30": dukascopy.INTERVAL_MIN_30,
        "H1": dukascopy.INTERVAL_HOUR_1,
        "H4": dukascopy.INTERVAL_HOUR_4,
        "D1": dukascopy.INTERVAL_DAY_1,
        "W1": dukascopy.INTERVAL_WEEK_1,
        "MN1": dukascopy.INTERVAL_MONTH_1,
    }
    if granularity not in interval_map:
        raise ValueError(f"Unsupported Dukascopy granularity: {granularity}")
    return interval_map[granularity]


def _dukascopy_instrument(symbol: str):
    try:
        from dukascopy_python import instruments
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "dukascopy-python is not installed. Install it with 'pip install -r requirements.txt'."
        ) from exc

    normalized = symbol.upper().replace("/", "_")
    if normalized == "XAUUSD":
        return instruments.INSTRUMENT_FX_METALS_XAU_USD
    if normalized == "XAGUSD":
        return instruments.INSTRUMENT_FX_METALS_XAG_USD

    attr_name = f"INSTRUMENT_FX_MAJORS_{normalized}"
    if hasattr(instruments, attr_name):
        return getattr(instruments, attr_name)
    return symbol


def _granularity_seconds(granularity: str) -> int:
    seconds_map = {
        "S1": 1,
        "S10": 10,
        "S30": 30,
        "M1": 60,
        "M5": 300,
        "M10": 600,
        "M15": 900,
        "M30": 1800,
        "H1": 3600,
        "H4": 14400,
        "D1": 86400,
        "W1": 604800,
        "MN1": 2592000,
    }
    return seconds_map.get(granularity, 60)


def _safe_dukascopy_fetch(
    symbol: str,
    interval,
    offer_side,
    start_dt: datetime,
    end_dt: datetime,
    granularity: str,
) -> pd.DataFrame:
    try:
        import dukascopy_python as dukascopy
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "dukascopy-python is not installed. Install it with 'pip install -r requirements.txt'."
        ) from exc

    cursor = int(start_dt.timestamp() * 1000)
    end_timestamp = int(end_dt.timestamp() * 1000)
    step_ms = _granularity_seconds(granularity) * 1000
    rows = []
    is_first_iteration = True

    while cursor <= end_timestamp:
        cleaned = []
        last_updates = dukascopy._fetch(
            instrument=symbol,
            interval=interval,
            offer_side=offer_side,
            last_update=cursor,
            limit=30_000,
        )

        if last_updates:
            cleaned = [row for row in last_updates if row and row[0] is not None]
            if not is_first_iteration and cleaned and cleaned[0][0] == cursor:
                cleaned = cleaned[1:]

            for row in cleaned:
                if row[0] > end_timestamp:
                    cursor = end_timestamp + 1
                    break
                if interval == dukascopy.INTERVAL_TICK:
                    row[-1] = row[-1] / 1_000_000
                    row[-2] = row[-2] / 1_000_000
                rows.append(row)
                cursor = row[0]

        if not last_updates or not cleaned:
            cursor = min(cursor + step_ms, end_timestamp + 1)

        is_first_iteration = False

    time_unit = dukascopy._interval_units[interval]
    columns = dukascopy._get_dataframe_columns_for_timeunit(time_unit)
    frame = pd.DataFrame(data=rows, columns=columns)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return frame.set_index("timestamp")


def import_from_dukascopy_api(
    symbol: str,
    granularity: str,
    start: str,
    end: str,
    offer_side: str = "bid",
) -> pd.DataFrame:
    try:
        import dukascopy_python as dukascopy
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "dukascopy-python is not installed. Install it with 'pip install -r requirements.txt'."
        ) from exc

    interval = _dukascopy_interval(granularity)
    instrument = _dukascopy_instrument(symbol)
    side = dukascopy.OFFER_SIDE_BID if offer_side.lower() == "bid" else dukascopy.OFFER_SIDE_ASK

    start_dt = pd.to_datetime(start, utc=True).to_pydatetime()
    end_dt = pd.to_datetime(end, utc=True).to_pydatetime()

    try:
        frame = dukascopy.fetch(instrument, interval, side, start_dt, end_dt)
    except TypeError:
        frame = _safe_dukascopy_fetch(
            symbol=instrument,
            interval=interval,
            offer_side=side,
            start_dt=start_dt,
            end_dt=end_dt,
            granularity=granularity,
        )

    if frame.empty:
        raise ValueError(
            "No Dukascopy data returned. Check symbol, date range, and granularity."
        )
    frame = frame.reset_index().rename(columns={"timestamp": "time"})
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
    elif config.provider == "dukascopy_api":
        if not config.start or not config.end:
            raise ValueError("--start and --end are required for dukascopy_api provider")
        data = import_from_dukascopy_api(
            symbol=config.symbol,
            granularity=config.granularity,
            start=config.start,
            end=config.end,
        )
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
    parser.add_argument("--provider", choices=["csv", "dukascopy", "dukascopy_api", "oanda"], default="csv")
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
