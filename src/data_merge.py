from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd


@dataclass
class MergeConfig:
    inputs: List[Path]
    output_path: Path


def _standardize(frame: pd.DataFrame) -> pd.DataFrame:
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
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    return frame[["time", "open", "high", "low", "close", "volume"]]


def merge_dataframes(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["time"]).sort_values("time")
    return merged


def merge_files(config: MergeConfig) -> Path:
    frames = []
    for path in config.inputs:
        frame = pd.read_csv(path)
        frames.append(_standardize(frame))
    merged = merge_dataframes(frames)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(config.output_path, index=False)
    return config.output_path


def _parse_args() -> MergeConfig:
    parser = argparse.ArgumentParser(description="Merge multiple price CSVs into one dataset.")
    parser.add_argument("inputs", nargs="+", help="Input CSV files to merge")
    parser.add_argument("--output", default="data/merged_prices.csv")
    args = parser.parse_args()

    return MergeConfig(inputs=[Path(p) for p in args.inputs], output_path=Path(args.output))


def main() -> None:
    config = _parse_args()
    path = merge_files(config)
    print(f"Merged data saved to {path}")


if __name__ == "__main__":
    main()
