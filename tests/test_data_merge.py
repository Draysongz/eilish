import pandas as pd

from src.data_merge import merge_dataframes


def test_merge_dataframes_deduplicates_and_sorts():
    df1 = pd.DataFrame(
        {
            "time": ["2024-01-01 00:00:00", "2024-01-01 00:01:00"],
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100, 110],
        }
    )
    df2 = pd.DataFrame(
        {
            "time": ["2024-01-01 00:01:00", "2024-01-01 00:02:00"],
            "open": [1.1, 1.2],
            "high": [1.2, 1.3],
            "low": [1.0, 1.1],
            "close": [1.15, 1.25],
            "volume": [110, 120],
        }
    )

    merged = merge_dataframes([df1, df2])
    assert len(merged) == 3
    assert list(merged["time"]) == sorted(merged["time"])
