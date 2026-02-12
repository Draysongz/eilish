import pandas as pd

from src.ai_filter import compute_feature_frame


def _make_frame(rows: int = 80):
    data = {
        "time": pd.date_range("2024-01-01", periods=rows, freq="min"),
        "open": [1.0 + i * 0.001 for i in range(rows)],
        "high": [1.002 + i * 0.001 for i in range(rows)],
        "low": [0.998 + i * 0.001 for i in range(rows)],
        "close": [1.001 + i * 0.001 for i in range(rows)],
    }
    return pd.DataFrame(data)


def test_feature_frame_has_expected_columns():
    frame = _make_frame()
    features = compute_feature_frame(frame)
    assert not features.empty
    for column in [
        "ema_distance",
        "atr",
        "rsi",
        "body",
        "volatility_5",
        "hour",
        "session_asia",
        "session_europe",
        "session_us",
    ]:
        assert column in features.columns
