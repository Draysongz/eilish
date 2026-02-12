from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


try:  # pragma: no cover - handled in runtime environments without xgboost
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


FeatureVector = Dict[str, float]


@dataclass
class AIConfig:
    enabled: bool
    model_path: Path
    train_data_path: Path
    probability_threshold: float


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def _session_flags(hour: int) -> Dict[str, int]:
    # Rough session buckets; adjust as needed.
    if 0 <= hour <= 7:
        return {"session_asia": 1, "session_europe": 0, "session_us": 0}
    if 8 <= hour <= 15:
        return {"session_asia": 0, "session_europe": 1, "session_us": 0}
    return {"session_asia": 0, "session_europe": 0, "session_us": 1}


def compute_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    data["ema20"] = _ema(data["close"], 20)
    data["ema50"] = _ema(data["close"], 50)
    data["ema_distance"] = (data["ema20"] - data["ema50"]) / data["close"]
    data["atr"] = _atr(data, 14) / data["close"]
    data["rsi"] = _rsi(data["close"], 14)
    data["body"] = (data["close"] - data["open"]).abs() / data["close"]
    data["volatility_5"] = data["close"].pct_change().rolling(5).std()

    time_col = data["time"] if "time" in data.columns else pd.Series(index=data.index, dtype="datetime64[ns]")
    timestamps = pd.to_datetime(time_col, errors="coerce")
    data["hour"] = timestamps.dt.hour.fillna(0).astype(int)
    session_flags = data["hour"].apply(_session_flags)
    session_df = pd.DataFrame(session_flags.tolist(), index=data.index)
    data = pd.concat([data, session_df], axis=1)

    features = data[[
        "ema_distance",
        "atr",
        "rsi",
        "body",
        "volatility_5",
        "hour",
        "session_asia",
        "session_europe",
        "session_us",
    ]]

    return features.dropna()


def compute_latest_features(frame: pd.DataFrame) -> Optional[FeatureVector]:
    features = compute_feature_frame(frame)
    if features.empty:
        return None
    latest = features.iloc[-1]
    return latest.to_dict()


def _extract_labels(frame: pd.DataFrame) -> pd.Series:
    if "label" in frame.columns:
        return frame["label"].astype(int)
    if "profit" in frame.columns:
        return (frame["profit"] > 0).astype(int)
    raise ValueError("Trades data must include a 'label' or 'profit' column.")


class XGBoostBackend:
    def __init__(self) -> None:
        if xgb is None:
            raise RuntimeError("xgboost is not installed. Install it to use the AI filter.")
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
        )

    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        self.model.fit(features, labels)

    def predict_proba(self, features: pd.DataFrame) -> float:
        proba = self.model.predict_proba(features)[0][1]
        return float(proba)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: Path) -> None:
        if xgb is None:
            raise RuntimeError("xgboost is not installed. Install it to use the AI filter.")
        self.model.load_model(str(path))


class AITradeFilter:
    def __init__(self, config: AIConfig) -> None:
        self.config = config
        self.backend = XGBoostBackend()
        self._load_or_train()

    def _load_or_train(self) -> None:
        if self.config.model_path.exists():
            self.backend.load(self.config.model_path)
            return

        if not self.config.train_data_path.exists():
            raise FileNotFoundError(
                f"Training data missing: {self.config.train_data_path}. Provide trades.csv or a pre-trained model."
            )

        raw = pd.read_csv(self.config.train_data_path)
        features = compute_feature_frame(raw)
        labels = _extract_labels(raw).reindex(features.index).dropna()
        features = features.loc[labels.index]

        if features.empty:
            raise ValueError("Not enough data to train the AI filter. Ensure trades.csv has sufficient bars.")

        self.backend.train(features, labels)
        self.backend.save(self.config.model_path)

    def evaluate(self, frame: pd.DataFrame) -> tuple[bool, float]:
        features = compute_latest_features(frame)
        if features is None:
            return False, 0.0
        feature_frame = pd.DataFrame([features])
        probability = self.backend.predict_proba(feature_frame)
        allowed = probability >= self.config.probability_threshold
        return allowed, probability


def build_ai_config(
    enabled: bool,
    model_path: str,
    train_data_path: str,
    probability_threshold: float,
) -> AIConfig:
    return AIConfig(
        enabled=enabled,
        model_path=Path(model_path),
        train_data_path=Path(train_data_path),
        probability_threshold=probability_threshold,
    )
