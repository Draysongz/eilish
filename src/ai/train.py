from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ai.metrics import FoldMetrics, optimize_threshold


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray


def _month_floor(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M")


def build_walk_forward_folds(
    frame: pd.DataFrame,
    time_col: str,
    train_months: int = 3,
    val_months: int = 1,
    min_folds: int = 3,
) -> List[WalkForwardFold]:
    months = _month_floor(frame[time_col])
    unique_months = sorted(months.dropna().unique())

    folds: List[WalkForwardFold] = []
    for start in range(0, len(unique_months) - (train_months + val_months) + 1):
        train_start = unique_months[start]
        train_end = unique_months[start + train_months - 1]
        val_end = unique_months[start + train_months + val_months - 1]

        train_mask = (months >= train_start) & (months <= train_end)
        val_mask = (months > train_end) & (months <= val_end)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        folds.append(
            WalkForwardFold(
                fold=len(folds) + 1,
                train_idx=train_idx,
                val_idx=val_idx,
            )
        )

    if len(folds) < min_folds:
        raise ValueError("Not enough data for the required walk-forward folds.")

    return folds


def train_walk_forward(
    frame: pd.DataFrame,
    feature_cols: Iterable[str],
    time_col: str,
    label_col: str,
    weight_col: str,
    tp_pips: float,
    sl_pips: float,
    thresholds: Iterable[float],
    train_months: int = 3,
    val_months: int = 1,
) -> List[FoldMetrics]:
    features = frame[list(feature_cols)].to_numpy(dtype=float)
    labels = frame[label_col].to_numpy(dtype=int)
    weights = frame[weight_col].to_numpy(dtype=float)

    folds = build_walk_forward_folds(
        frame,
        time_col=time_col,
        train_months=train_months,
        val_months=val_months,
    )

    results: List[FoldMetrics] = []
    for fold in folds:
        x_train = features[fold.train_idx]
        y_train = labels[fold.train_idx]
        w_train = weights[fold.train_idx]

        x_val = features[fold.val_idx]
        y_val = labels[fold.val_idx]

        model = LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            n_jobs=1,
        )
        model.fit(x_train, y_train, sample_weight=w_train)
        proba = model.predict_proba(x_val)[:, 1]

        best_threshold, metrics = optimize_threshold(
            y_val,
            proba,
            tp_pips,
            sl_pips,
            thresholds,
        )

        results.append(
            FoldMetrics(
                fold=fold.fold,
                trades=metrics["trades"],
                win_rate=metrics["win_rate"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                expectancy=metrics["expectancy"],
                profit_factor=metrics["profit_factor"],
                best_threshold=best_threshold,
            )
        )

    return results


def summary_table(results: List[FoldMetrics]) -> str:
    header = "fold | trades | win_rate | expectancy | profit_factor | best_threshold"
    lines = [header]
    for row in results:
        lines.append(
            f"{row.fold} | {row.trades} | {row.win_rate:.3f} | "
            f"{row.expectancy:.2f} | {row.profit_factor:.2f} | {row.best_threshold:.2f}"
        )
    return "\n".join(lines)
