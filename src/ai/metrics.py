from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FoldMetrics:
    fold: int
    trades: int
    win_rate: float
    precision: float
    recall: float
    expectancy: float
    profit_factor: float
    best_threshold: float


def _safe_div(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


def evaluate_threshold(
    y_true: Iterable[int],
    proba: Iterable[float],
    threshold: float,
    tp_pips: float,
    sl_pips: float,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)

    take_mask = proba >= threshold
    trades = int(take_mask.sum())
    if trades == 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
        }

    taken_y = y_true[take_mask]
    wins = int((taken_y == 1).sum())
    losses = trades - wins

    win_rate = _safe_div(wins, trades)
    precision = win_rate
    recall = _safe_div(wins, int((y_true == 1).sum()))

    expectancy = win_rate * tp_pips - (1 - win_rate) * sl_pips
    gross_profit = wins * tp_pips
    gross_loss = losses * sl_pips
    profit_factor = _safe_div(gross_profit, gross_loss)

    return {
        "trades": trades,
        "win_rate": win_rate,
        "precision": precision,
        "recall": recall,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
    }


def optimize_threshold(
    y_true: Iterable[int],
    proba: Iterable[float],
    tp_pips: float,
    sl_pips: float,
    thresholds: Iterable[float],
) -> tuple[float, dict]:
    best_threshold = None
    best_metrics = None
    best_expectancy = -float("inf")

    for threshold in thresholds:
        metrics = evaluate_threshold(y_true, proba, threshold, tp_pips, sl_pips)
        if metrics["expectancy"] > best_expectancy:
            best_expectancy = metrics["expectancy"]
            best_threshold = threshold
            best_metrics = metrics

    return float(best_threshold), best_metrics
