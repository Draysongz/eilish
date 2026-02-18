from __future__ import annotations

import os
import sys
from dataclasses import asdict
from typing import Dict, Tuple

from analytics.metrics import PerformanceMetrics


def _supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def _color(text: str, color_code: str) -> str:
    if not _supports_color():
        return text
    return f"\x1b[{color_code}m{text}\x1b[0m"


def _better_is_higher(metric_name: str) -> bool:
    lower_is_better = {
        "total_trades",
        "trade_frequency",
        "max_drawdown",
        "losses",
        "consecutive_losses_max",
        "rapid_reentry_count",
    }
    return metric_name not in lower_is_better


def _format_metric(metric_name: str, value: float) -> str:
    if metric_name in {"winrate"}:
        return f"{value * 100:.1f}%"
    if metric_name in {"net_profit", "average_win", "average_loss", "max_drawdown"}:
        return f"{value:+.2f}"
    if metric_name in {"reward_risk_ratio", "profit_factor"}:
        return f"{value:.2f}" if value != float("inf") else "inf"
    if metric_name in {"trade_frequency"}:
        return f"{value:.2f}/hr"
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.2f}"


def _compare(metric_name: str, prod: float, shadow: float) -> Tuple[str, str]:
    higher_is_better = _better_is_higher(metric_name)
    if higher_is_better:
        prod_better = prod > shadow
        shadow_better = shadow > prod
    else:
        prod_better = prod < shadow
        shadow_better = shadow < prod

    prod_text = _format_metric(metric_name, prod)
    shadow_text = _format_metric(metric_name, shadow)

    if prod_better:
        prod_text = _color(prod_text, "32")
    elif shadow_better:
        shadow_text = _color(shadow_text, "32")

    return prod_text, shadow_text


def generate_report(prod: PerformanceMetrics, shadow: PerformanceMetrics) -> str:
    metrics_map: Dict[str, str] = {
        "total_trades": "Total Trades",
        "wins": "Wins",
        "losses": "Losses",
        "winrate": "Winrate",
        "net_profit": "Net Profit",
        "average_win": "Average Win",
        "average_loss": "Average Loss",
        "reward_risk_ratio": "Avg RR",
        "max_drawdown": "Max Drawdown",
        "trade_frequency": "Trade Frequency",
        "consecutive_losses_max": "Max Loss Streak",
        "profit_factor": "Profit Factor",
        "rapid_reentry_count": "Rapid Re-entries",
    }

    prod_dict = asdict(prod)
    shadow_dict = asdict(shadow)

    lines = [
        "================ PERFORMANCE COMPARISON ================",
        "",
        f"{'Metric':<26} {'Production':>14} {'Shadow':>14}",
    ]

    for key, label in metrics_map.items():
        prod_val = float(prod_dict.get(key, 0.0))
        shadow_val = float(shadow_dict.get(key, 0.0))
        prod_text, shadow_text = _compare(key, prod_val, shadow_val)
        lines.append(f"{label:<26} {prod_text:>14} {shadow_text:>14}")

    lines.append("")
    return "\n".join(lines)
