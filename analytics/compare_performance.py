from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
import csv

from analytics.metrics import calculate_metrics, load_production_trades, load_shadow_trades
from analytics.report_generator import generate_report


PROD_LOG = Path("logs/trades.csv")
SHADOW_LOG = Path("logs/shadow_trades.csv")
SUMMARY_LOG = Path("logs/performance_comparison.csv")


def write_summary(metrics, system: str, summary_path: Path = SUMMARY_LOG) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                [
                    "date",
                    "system",
                    "trades",
                    "winrate",
                    "net_profit",
                    "avg_rr",
                    "drawdown",
                    "churn_events",
                    "profit_factor",
                ]
            )
        writer.writerow(
            [
                date.today().isoformat(),
                system,
                metrics.total_trades,
                f"{metrics.winrate:.4f}",
                f"{metrics.net_profit:.2f}",
                f"{metrics.reward_risk_ratio:.4f}",
                f"{metrics.max_drawdown:.2f}",
                metrics.rapid_reentry_count,
                f"{metrics.profit_factor:.4f}" if metrics.profit_factor != float("inf") else "inf",
            ]
        )


def run_daily_comparison() -> str:
    prod_trades = load_production_trades(PROD_LOG)
    shadow_trades = load_shadow_trades(SHADOW_LOG)

    prod_metrics = calculate_metrics(prod_trades)
    shadow_metrics = calculate_metrics(shadow_trades)

    write_summary(prod_metrics, "production")
    write_summary(shadow_metrics, "shadow")

    return generate_report(prod_metrics, shadow_metrics)


def main() -> None:
    report = run_daily_comparison()
    print(report)


if __name__ == "__main__":
    main()
