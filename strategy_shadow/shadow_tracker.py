from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from strategy_shadow.paper_executor import PaperTrade


class ShadowTracker:
    def __init__(self, csv_path: Path = Path("logs/shadow_trades.csv")) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "time_open",
                        "time_close",
                        "symbol",
                        "direction",
                        "entry",
                        "exit",
                        "sl",
                        "tp",
                        "result",
                        "profit",
                    ]
                )

    def record_close(self, trade: PaperTrade, exit_price: float) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    trade.open_time.isoformat(),
                    trade.close_time.isoformat() if trade.close_time else datetime.utcnow().isoformat(),
                    trade.symbol,
                    trade.direction,
                    f"{trade.entry_price:.5f}",
                    f"{exit_price:.5f}",
                    f"{trade.sl:.5f}",
                    f"{trade.tp:.5f}",
                    trade.result or "",
                    f"{trade.profit:.5f}" if trade.profit is not None else "",
                ]
            )
