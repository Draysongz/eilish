from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskCheck:
    allowed: bool
    reason: str


def evaluate_risk(spread: float, max_spread: float, open_positions: int, max_positions: int) -> RiskCheck:
    if spread > max_spread:
        return RiskCheck(False, f"Spread too high: {spread:.2f} pips")
    if open_positions >= max_positions:
        return RiskCheck(False, f"Max positions reached: {open_positions}")
    return RiskCheck(True, "OK")
