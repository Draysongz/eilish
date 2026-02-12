from src.risk import evaluate_risk


def test_risk_blocks_high_spread():
    result = evaluate_risk(spread=2.5, max_spread=1.0, open_positions=0, max_positions=1)
    assert not result.allowed
    assert "Spread" in result.reason


def test_risk_blocks_max_positions():
    result = evaluate_risk(spread=0.5, max_spread=1.0, open_positions=2, max_positions=2)
    assert not result.allowed
    assert "Max positions" in result.reason


def test_risk_allows_safe_trade():
    result = evaluate_risk(spread=0.5, max_spread=1.0, open_positions=0, max_positions=2)
    assert result.allowed
