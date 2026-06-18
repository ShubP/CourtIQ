"""Unit tests for the betting math (no network/DB)."""
import math

from sportsiq.ml.probability import (
    american_to_decimal,
    american_to_implied_prob,
    best_edge,
    expected_value,
    poisson_cdf,
    prob_over,
)


def test_american_to_decimal():
    assert math.isclose(american_to_decimal(100), 2.0)
    assert math.isclose(american_to_decimal(-110), 1.0 + 100 / 110)
    assert math.isclose(american_to_decimal(150), 2.5)


def test_implied_prob_sums_above_one_with_vig():
    # A -110/-110 market implies >100% total (the hold).
    total = american_to_implied_prob(-110) + american_to_implied_prob(-110)
    assert total > 1.0


def test_poisson_cdf_basic():
    # P(X<=0) for Poisson(1) = e^-1
    assert math.isclose(poisson_cdf(0, 1.0), math.exp(-1), rel_tol=1e-6)
    assert math.isclose(poisson_cdf(100, 2.0), 1.0, rel_tol=1e-6)


def test_prob_over_monotonic_in_prediction():
    low = prob_over(1.0, 1.5)
    high = prob_over(3.0, 1.5)
    assert 0 < low < high < 1


def test_expected_value_sign():
    # 60% to win at +100 (even money) is clearly +EV.
    assert expected_value(0.60, 100) > 0
    # 40% at -110 is -EV.
    assert expected_value(0.40, -110) < 0


def test_best_edge_picks_higher_ev_side():
    # Model loves the over (80%); over should win and be +EV.
    rec, edge = best_edge(0.80, over_odds=-110, under_odds=-110)
    assert rec == "Over"
    assert edge > 0


def test_best_edge_passes_when_no_value():
    # Coin flip at -110 both sides => no +EV side.
    rec, edge = best_edge(0.50, over_odds=-110, under_odds=-110)
    assert rec == "Pass"
