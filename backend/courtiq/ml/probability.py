"""Betting math: odds conversion, over/under probabilities, and edge (EV)."""
from __future__ import annotations

import math


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def american_to_implied_prob(odds: int) -> float:
    """Implied probability (includes the book's vig)."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def novig_prob_over(over_odds: int | None, under_odds: int | None) -> float | None:
    """Market's no-vig P(over) from two-way odds (removes the book's hold)."""
    if over_odds is None or under_odds is None:
        return None
    io = american_to_implied_prob(over_odds)
    iu = american_to_implied_prob(under_odds)
    if io + iu <= 0:
        return None
    return io / (io + iu)


def american_from_prob(p: float) -> int:
    """Inverse of american_to_implied_prob — price a probability as US odds."""
    p = min(max(p, 0.02), 0.98)
    if p >= 0.5:
        return -round(100.0 * p / (1.0 - p))
    return round(100.0 * (1.0 - p) / p)


def poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) for X ~ Poisson(lam)."""
    if lam <= 0:
        return 1.0
    total = 0.0
    term = math.exp(-lam)
    for i in range(0, k + 1):
        if i > 0:
            term *= lam / i
        total += term
    return min(1.0, total)


def prob_over(predicted: float, line: float, dispersion_r: float | None = None) -> float:
    """P(stat > line) given a predicted mean.

    Counting sports stats are *over-dispersed* (variance > mean), so a plain
    Poisson badly under-weights the tails and inflates edges. When a Negative
    Binomial dispersion `r` is supplied we model:
        Var = mean + mean^2 / r        (r -> inf reduces to Poisson)
    which widens the distribution realistically. Falls back to Poisson if r is
    None (or huge).
    """
    mean = max(predicted, 1e-6)
    floor_line = math.floor(line)  # lines are x.5 -> "over" = count >= floor+1

    if dispersion_r is None or dispersion_r > 1e6:
        p_over = 1.0 - poisson_cdf(floor_line, mean)
    else:
        from scipy.stats import nbinom

        r = max(dispersion_r, 1e-6)
        p = r / (r + mean)  # scipy nbinom: mean = r*(1-p)/p
        p_over = float(1.0 - nbinom.cdf(floor_line, r, p))

    return min(max(p_over, 1e-6), 1 - 1e-6)


def expected_value(prob_win: float, american_odds: int) -> float:
    """EV per $1 stake. Positive = +EV bet."""
    dec = american_to_decimal(american_odds)
    return prob_win * (dec - 1.0) - (1.0 - prob_win)


def best_edge(
    prob_over_val: float,
    over_odds: int | None,
    under_odds: int | None,
) -> tuple[str, float]:
    """Pick the side with higher EV; return (recommendation, edge).

    Edge here is the EV per $1 on the recommended side. If only one side has
    odds, evaluate that side. 'Pass' when neither side is +EV.
    """
    prob_under_val = 1.0 - prob_over_val
    candidates: list[tuple[str, float]] = []
    if over_odds is not None:
        candidates.append(("Over", expected_value(prob_over_val, over_odds)))
    if under_odds is not None:
        candidates.append(("Under", expected_value(prob_under_val, under_odds)))
    if not candidates:
        return "Pass", 0.0
    rec, edge = max(candidates, key=lambda c: c[1])
    if edge <= 0:
        return "Pass", edge
    return rec, edge
