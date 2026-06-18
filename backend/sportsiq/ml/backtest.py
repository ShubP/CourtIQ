"""Backtesting + evaluation metrics for the prop models.

Time-ordered split (train on earlier games, evaluate on later ones) so there's
no look-ahead. Produces the numbers shown on the Model Performance dashboard:
projection MAE/RMSE, Brier score (raw vs calibrated), a calibration curve,
hit-rate and a proxy ROI. Run with:  python -m sportsiq.ml.backtest
"""
from __future__ import annotations

import json
import logging

import numpy as np

from ..config import MARKETS
from ..db import init_db, session_scope
from ..models import ModelMetric
from . import features as F
from .probability import expected_value, prob_over
from .train import _blend, _fit_calibrator, _new_regressor, estimate_dispersion, pick_blend

log = logging.getLogger("sportsiq.backtest")

_OFFSETS = (-1.5, -0.5, 0.5, 1.5)


def _calibration_curve(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Reliability curve: mean predicted vs observed frequency per probability bin."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        out.append({
            "pred": round(float(p[mask].mean()), 3),
            "actual": round(float(y[mask].mean()), 3),
            "n": int(mask.sum()),
        })
    return out


def backtest_market(logs_by_player: dict[int, list], market: str) -> dict | None:
    spec = MARKETS[market]
    stat = spec["stat"]
    frame, opp = F.build_training_frame(logs_by_player, stat)
    if frame.empty or len(frame) < 120:
        log.warning("Not enough data to backtest %s (rows=%d)", market, len(frame))
        return None

    cols = F.feature_columns(stat)
    frame = frame.sort_values("game_date").reset_index(drop=True)
    cut = int(len(frame) * 0.7)
    train, test = frame.iloc[:cut], frame.iloc[cut:]

    model = _new_regressor()
    model.fit(train[cols].to_numpy(float), train["target"].to_numpy(float))
    blend = pick_blend(train, cols, stat)
    base_col = f"{stat}_avg_season"
    mu_train = _blend(
        np.clip(model.predict(train[cols].to_numpy(float)), 0, None),
        train[base_col].to_numpy(float), blend,
    )
    disp = estimate_dispersion(train["target"].to_numpy(float), mu_train)
    calibrator, _ = _fit_calibrator(train, cols, stat, opp, blend)

    # Point-projection accuracy on the test set (blended prediction).
    mu_test = _blend(
        np.clip(model.predict(test[cols].to_numpy(float)), 0, None),
        test[base_col].to_numpy(float), blend,
    )
    actual = test["target"].to_numpy(float)
    mae = float(np.mean(np.abs(mu_test - actual)))
    rmse = float(np.sqrt(np.mean((mu_test - actual) ** 2)))

    # Baseline = naive "predict the player's season average so far".
    base_col = f"{stat}_avg_season"
    baseline = test[base_col].to_numpy(float) if base_col in test else mu_test
    baseline_mae = float(np.mean(np.abs(baseline - actual)))
    skill_pct = (
        round((baseline_mae - mae) / baseline_mae * 100, 1) if baseline_mae else 0.0
    )

    # Probability quality: evaluate several lines per row.
    raw_p, cal_p, outcome = [], [], []
    bet_wins = bet_n = 0
    roi = 0.0
    for mu_i, y_i in zip(mu_test, actual):
        base = round(mu_i)
        for k, off in enumerate(_OFFSETS):
            line = base + off
            if line < 0.5:
                continue
            rp = prob_over(float(mu_i), line, disp)
            cp = float(calibrator.predict([rp])[0]) if calibrator is not None else rp
            o = 1.0 if y_i > line else 0.0
            raw_p.append(rp)
            cal_p.append(cp)
            outcome.append(o)
            # Proxy bet only at the central line, priced at -110 both sides.
            if off == 0.5:
                side_over = cp >= 0.5
                won = (side_over and o == 1.0) or (not side_over and o == 0.0)
                roi += expected_value(1.0 if won else 0.0, -110)  # realized at -110
                bet_wins += int(won)
                bet_n += 1

    raw_p = np.array(raw_p)
    cal_p = np.array(cal_p)
    outcome = np.array(outcome)
    brier_raw = float(np.mean((raw_p - outcome) ** 2))
    brier_cal = float(np.mean((cal_p - outcome) ** 2))
    hit_rate = (bet_wins / bet_n) if bet_n else 0.0
    roi_proxy = (roi / bet_n) if bet_n else 0.0

    return {
        "market": market,
        "market_label": spec["label"],
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "baseline_mae": round(baseline_mae, 3),
        "skill_pct": skill_pct,
        "brier_raw": round(brier_raw, 4),
        "brier_calibrated": round(brier_cal, 4),
        "hit_rate": round(hit_rate, 4),
        "roi_proxy": round(roi_proxy, 4),
        "dispersion_r": round(float(disp), 2),
        "blend_weight": round(float(blend), 2),
        "calibration_json": json.dumps(_calibration_curve(cal_p, outcome)),
    }


def run_backtest(logs_by_group: dict[str, dict[int, list]]) -> list[dict]:
    """Backtest every market and persist metrics. Returns the metric dicts."""
    from .train import MODEL_VERSION
    init_db()
    results = []
    for market, spec in MARKETS.items():
        logs = logs_by_group.get(spec["group"], {})
        res = backtest_market(logs, market)
        if not res:
            continue
        res["model_version"] = MODEL_VERSION
        results.append(res)
        log.info(
            "Backtest %s: MAE=%.2f Brier raw=%.3f->cal=%.3f hit=%.1f%% disp_r=%.1f",
            market, res["mae"], res["brier_raw"], res["brier_calibrated"],
            res["hit_rate"] * 100, res["dispersion_r"],
        )
    if results:
        with session_scope() as s:
            for r in results:
                s.add(ModelMetric(**r))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Gather data the same way the pipeline does, then backtest.
    from ..pipeline import _gather_backtest_logs
    logs_by_group = _gather_backtest_logs()
    run_backtest(logs_by_group)
