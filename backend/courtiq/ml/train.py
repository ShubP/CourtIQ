"""Train one model per prop market and persist artifacts to models/.

Each artifact bundles three things so predictions are realistic and calibrated:
  1. an XGBoost mean model (Poisson objective) for the expected stat,
  2. a Negative-Binomial dispersion `r` capturing real over-dispersion,
  3. an isotonic calibrator mapping raw P(over) -> empirically-calibrated P(over),
     fit on a time-held-out slice so probabilities match reality.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRegressor

from ..config import MARKETS, MODELS_DIR
from . import features as F
from .probability import prob_over

log = logging.getLogger(__name__)
MODEL_VERSION = "xgb-negbinom-isotonic-v2"


@dataclass
class ModelArtifact:
    market: str
    stat: str
    group: str
    feature_columns: list[str]
    opp_factor: dict
    league_mean: float
    model: object
    dispersion_r: float
    blend_weight: float  # weight on GBM vs season-average baseline (0..1)
    calibrator: object | None
    model_version: str
    trained_at: str
    n_rows: int
    fallback_mean: float


def _model_path(market: str):
    return MODELS_DIR / f"{market}.joblib"


def _new_regressor() -> XGBRegressor:
    return XGBRegressor(
        objective="count:poisson",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        n_jobs=0,
        random_state=42,
    )


def estimate_dispersion(y_true: np.ndarray, mu: np.ndarray) -> float:
    """Aggregate method-of-moments NegBinom size r from Var = mu + mu^2/r.

    Uses aggregate moments (not per-row clipped ratios, which bias toward
    Poisson): r = E[mu^2] / (E[(y-mu)^2] - E[mu]).
    """
    mu = np.clip(mu, 1e-6, None)
    resid_var = float(np.mean((y_true - mu) ** 2))
    mean_mu = float(np.mean(mu))
    mean_mu2 = float(np.mean(mu**2))
    excess = resid_var - mean_mu  # over-dispersion beyond Poisson
    if excess <= 1e-6:
        return 1e9  # ~Poisson (no meaningful over-dispersion)
    return float(min(max(mean_mu2 / excess, 0.5), 1e6))


def pick_blend(frame, cols, stat) -> float:
    """Tune the GBM/baseline blend on a time-held-out slice (minimize MAE).

    The 'baseline' is the player's running season average ({stat}_avg_season),
    which is surprisingly strong; blending guards against the GBM underfitting.
    """
    base_col = f"{stat}_avg_season"
    if base_col not in frame or len(frame) < 120:
        return 1.0
    frame = frame.sort_values("game_date").reset_index(drop=True)
    cut = int(len(frame) * 0.8)
    early, late = frame.iloc[:cut], frame.iloc[cut:]
    if len(late) < 25:
        return 1.0
    m = _new_regressor()
    m.fit(early[cols].to_numpy(float), early["target"].to_numpy(float))
    mu = np.clip(m.predict(late[cols].to_numpy(float)), 0, None)
    base = late[base_col].to_numpy(float)
    actual = late["target"].to_numpy(float)
    best_w, best_mae = 1.0, 1e9
    for w in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        mae = float(np.mean(np.abs(w * mu + (1 - w) * base - actual)))
        if mae < best_mae:
            best_w, best_mae = w, mae
    return best_w


def _blend(mu: np.ndarray, base: np.ndarray, w: float) -> np.ndarray:
    return np.clip(w * mu + (1 - w) * base, 0, None)


def _fit_calibrator(frame, cols, stat, opp, blend: float = 1.0) -> tuple[object | None, float]:
    """Time-split: train on early rows, fit isotonic on later rows' P(over).

    Probabilities are computed from the *blended* projection so calibration
    and dispersion match what we actually predict with.
    """
    base_col = f"{stat}_avg_season"
    if len(frame) < 120:
        return None, 1e9
    frame = frame.sort_values("game_date").reset_index(drop=True)
    cut = int(len(frame) * 0.8)
    early, late = frame.iloc[:cut], frame.iloc[cut:]
    if len(late) < 25:
        return None, 1e9

    m = _new_regressor()
    m.fit(early[cols].to_numpy(float), early["target"].to_numpy(float))
    mu_early = _blend(
        np.clip(m.predict(early[cols].to_numpy(float)), 0, None),
        early[base_col].to_numpy(float), blend,
    )
    disp = estimate_dispersion(early["target"].to_numpy(float), mu_early)

    mu_late = _blend(
        np.clip(m.predict(late[cols].to_numpy(float)), 0, None),
        late[base_col].to_numpy(float), blend,
    )
    actual = late["target"].to_numpy(float)
    # Evaluate several lines around each projection so the calibrator learns
    # across the whole probability range (not just near P=0.5).
    raw_p, outcome = [], []
    for mu_i, y_i in zip(mu_late, actual):
        base = round(mu_i)
        for off in (-1.5, -0.5, 0.5, 1.5):
            line = base + off
            if line < 0.5:
                continue
            raw_p.append(prob_over(float(mu_i), line, disp))
            outcome.append(1.0 if y_i > line else 0.0)
    raw_p = np.array(raw_p)
    outcome = np.array(outcome)
    if len(outcome) < 40 or len(set(outcome)) < 2:
        return None, disp

    # Fit on first 70% of these pairs, validate on the rest; keep the
    # calibrator only if it actually lowers Brier (guards against hurting).
    n = len(raw_p)
    idx = int(n * 0.7)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_p[:idx], outcome[:idx])
    val_raw, val_y = raw_p[idx:], outcome[idx:]
    if len(val_y) >= 10:
        brier_raw = float(np.mean((val_raw - val_y) ** 2))
        brier_cal = float(np.mean((iso.predict(val_raw) - val_y) ** 2))
        if brier_cal >= brier_raw:
            return None, disp  # calibration doesn't help -> skip it
    # Refit on all pairs for the production calibrator.
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_p, outcome)
    return iso, disp


def train_market(logs_by_player: dict[int, list], market: str) -> ModelArtifact | None:
    spec = MARKETS[market]
    stat = spec["stat"]
    frame, opp = F.build_training_frame(logs_by_player, stat)
    if frame.empty or len(frame) < 30:
        log.warning("Not enough data to train %s (rows=%d)", market, len(frame))
        return None

    cols = F.feature_columns(stat)
    X = frame[cols].to_numpy(dtype=float)
    y = frame["target"].to_numpy(dtype=float)

    # Tune the GBM/baseline blend, then fit calibrator + dispersion on the
    # blended projection from a time-held-out slice.
    blend_weight = pick_blend(frame, cols, stat)
    calibrator, _disp_cal = _fit_calibrator(frame, cols, stat, opp, blend_weight)

    # Final model trained on all rows; dispersion estimated on blended preds.
    model = _new_regressor()
    model.fit(X, y)
    base_all = frame[f"{stat}_avg_season"].to_numpy(float)
    mu_all = _blend(np.clip(model.predict(X), 0, None), base_all, blend_weight)
    dispersion_r = estimate_dispersion(y, mu_all)

    artifact = ModelArtifact(
        market=market,
        stat=stat,
        group=spec["group"],
        feature_columns=cols,
        opp_factor=opp.get("opp_factor", {}),
        league_mean=float(opp.get("league_mean", float(np.mean(y)))),
        model=model,
        dispersion_r=dispersion_r,
        blend_weight=blend_weight,
        calibrator=calibrator,
        model_version=MODEL_VERSION,
        trained_at=datetime.utcnow().isoformat(),
        n_rows=int(len(frame)),
        fallback_mean=float(np.mean(y)),
    )
    joblib.dump(artifact, _model_path(market))
    log.info(
        "Trained %s: rows=%d blend=%.2f dispersion_r=%.1f calibrated=%s",
        market, len(frame), blend_weight, dispersion_r, calibrator is not None,
    )
    return artifact


def train_all(logs_by_group: dict[str, dict[int, list]]) -> dict[str, ModelArtifact]:
    """Train every market. logs_by_group maps 'hitting'/'pitching' -> {player_id: logs}."""
    trained: dict[str, ModelArtifact] = {}
    for market, spec in MARKETS.items():
        logs = logs_by_group.get(spec["group"], {})
        art = train_market(logs, market)
        if art:
            trained[market] = art
    return trained


def load_model(market: str) -> ModelArtifact | None:
    path = _model_path(market)
    if not path.exists():
        return None
    return joblib.load(path)
