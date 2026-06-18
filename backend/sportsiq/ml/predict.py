"""Turn a trained model + a player's history + a sportsbook line into an edge."""
from __future__ import annotations

import numpy as np

from ..config import MODEL_VS_MARKET
from .probability import best_edge, novig_prob_over, prob_over
from .train import ModelArtifact
from . import features as F


def predict_one(
    artifact: ModelArtifact,
    logs: list,
    line: float,
    over_odds: int | None,
    under_odds: int | None,
    is_home: int,
    opponent_team_id: int | None,
) -> dict | None:
    """Predict one player/market vs a line. Returns None if history is too thin."""
    feat = F.build_prediction_features(
        logs, artifact.stat, is_home, opponent_team_id, artifact.opp_factor
    )
    if feat is None:
        return None

    x = np.array([[feat.get(c, 0.0) for c in artifact.feature_columns]], dtype=float)
    model_pred = max(float(artifact.model.predict(x)[0]), 0.0)
    # Blend the GBM with the season-average baseline (tuned in training).
    blend = getattr(artifact, "blend_weight", 1.0)
    baseline = feat.get(f"{artifact.stat}_avg_season", model_pred)
    predicted = max(blend * model_pred + (1 - blend) * baseline, 0.0)

    # Negative-Binomial tail prob, then isotonic calibration if available.
    dispersion = getattr(artifact, "dispersion_r", None)
    p_over = prob_over(predicted, line, dispersion)
    calibrator = getattr(artifact, "calibrator", None)
    if calibrator is not None:
        p_over = float(calibrator.predict([p_over])[0])
        p_over = min(max(p_over, 1e-6), 1 - 1e-6)

    # Blend with the market's no-vig probability (sharp markets are a strong
    # prior) so edges reflect confident disagreement, not model noise.
    market_p = novig_prob_over(over_odds, under_odds)
    if market_p is not None:
        w = MODEL_VS_MARKET
        p_over = min(max(w * p_over + (1 - w) * market_p, 1e-6), 1 - 1e-6)

    recommendation, edge = best_edge(p_over, over_odds, under_odds)

    return {
        "predicted_value": round(predicted, 2),
        "prob_over": round(p_over, 4),
        "prob_under": round(1.0 - p_over, 4),
        "edge": round(edge, 4),
        "recommendation": recommendation,
        "model_version": artifact.model_version,
    }
