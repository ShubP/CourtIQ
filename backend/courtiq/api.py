"""FastAPI app serving the precomputed edge board to the frontend.

Read-only: the pipeline writes the board; the API just serves it. Run with:
    uvicorn courtiq.api:app --reload --port 8080
"""
from __future__ import annotations

import os

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import desc, select

import json

from .config import MARKETS
from .db import SessionLocal, init_db
from .models import (
    Game,
    ModelMetric,
    PipelineRun,
    Player,
    PlayerGameLog,
    Prediction,
    Team,
)

app = FastAPI(title="CourtIQ API", version="0.1.0")

# CORS: allow the Vite dev server and (optionally) a deployed frontend origin.
_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
if os.getenv("FRONTEND_ORIGIN"):
    _origins.append(os.getenv("FRONTEND_ORIGIN"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/markets")
def markets() -> list[dict]:
    return [{"key": k, "label": v["label"], "group": v["group"]} for k, v in MARKETS.items()]


@app.get("/meta")
def meta() -> dict:
    with SessionLocal() as s:
        run = s.execute(select(PipelineRun).order_by(desc(PipelineRun.id)).limit(1)).scalar_one_or_none()
        if not run:
            return {"last_run": None, "synthetic": True, "games": 0, "predictions": 0}
        return {
            "last_run": run.ran_at.isoformat() if run.ran_at else None,
            "synthetic": (run.note or "").startswith("synthetic"),
            "games": run.games,
            "props": run.props,
            "predictions": run.predictions,
            "note": run.note,
        }


def _prediction_to_dict(p: Prediction, headshot: str | None) -> dict:
    spec = MARKETS.get(p.market, {})
    return {
        "player_id": p.player_id,
        "player_name": p.player_name,
        "headshot_url": headshot,
        "team": p.team_abbrev,
        "opponent": p.opponent_abbrev,
        "market": p.market,
        "market_label": spec.get("label", p.market),
        "line": p.line,
        "predicted_value": p.predicted_value,
        "prob_over": p.prob_over,
        "prob_under": p.prob_under,
        "over_odds": p.over_odds,
        "under_odds": p.under_odds,
        "recommendation": p.recommendation,
        "edge": p.edge,
        "edge_pct": round(p.edge * 100, 1),
        "bookmaker": p.bookmaker,
        "game_pk": p.game_pk,
        "game_date": p.game_date,
    }


@app.get("/board")
def board(
    market: str | None = Query(None, description="Filter by market key"),
    date: str | None = Query(None, description="Filter by YYYY-MM-DD game date"),
    recommendation: str | None = Query(None, description="Over/Under/Pass"),
    min_edge: float = Query(-1.0, description="Minimum edge (EV per $1)"),
    limit: int = Query(200, ge=1, le=1000),
) -> list[dict]:
    """The ranked edge board — highest edge first."""
    with SessionLocal() as s:
        stmt = select(Prediction).where(Prediction.edge >= min_edge)
        if market:
            stmt = stmt.where(Prediction.market == market)
        if date:
            stmt = stmt.where(Prediction.game_date == date)
        if recommendation:
            stmt = stmt.where(Prediction.recommendation == recommendation)
        stmt = stmt.order_by(desc(Prediction.edge)).limit(limit)
        preds = s.execute(stmt).scalars().all()

        ids = {p.player_id for p in preds}
        heads = {}
        if ids:
            for pl in s.execute(select(Player).where(Player.player_id.in_(ids))).scalars():
                heads[pl.player_id] = pl.headshot_url
        return [_prediction_to_dict(p, heads.get(p.player_id)) for p in preds]


@app.get("/model-performance")
def model_performance() -> list[dict]:
    """Latest backtest metrics per market for the performance dashboard."""
    with SessionLocal() as s:
        rows = s.execute(select(ModelMetric).order_by(desc(ModelMetric.id))).scalars().all()
        latest: dict[str, ModelMetric] = {}
        for r in rows:
            latest.setdefault(r.market, r)  # first seen = newest (desc id)
        out = []
        for r in latest.values():
            out.append({
                "market": r.market,
                "market_label": r.market_label,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "mae": r.mae,
                "rmse": r.rmse,
                "baseline_mae": r.baseline_mae,
                "skill_pct": r.skill_pct,
                "brier_raw": r.brier_raw,
                "brier_calibrated": r.brier_calibrated,
                "brier_improvement_pct": (
                    round((r.brier_raw - r.brier_calibrated) / r.brier_raw * 100, 1)
                    if r.brier_raw else None
                ),
                "hit_rate": r.hit_rate,
                "roi_proxy": r.roi_proxy,
                "dispersion_r": r.dispersion_r,
                "blend_weight": r.blend_weight,
                "calibration": json.loads(r.calibration_json) if r.calibration_json else [],
                "model_version": r.model_version,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            })
        return out


@app.get("/games")
def games(date: str | None = None) -> list[dict]:
    with SessionLocal() as s:
        teams = {t.team_id: t for t in s.execute(select(Team)).scalars()}
        stmt = select(Game).order_by(Game.game_date)
        if date:
            stmt = stmt.where(Game.game_date == date)
        out = []
        for g in s.execute(stmt).scalars():
            home = teams.get(g.home_team_id)
            away = teams.get(g.away_team_id)
            out.append({
                "game_pk": g.game_pk,
                "game_date": g.game_date,
                "status": g.status,
                "home": home.abbrev if home else None,
                "away": away.abbrev if away else None,
                "home_name": home.name if home else None,
                "away_name": away.name if away else None,
                "venue": g.venue,
            })
        return out


@app.get("/player/{player_id}/history")
def player_history(player_id: int, stat: str = "hits", limit: int = 15) -> dict:
    """Recent per-game values for a stat — powers the player chart."""
    with SessionLocal() as s:
        player = s.get(Player, player_id)
        rows = s.execute(
            select(PlayerGameLog)
            .where(PlayerGameLog.player_id == player_id)
            .order_by(desc(PlayerGameLog.game_date))
            .limit(limit)
        ).scalars().all()
        teams = {t.team_id: t.abbrev for t in s.execute(select(Team)).scalars()}
        history = []
        for r in reversed(rows):
            val = getattr(r, stat, None)
            if val is None:
                continue
            history.append({
                "game_date": r.game_date,
                "opponent": teams.get(r.opponent_team_id, ""),
                "value": val,
            })
        avg = round(sum(h["value"] for h in history) / len(history), 2) if history else 0
        return {
            "player_id": player_id,
            "player_name": player.full_name if player else "",
            "stat": stat,
            "history": history,
            "average": avg,
        }
