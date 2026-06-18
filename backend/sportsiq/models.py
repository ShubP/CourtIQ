"""SQLAlchemy ORM models — the SportsIQ schema.

Works on both SQLite (local dev) and Postgres (Neon/RDS in prod) via DATABASE_URL.
No auth/users: the app is open per the product spec.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    team_id: Mapped[int] = mapped_column(Integer, primary_key=True)  # MLB team id
    abbrev: Mapped[str] = mapped_column(String(10))
    name: Mapped[str] = mapped_column(String(100))


class Player(Base):
    __tablename__ = "players"

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)  # MLB person id
    full_name: Mapped[str] = mapped_column(String(120))
    position: Mapped[str | None] = mapped_column(String(10), nullable=True)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.team_id"), nullable=True)
    bat_side: Mapped[str | None] = mapped_column(String(2), nullable=True)
    pitch_hand: Mapped[str | None] = mapped_column(String(2), nullable=True)
    headshot_url: Mapped[str | None] = mapped_column(String(255), nullable=True)


class Game(Base):
    __tablename__ = "games"

    game_pk: Mapped[int] = mapped_column(Integer, primary_key=True)  # MLB gamePk
    game_date: Mapped[str] = mapped_column(String(10), index=True)   # YYYY-MM-DD
    game_datetime: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    home_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.team_id"), nullable=True)
    away_team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.team_id"), nullable=True)
    home_probable_pitcher_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_probable_pitcher_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str | None] = mapped_column(String(120), nullable=True)


class PlayerGameLog(Base):
    """One row per player per game played (historical box-score line).

    Columns cover both hitting and pitching; irrelevant ones stay NULL.
    """
    __tablename__ = "player_game_logs"
    __table_args__ = (
        UniqueConstraint("player_id", "game_pk", "stat_group", name="uq_log_player_game_group"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.player_id"), index=True)
    game_pk: Mapped[int | None] = mapped_column(Integer, nullable=True)
    game_date: Mapped[str] = mapped_column(String(10), index=True)
    opponent_team_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_home: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    stat_group: Mapped[str] = mapped_column(String(10))  # 'hitting' | 'pitching'

    # Hitting
    at_bats: Mapped[float | None] = mapped_column(Float, nullable=True)
    hits: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_bases: Mapped[float | None] = mapped_column(Float, nullable=True)
    home_runs: Mapped[float | None] = mapped_column(Float, nullable=True)
    rbis: Mapped[float | None] = mapped_column(Float, nullable=True)
    runs: Mapped[float | None] = mapped_column(Float, nullable=True)
    hits_runs_rbis: Mapped[float | None] = mapped_column(Float, nullable=True)  # H+R+RBI combo
    walks: Mapped[float | None] = mapped_column(Float, nullable=True)
    strikeouts_batting: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Pitching
    innings_outs: Mapped[float | None] = mapped_column(Float, nullable=True)  # outs recorded
    strikeouts: Mapped[float | None] = mapped_column(Float, nullable=True)    # Ks thrown
    earned_runs: Mapped[float | None] = mapped_column(Float, nullable=True)
    hits_allowed: Mapped[float | None] = mapped_column(Float, nullable=True)
    walks_allowed: Mapped[float | None] = mapped_column(Float, nullable=True)
    batters_faced: Mapped[float | None] = mapped_column(Float, nullable=True)


class Prop(Base):
    """A live sportsbook player-prop line for an upcoming game."""
    __tablename__ = "props"
    __table_args__ = (
        UniqueConstraint("game_pk", "player_id", "market", "bookmaker", name="uq_prop"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    player_id: Mapped[int] = mapped_column(Integer, index=True)
    player_name: Mapped[str] = mapped_column(String(120))
    market: Mapped[str] = mapped_column(String(40), index=True)
    line: Mapped[float] = mapped_column(Float)
    over_odds: Mapped[int | None] = mapped_column(Integer, nullable=True)   # American odds
    under_odds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bookmaker: Mapped[str] = mapped_column(String(60))
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    """Model output + edge for one player/market/game."""
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("game_pk", "player_id", "market", name="uq_prediction"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    player_id: Mapped[int] = mapped_column(Integer, index=True)
    player_name: Mapped[str] = mapped_column(String(120))
    team_abbrev: Mapped[str | None] = mapped_column(String(10), nullable=True)
    opponent_abbrev: Mapped[str | None] = mapped_column(String(10), nullable=True)
    market: Mapped[str] = mapped_column(String(40), index=True)
    line: Mapped[float] = mapped_column(Float)
    predicted_value: Mapped[float] = mapped_column(Float)
    prob_over: Mapped[float] = mapped_column(Float)
    prob_under: Mapped[float] = mapped_column(Float)
    over_odds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    under_odds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Best side's expected value per $1 stake; the headline "edge".
    edge: Mapped[float] = mapped_column(Float, index=True)
    recommendation: Mapped[str] = mapped_column(String(8))  # 'Over' | 'Under' | 'Pass'
    bookmaker: Mapped[str | None] = mapped_column(String(60), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    game_date: Mapped[str | None] = mapped_column(String(10), index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ModelMetric(Base):
    """Backtest metrics per market — powers the Model Performance dashboard."""
    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market: Mapped[str] = mapped_column(String(40), index=True)
    market_label: Mapped[str | None] = mapped_column(String(60), nullable=True)
    n_train: Mapped[int] = mapped_column(Integer, default=0)
    n_test: Mapped[int] = mapped_column(Integer, default=0)
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    skill_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    brier_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    brier_calibrated: Mapped[float | None] = mapped_column(Float, nullable=True)
    hit_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    roi_proxy: Mapped[float | None] = mapped_column(Float, nullable=True)
    dispersion_r: Mapped[float | None] = mapped_column(Float, nullable=True)
    blend_weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibration_json: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(60), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ran_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    games: Mapped[int] = mapped_column(Integer, default=0)
    props: Mapped[int] = mapped_column(Integer, default=0)
    predictions: Mapped[int] = mapped_column(Integer, default=0)
    note: Mapped[str | None] = mapped_column(String(255), nullable=True)
