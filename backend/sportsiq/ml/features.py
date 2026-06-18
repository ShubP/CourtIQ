"""Feature engineering for per-game player-prop prediction.

The core idea: predict a player's stat in their *next* game from leak-free
features computed only from prior games (recent form + volume + home/away +
a data-derived opponent factor).
"""
from __future__ import annotations

import pandas as pd

# Companion "volume" stats per target — context that drives the target.
VOLUME_STATS: dict[str, list[str]] = {
    # hitting
    "hits": ["at_bats"],
    "total_bases": ["at_bats", "hits"],
    "home_runs": ["at_bats"],
    "rbis": ["at_bats", "hits"],
    "runs": ["at_bats", "hits"],
    "hits_runs_rbis": ["at_bats", "hits"],
    # pitching
    "strikeouts": ["innings_outs", "batters_faced"],
}

_WINDOWS = [3, 5, 10]


def _logs_to_frame(rows: list) -> pd.DataFrame:
    """GameLogRow list -> tidy DataFrame sorted by date."""
    records = []
    for r in rows:
        rec = {
            "player_id": r.player_id,
            "game_date": r.game_date,
            "opponent_team_id": r.opponent_team_id,
            "is_home": 1 if r.is_home else 0,
            **r.stats,
        }
        records.append(rec)
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values("game_date").reset_index(drop=True)
    return df


def feature_columns(stat: str) -> list[str]:
    cols = ["is_home", "games_played", "opp_factor"]
    for w in _WINDOWS:
        cols.append(f"{stat}_avg_l{w}")
    cols.append(f"{stat}_avg_season")
    for vs in VOLUME_STATS.get(stat, []):
        for w in _WINDOWS:
            cols.append(f"{vs}_avg_l{w}")
        cols.append(f"{vs}_avg_season")
    return cols


def _rolling_prior(series: pd.Series, window: int) -> pd.Series:
    """Mean of the previous `window` values (shifted to avoid leakage)."""
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def _expanding_prior(series: pd.Series) -> pd.Series:
    return series.shift(1).expanding(min_periods=1).mean()


def build_player_training_rows(rows: list, stat: str) -> pd.DataFrame:
    """Build leak-free training rows (features from prior games) for one player."""
    df = _logs_to_frame(rows)
    if df.empty or stat not in df.columns:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    out["player_id"] = df["player_id"]
    out["game_date"] = df["game_date"]
    out["opponent_team_id"] = df["opponent_team_id"]
    out["is_home"] = df["is_home"]
    out["games_played"] = range(len(df))  # games before this one
    out["target"] = df[stat]

    for w in _WINDOWS:
        out[f"{stat}_avg_l{w}"] = _rolling_prior(df[stat], w)
    out[f"{stat}_avg_season"] = _expanding_prior(df[stat])

    for vs in VOLUME_STATS.get(stat, []):
        if vs not in df.columns:
            continue
        for w in _WINDOWS:
            out[f"{vs}_avg_l{w}"] = _rolling_prior(df[vs], w)
        out[f"{vs}_avg_season"] = _expanding_prior(df[vs])

    # Require at least 2 prior games of context.
    out = out[out["games_played"] >= 2].reset_index(drop=True)
    return out


def build_training_frame(logs_by_player: dict[int, list], stat: str) -> tuple[pd.DataFrame, dict]:
    """Assemble a training frame across many players + a data-derived opponent factor."""
    frames = [
        f for f in (build_player_training_rows(rows, stat) for rows in logs_by_player.values())
        if not f.empty
    ]
    if not frames:
        return pd.DataFrame(), {}
    data = pd.concat(frames, ignore_index=True)

    # Opponent factor: how a given opponent inflates/suppresses the target,
    # relative to the league mean. Coarse team-level prior.
    overall = data["target"].mean() or 1.0
    opp_means = data.groupby("opponent_team_id")["target"].mean()
    opp_factor = (opp_means / overall).to_dict()
    data["opp_factor"] = data["opponent_team_id"].map(opp_factor).fillna(1.0)

    data = data.fillna(0.0)
    return data, {"opp_factor": opp_factor, "league_mean": overall}


def build_prediction_features(
    rows: list,
    stat: str,
    is_home: int,
    opponent_team_id: int | None,
    opp_factor_map: dict,
) -> dict | None:
    """Build a single feature row for an upcoming game from all prior history."""
    df = _logs_to_frame(rows)
    if df.empty or stat not in df.columns or len(df) < 2:
        return None

    feat: dict[str, float] = {
        "is_home": float(is_home),
        "games_played": float(len(df)),
        "opp_factor": float(opp_factor_map.get(opponent_team_id, 1.0))
        if opponent_team_id is not None else 1.0,
    }
    for w in _WINDOWS:
        feat[f"{stat}_avg_l{w}"] = float(df[stat].tail(w).mean())
    feat[f"{stat}_avg_season"] = float(df[stat].mean())

    for vs in VOLUME_STATS.get(stat, []):
        if vs not in df.columns:
            continue
        for w in _WINDOWS:
            feat[f"{vs}_avg_l{w}"] = float(df[vs].tail(w).mean())
        feat[f"{vs}_avg_season"] = float(df[vs].mean())
    return feat
