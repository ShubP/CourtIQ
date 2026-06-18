"""The Odds API client — live MLB player-prop lines.

Player props are per-event on the v4 API, so we list events then fetch each
event's markets. Free tier works; player-prop markets cost credits per event,
so markets are configurable. Docs: https://the-odds-api.com/
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from datetime import datetime, timedelta, timezone

from ..config import (
    LOOKAHEAD_DAYS,
    ODDS_API_BASE,
    ODDS_API_BOOKMAKERS,
    ODDS_API_KEY,
    ODDS_API_REGION,
    ODDS_FETCH_MARKETS,
)

log = logging.getLogger(__name__)
_TIMEOUT = 20
SPORT_KEY = "baseball_mlb"


@dataclass
class OddsEvent:
    event_id: str
    commence_time: str
    home_team: str
    away_team: str


@dataclass
class RawProp:
    event_id: str
    player_name: str
    market: str
    line: float
    over_odds: int | None
    under_odds: int | None
    bookmaker: str
    home_team: str
    away_team: str
    commence_time: str


def has_key() -> bool:
    return bool(ODDS_API_KEY)


def _log_quota(resp: requests.Response) -> None:
    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    if remaining is not None:
        log.info("Odds API quota: %s remaining, %s used", remaining, used)


def get_events() -> list[OddsEvent]:
    """Upcoming MLB events. Cheap (does not consume the props quota)."""
    if not has_key():
        return []
    resp = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT_KEY}/events",
        params={"apiKey": ODDS_API_KEY},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    _log_quota(resp)
    return [
        OddsEvent(
            event_id=e["id"],
            commence_time=e.get("commence_time", ""),
            home_team=e.get("home_team", ""),
            away_team=e.get("away_team", ""),
        )
        for e in resp.json()
    ]


def get_event_props(event: OddsEvent, markets: str = ODDS_FETCH_MARKETS) -> list[RawProp]:
    """Fetch player-prop lines for one event across the configured markets."""
    if not has_key():
        return []
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_API_REGION,
        "markets": markets,
        "oddsFormat": "american",
    }
    if ODDS_API_BOOKMAKERS:
        params["bookmakers"] = ODDS_API_BOOKMAKERS

    resp = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT_KEY}/events/{event.event_id}/odds",
        params=params,
        timeout=_TIMEOUT,
    )
    if resp.status_code == 422:
        # Event has no player-prop markets available yet.
        return []
    resp.raise_for_status()
    _log_quota(resp)
    data = resp.json()

    # For each (player, market) keep one representative line. We pick the first
    # bookmaker we see; a later enhancement could shop for the best price/line.
    seen: dict[tuple[str, str], RawProp] = {}
    for bm in data.get("bookmakers", []):
        bm_key = bm.get("key", "")
        for market in bm.get("markets", []):
            mkey = market.get("key", "")
            # Collect Over/Under for each player within this market.
            by_player: dict[str, dict] = {}
            for oc in market.get("outcomes", []):
                player = oc.get("description") or oc.get("name", "")
                side = (oc.get("name") or "").lower()
                entry = by_player.setdefault(player, {"point": oc.get("point")})
                entry["point"] = oc.get("point", entry.get("point"))
                if side == "over":
                    entry["over"] = oc.get("price")
                elif side == "under":
                    entry["under"] = oc.get("price")
            for player, e in by_player.items():
                if e.get("point") is None:
                    continue
                key = (player, mkey)
                if key in seen:
                    continue
                seen[key] = RawProp(
                    event_id=event.event_id,
                    player_name=player,
                    market=mkey,
                    line=float(e["point"]),
                    over_odds=e.get("over"),
                    under_odds=e.get("under"),
                    bookmaker=bm_key,
                    home_team=data.get("home_team", event.home_team),
                    away_team=data.get("away_team", event.away_team),
                    commence_time=data.get("commence_time", event.commence_time),
                )
    return list(seen.values())


def _within_window(commence_time: str, days: int) -> bool:
    """True if the event starts within the next `days` days (UTC)."""
    if not commence_time:
        return True
    try:
        ct = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    except ValueError:
        return True
    cutoff = datetime.now(timezone.utc) + timedelta(days=days)
    return ct.date() <= cutoff.date()


def get_all_upcoming_props(
    markets: str = ODDS_FETCH_MARKETS, days: int = LOOKAHEAD_DAYS
) -> list[RawProp]:
    """Fetch props for events within the next `days` days (credit control).

    Listing events is free; each event's player-prop call costs
    (#markets) credits, so we only spend on near-term games.
    """
    props: list[RawProp] = []
    events = [e for e in get_events() if _within_window(e.commence_time, days)]
    log.info("Fetching props for %d upcoming events (markets=%s)", len(events), markets)
    for event in events:
        try:
            props.extend(get_event_props(event, markets=markets))
        except requests.HTTPError as exc:  # keep going if one event fails
            log.warning("Failed props for event %s: %s", event.event_id, exc)
    return props
