# crawler.py
# BFS school crawler + event bracket fetcher.
#
# Event flow (updated):
#   1. Detect eventId in school meets
#   2. GET /event/{id}           — metadata (gender, flights, divisions/hosts)
#   3. GET /event/{id}/seed_list_by_params  ← NEW: player id → name/school map
#   4. POST bracket for each division/host/matchType/flight
#   5. parse_bracket_matches() uses the seed-list lookup to resolve player info

import asyncio
import logging
from collections import deque

import aiohttp

from api_fetcher import (fetch_bracket, fetch_event,
                          fetch_school_report, fetch_seed_list)
from config import MAX_SCHOOLS, TARGET_GENDER, TARGET_STATE
from match_parser import (build_event_player_lookup, extract_school_meta,
                           match_key, parse_bracket_matches,
                           parse_school_matches)

_REQUEST_DELAY   = 0.15
_MAX_CONNECTIONS = 16
_BRACKET_CHUNK   = 6


def _school_state(data: dict) -> str:
    return (data.get("school", {})
                .get("city", {})
                .get("state", {})
                .get("abbr") or "")


def _gender_name(gender_id) -> str | None:
    return {1: "Boys", 2: "Girls"}.get(gender_id)


def _gender_ok(target: str | None, gender: str | None) -> bool:
    return target is None or gender == target


# ── event bracket fetcher ─────────────────────────────────────────────────────

# crawler.py

async def _fetch_event_matches(session: aiohttp.ClientSession,
                               event_id, seen_keys: set) -> list[dict]:
    event_data = await fetch_event(session, event_id)
    if not event_data:
        return []

    gender_id    = event_data.get("genderId")
    event_gender = _gender_name(gender_id)
    event_date   = event_data.get("dateEventStart", "")
    n_singles    = int(event_data.get("flightSinglesNumber") or 4)
    n_doubles    = int(event_data.get("flightDoublesNumber") or 4)

    if not _gender_ok(TARGET_GENDER, event_gender):
        logging.debug("Event %s: gender=%s skipped (target=%s)",
                      event_id, event_gender, TARGET_GENDER)
        return []

    tasks = []
    for div in event_data.get("divisions", []):
        div_id = div.get("id")
        for host in div.get("hosts", []):
            host_id = host.get("id")
            for mt, n_fl in [("Singles", n_singles), ("Doubles", n_doubles)]:
                for fl in range(1, n_fl + 1):
                    tasks.append((div_id, host_id, mt, fl))

    logging.info("Event %s: %d bracket slices to fetch", event_id, len(tasks))

    all_matches: list[dict] = []

    for i in range(0, len(tasks), _BRACKET_CHUNK):
        chunk = tasks[i:i + _BRACKET_CHUNK]

        seed_tasks = [
            fetch_seed_list(session, event_id, div_id, host_id, mt, fl)
            for div_id, host_id, mt, fl in chunk
        ]
        bracket_tasks = [
            fetch_bracket(session, event_id, host_id, div_id, mt, fl)
            for div_id, host_id, mt, fl in chunk
        ]

        seed_results, bracket_results = await asyncio.gather(
            asyncio.gather(*seed_tasks),
            asyncio.gather(*bracket_tasks),
        )

        for (div_id, host_id, mt, fl), seed_list_raw, bdata in zip(chunk, seed_results, bracket_results):
            if not seed_list_raw or not bdata:
                continue

            player_lookup = build_event_player_lookup(seed_list_raw)
            if not player_lookup:
                logging.warning(
                    "Event %s %s flight %s: seed list returned no usable players",
                    event_id, mt, fl
                )
                continue

            matches = parse_bracket_matches(
                bdata, event_id, event_date, gender_id, mt, fl, player_lookup
            )
            for m in matches:
                key = match_key(m)
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_matches.append(m)

        await asyncio.sleep(_REQUEST_DELAY)

    logging.info("Event %s: %d new matches extracted.", event_id, len(all_matches))
    return all_matches


# ── main BFS crawl ────────────────────────────────────────────────────────────

async def crawl_school_matches(
    seed_id: int,
    max_schools: int | None = MAX_SCHOOLS,
) -> tuple[list[dict], dict]:
    """
    BFS from seed_id.  Returns (matches, school_meta).
    school_meta: {school_id -> {name, division_boys, division_girls}}
    """
    processed:      set   = set()
    queue:          deque = deque([seed_id])
    all_matches:    list  = []
    seen_keys:      set   = set()
    seen_event_ids: set   = set()
    school_meta:    dict  = {}
    skipped_oos:    int   = 0

    connector = aiohttp.TCPConnector(limit=_MAX_CONNECTIONS)
    async with aiohttp.ClientSession(connector=connector) as session:

        while queue:
            school_id = queue.popleft()
            if school_id in processed:
                continue
            if max_schools is not None and len(processed) >= max_schools:
                logging.info("max_schools=%d reached.", max_schools)
                break

            processed.add(school_id)
            data = await fetch_school_report(session, school_id, gender_id=1)
            if data is None:
                continue

            # ── state filter ───────────────────────────────────────────────
            if TARGET_STATE:
                state = _school_state(data)
                if state and state.upper() != TARGET_STATE.upper():
                    skipped_oos += 1
                    await asyncio.sleep(_REQUEST_DELAY)
                    continue

            meta = extract_school_meta(data)
            if meta["id"]:
                school_meta[meta["id"]] = meta

            logging.info("Processing school %s  [done=%d  queued=%d  matches=%d]",
                         school_id, len(processed), len(queue), len(all_matches))

            # ── regular-season matches ────────────────────────────────────
            for m in parse_school_matches(data, source_school_id=school_id):
                key = match_key(m)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_matches.append(m)
                for sid in (m["winner_school_id"], m["loser_school_id"]):
                    if sid and sid not in processed:
                        queue.append(sid)

            # ── event/tournament bracket fetching ─────────────────────────
            for meet in data.get("meets", []):
                eid = meet.get("eventId")
                if eid and eid not in seen_event_ids:
                    seen_event_ids.add(eid)
                    event_matches = await _fetch_event_matches(
                        session, eid, seen_keys
                    )
                    for m in event_matches:
                        all_matches.append(m)
                        for sid in (m["winner_school_id"], m["loser_school_id"]):
                            if sid and sid not in processed:
                                queue.append(sid)

            await asyncio.sleep(_REQUEST_DELAY)

    logging.info("Crawl complete — MI schools: %d  OOS skipped: %d  matches: %d",
                 len(processed) - skipped_oos, skipped_oos, len(all_matches))
    return all_matches, school_meta
