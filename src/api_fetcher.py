import asyncio
import logging
import os

import aiohttp

from config import IS_NOT_VARSITY, YEAR

_TIMEOUT = aiohttp.ClientTimeout(total=20)


def _get_headers():
    token = os.environ.get("TENNIS_TOKEN", "undefined")
    return {
        "Accept":       "*/*",
        "Content-Type": "application/json",
        "Origin":       "https://tennisreporting.com",
        "Referer":      "https://tennisreporting.com/",
        "User-Agent":   "Mozilla/5.0",
        "token":        token,
    }


async def fetch_school_report(session, school_id, gender_id=1, retries=3, backoff=2.0):
    url = (f"https://api.tennisreporting.com/report/school/{school_id}"
           f"?year={YEAR}&genderId={gender_id}&isNotVarsity={IS_NOT_VARSITY}")
    return await _get(session, url, f"school {school_id}", retries, backoff)


async def fetch_event(session, event_id, retries=3, backoff=2.0):
    url = f"https://api.tennisreporting.com/event/{event_id}"
    return await _get(session, url, f"event {event_id}", retries, backoff)


async def fetch_seed_list(
    session,
    event_id,
    division_id,
    host_id,
    match_type,
    flight,
    is_consolation=False,
    retries=3,
    backoff=2.0,
):
    url = f"https://api.tennisreporting.com/event/{event_id}/seed_list_by_params"
    payload = {
        "division": division_id,
        "host": host_id,
        "matchType": match_type,
        "flight": flight,
        "isConsolation": is_consolation,
    }
    label = f"seed_list e={event_id} h={host_id} {match_type}[{flight}]"
    return await _post(session, url, payload, label, retries, backoff)


async def fetch_bracket(session, event_id, host_id, division_id,
                        match_type, flight, is_consolation=False,
                        retries=3, backoff=2.0):
    url = f"https://api.tennisreporting.com/event/{event_id}/host/{host_id}/bracket/get"
    payload = {
        "division":      division_id,
        "host":          host_id,
        "matchType":     match_type,
        "flight":        flight,
        "isConsolation": is_consolation,
    }
    label = f"bracket e={event_id} h={host_id} {match_type}[{flight}]"
    return await _post(session, url, payload, label, retries, backoff)


async def _get(session, url, label, retries, backoff):
    for attempt in range(1, retries + 1):
        try:
            async with session.get(url, headers=_get_headers(), timeout=_TIMEOUT) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)
                logging.warning("%s: HTTP %s (attempt %d/%d)", label, resp.status, attempt, retries)
        except asyncio.TimeoutError:
            logging.warning("%s: timeout (attempt %d/%d)", label, attempt, retries)
        except Exception as exc:
            logging.error("%s: %s (attempt %d/%d)", label, exc, attempt, retries)
        if attempt < retries:
            await asyncio.sleep(backoff * attempt)
    logging.error("%s: giving up after %d attempts.", label, retries)
    return None


async def _post(session, url, payload, label, retries, backoff):
    for attempt in range(1, retries + 1):
        try:
            async with session.post(url, headers=_get_headers(),
                                    json=payload, timeout=_TIMEOUT) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)
                logging.warning("%s: HTTP %s (attempt %d/%d)", label, resp.status, attempt, retries)
        except asyncio.TimeoutError:
            logging.warning("%s: timeout (attempt %d/%d)", label, attempt, retries)
        except Exception as exc:
            logging.error("%s: %s (attempt %d/%d)", label, exc, attempt, retries)
        if attempt < retries:
            await asyncio.sleep(backoff * attempt)
    logging.error("%s: giving up after %d attempts.", label, retries)
    return None
