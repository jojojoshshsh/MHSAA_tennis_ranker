# graph_engine.py
from collections import defaultdict
from datetime import datetime
from statistics import mean
import logging

from config import MIN_MATCHES, TARGET_GENDER
from trueskill_engine import compute_trueskill

# ============================================================
# TUNABLE WEIGHTS
# ============================================================
#
# Each weight is a plain named constant.
#
# Derivation of original weights (for reference):
#   original denominator : 28.25
#   REACH         = (10.00 + 15)     / 28.25  ≈ 0.8850
#   QUALITY_WINS  = (3.00 + 3.5 + 5) / 28.25  ≈ 0.4071
#   TS_MU         = (3+1+3+2+1+0.4+1+1)/28.25 ≈ 0.4301
#   SOS           = (3.20+2.25+0.10+4.5)/28.25 ≈ 0.3575
#   LOCAL_TS_MU   = (0.30+1+0.5+1+4.25)/28.25  ≈ 0.2513
#   LOCAL_SOS     = 0.2 + 0.1 + 0.1             = 0.4000
#   LOCAL_REACH   = 1.75
#   H2H           = 0.005  (bonus applied after base score is fixed)

TGRS_REACH_WEIGHT        = (10.00 + 15)             / 28.25   # global win-graph reachability
TGRS_QUALITY_WINS_WEIGHT = (3.00 + 3.5 + 5)         / 28.25   # avg TS-mu of top-3 victims (global)
TGRS_TS_MU_WEIGHT        = (3.00+1+3+2+1+0.4+1+1)   / 28.25   # global TrueSkill mean
TGRS_SOS_WEIGHT          = (3.20+2.25+0.10+4.5)      / 28.25   # global strength-of-schedule
TGRS_LOCAL_TS_MU_WEIGHT  = (0.30+1+0.5+1+4.25)       / 28.25   # local-bucket TrueSkill mean
TGRS_LOCAL_SOS_WEIGHT    = 0.2 + 0.1 + 0.1                     # local strength-of-schedule
TGRS_LOCAL_REACH_WEIGHT  = 1.75                                 # local win-graph reachability



# ============================================================
# UTILITIES
# ============================================================

def _parse_dt(raw):
    if not raw:
        return datetime.min
    try:
        return datetime.fromisoformat(
            str(raw).replace("Z", "+00:00")
        ).replace(tzinfo=None)
    except Exception:
        return datetime.min


def _bare_entity_from_match(match, category, side):
    """
    Return a bare entity key with NO flight or division scope.
    Used for global structures (TrueSkill, win-graph, SOS) so that
    cross-bucket matches actually connect players to each other.

    Singles  → player_id_str
    Doubles  → (id_a, id_b)  sorted tuple
    """
    ids = match.get(f"{side}_player_ids") or []

    if category == "singles":
        if len(ids) != 1:
            return None
        return str(ids[0])

    if len(ids) != 2:
        return None
    return tuple(sorted(str(x) for x in ids))


def _local_entity_from_match(match, category, side, flight, division):
    """
    Return a (flight, division)-scoped entity key.
    Used for local/bucket structures so the same player in different
    flights or divisions is treated as a separate entity per bucket.

    Singles  → (player_id_str, flight_str, division_str)
    Doubles  → (id_a, id_b, flight_str, division_str)
    """
    ids = match.get(f"{side}_player_ids") or []
    flight_str   = str(flight)
    division_str = str(division)

    if category == "singles":
        if len(ids) != 1:
            return None
        return (str(ids[0]), flight_str, division_str)

    if len(ids) != 2:
        return None
    sorted_ids = tuple(sorted(str(x) for x in ids))
    return sorted_ids + (flight_str, division_str)


def _pair_key(a, b):
    return tuple(sorted((repr(a), repr(b))))


def _parse_score(score):
    games_for = games_against = sets_for = sets_against = 0
    for token in str(score or "").split():
        try:
            w, l = map(int, token.split("-"))
        except Exception:
            continue
        games_for += w
        games_against += l
        if w > l:
            sets_for += 1
        else:
            sets_against += 1
    return games_for, games_against, sets_for, sets_against


def _local_entity_label(entity):
    """Human-readable label; strips trailing flight+division tags."""
    if isinstance(entity, tuple):
        # Singles: ("pid", flight, div)      → pid
        # Doubles: (id_a, id_b, flight, div) → "id_a / id_b"
        if len(entity) == 3:
            return str(entity[0])
        return " / ".join(entity[:-2])
    return str(entity)


def _bare_player_id(local_entity):
    """
    Extract bare player/pair ID from a local (flight+division-scoped) key,
    for use in player_lookup / pair_lookup.
    """
    if isinstance(local_entity, tuple):
        if len(local_entity) == 3:      # singles (pid, flight, div)
            return local_entity[0]
        return local_entity[:-2]        # doubles (id_a, id_b, flight, div)
    return local_entity


def _bare_from_local(local_entity):
    """
    Derive the bare global entity key from a local scoped key.
    Inverse of the scoping applied by _local_entity_from_match.

    Singles local (pid, flight, div)      → pid  (str)
    Doubles local (id_a, id_b, flight, div) → (id_a, id_b)  (tuple)
    """
    return _bare_player_id(local_entity)


def _reachability_score(start, graph):
    """Count unique nodes reachable from `start` via directed win-edges."""
    seen = set()
    stack = list(graph.get(start, []))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        for nxt in graph.get(node, []):
            if nxt not in seen:
                stack.append(nxt)
    return len(seen)


def _normalize_division(raw):
    s = str(raw or "").strip().lower()
    if s in {"1", "div 1", "div1", "division 1", "division1"}:
        return "1"
    if s in {"2", "div 2", "div2", "division 2", "division2"}:
        return "2"
    if s in {"3", "div 3", "div3", "division 3", "division3"}:
        return "3"
    if s in {"4", "4/other", "4 other", "4other", "other", "division 4", "division4"}:
        return "4_other"
    return "4_other"


def _match_division(match, category, player_lookup=None, pair_lookup=None):
    player_lookup = player_lookup or {}
    pair_lookup = pair_lookup or {}

    if category == "singles":
        for side in ("winner", "loser"):
            ids = match.get(f"{side}_player_ids") or []
            if len(ids) == 1:
                pid = str(ids[0])
                meta = player_lookup.get(pid, {})
                return _normalize_division(meta.get("division", "4_other"))
        return "4_other"

    if category == "doubles":
        for side in ("winner", "loser"):
            ids = match.get(f"{side}_player_ids") or []
            if len(ids) == 2:
                key = tuple(sorted(str(x) for x in ids))
                meta = pair_lookup.get(key, {})
                return _normalize_division(meta.get("division", "4_other"))
        return "4_other"

    return "4_other"


def _safe_mean(values, default=0.0):
    vals = list(values)
    if not vals:
        return default
    return mean(vals)


def _top_n_average(values, n=5, default=0.0):
    vals = sorted(values, reverse=True)[:n]
    if not vals:
        return default
    return mean(vals)


def _safe_slug(value):
    import re
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


class _SlugRegistry:
    """
    Tracks (original_value → slug) mappings and raises if two distinct
    values collide onto the same slug within the same namespace.
    """
    def __init__(self, namespace: str = ""):
        self._namespace = namespace
        self._seen: dict[str, str] = {}   # slug → original value

    def register(self, value: str) -> str:
        slug = _safe_slug(value)
        original = str(value or "").strip()
        if slug in self._seen and self._seen[slug] != original:
            raise ValueError(
                f"Slug collision in {self._namespace!r}: "
                f"{original!r} and {self._seen[slug]!r} both map to {slug!r}. "
                "Rename one value or extend _safe_slug to disambiguate."
            )
        self._seen[slug] = original
        return slug


# ============================================================
# BUILD POOLS
# ============================================================

def _build_pool(matches, category, flight, division):
    """
    Returns
    -------
    graph    : {local_entity: set of local_entities beaten}
    stats    : raw per-local-entity statistics for this bucket
    ts_pairs : list of (local_winner, local_loser) in chronological order

    All keys are local (flight+division)-scoped via _local_entity_from_match.
    """
    matches = sorted(
        matches,
        key=lambda m: _parse_dt(m.get("match_updated_at"))
    )

    latest_by_pair = {}
    ts_pairs = []

    stats = defaultdict(lambda: {
        "raw_wins":      0,
        "raw_losses":    0,
        "raw_matches":   0,
        "raw_last_date": datetime.min,
        "wins":          0,
        "losses":        0,
        "game_diff":     0,
        "set_diff":      0,
        "opponents":     set(),
        "vs":            {},
    })

    for m in matches:
        winner = _local_entity_from_match(m, category, "winner", flight, division)
        loser  = _local_entity_from_match(m, category, "loser",  flight, division)
        if winner is None or loser is None:
            continue

        dt = _parse_dt(m.get("match_updated_at"))
        wg, lg, ws, ls = _parse_score(m.get("set_score", ""))

        stats[winner]["raw_wins"] += 1
        stats[winner]["raw_matches"] += 1
        stats[winner]["raw_last_date"] = max(stats[winner]["raw_last_date"], dt)

        stats[loser]["raw_losses"] += 1
        stats[loser]["raw_matches"] += 1
        stats[loser]["raw_last_date"] = max(stats[loser]["raw_last_date"], dt)

        stats[winner]["game_diff"] += (wg - lg)
        stats[loser]["game_diff"] -= (wg - lg)
        stats[winner]["set_diff"] += (ws - ls)
        stats[loser]["set_diff"] -= (ws - ls)

        ts_pairs.append((winner, loser))

        pair = _pair_key(winner, loser)
        latest_by_pair[pair] = {"winner": winner, "loser": loser, "date": dt}

    graph = defaultdict(set)
    for rec in latest_by_pair.values():
        w, l = rec["winner"], rec["loser"]
        graph[w].add(l)
        stats[w]["wins"] += 1
        stats[l]["losses"] += 1
        stats[w]["opponents"].add(l)
        stats[l]["opponents"].add(w)
        stats[w]["vs"][l] = rec
        stats[l]["vs"][w] = rec

    for entity in stats:
        graph.setdefault(entity, set())

    return graph, stats, ts_pairs


def build_overall_stats(matches, category):
    """
    Build season-wide stats for the displayed W/L columns.
    Keyed by local (flight+division)-scoped entity key to match the keys
    used in pool stats and in create_rankings.
    """
    stats = defaultdict(lambda: {
        "raw_wins":      0,
        "raw_losses":    0,
        "raw_matches":   0,
        "raw_last_date": datetime.min,
    })

    matches = sorted(
        matches,
        key=lambda m: _parse_dt(m.get("match_updated_at"))
    )

    for m in matches:
        flight   = str(m.get("flight") or "?")
        division = str(m.get("_resolved_division") or "?")
        winner   = _local_entity_from_match(m, category, "winner", flight, division)
        loser    = _local_entity_from_match(m, category, "loser",  flight, division)
        if winner is None or loser is None:
            continue

        dt = _parse_dt(m.get("match_updated_at"))

        stats[winner]["raw_wins"] += 1
        stats[winner]["raw_matches"] += 1
        stats[winner]["raw_last_date"] = max(stats[winner]["raw_last_date"], dt)

        stats[loser]["raw_losses"] += 1
        stats[loser]["raw_matches"] += 1
        stats[loser]["raw_last_date"] = max(stats[loser]["raw_last_date"], dt)

    return dict(stats)


# ============================================================
# MAIN ENGINE
# ============================================================

def build_graph_pools(matches, player_lookup=None, pair_lookup=None):
    """
    Build pools split by: gender -> category -> division -> flight.

    LOCAL structures (graph, stats, ts_pairs) use (flight, division)-scoped
    entity keys so the same player in different buckets is a distinct node.

    GLOBAL structures (global_ts_pairs, global_ts_ratings, global_graph,
    global_opponents_by_entity) use BARE entity keys (no flight/division)
    so that cross-bucket matches genuinely connect the same player across
    flights and divisions — making global SOS and global ts_mu meaningfully
    different from their local counterparts.
    """
    player_lookup = player_lookup or {}
    pair_lookup   = pair_lookup   or {}

    grouped = defaultdict(list)

    # (gender, category) → [(dt, bare_winner, bare_loser)]
    global_ts_rows = defaultdict(list)

    for match in matches:
        gender   = match.get("gender")
        category = match.get("match_type", "").lower().strip()
        flight   = str(match.get("flight") or "?")

        if gender in ("Boys", "Girls") and category in ("singles", "doubles"):
            division = _match_division(match, category, player_lookup, pair_lookup)
            match["_resolved_division"] = division
            grouped[(gender, category, division, flight)].append(match)

            # Global pairs use BARE keys so cross-bucket matches connect.
            bare_winner = _bare_entity_from_match(match, category, "winner")
            bare_loser  = _bare_entity_from_match(match, category, "loser")
            if bare_winner is not None and bare_loser is not None:
                dt = _parse_dt(match.get("match_updated_at"))
                global_ts_rows[(gender, category)].append((dt, bare_winner, bare_loser))

    pools = {
        g: {
            "singles": defaultdict(dict),
            "doubles": defaultdict(dict),
        }
        for g in ("Boys", "Girls")
    }

    for (gender, category, division, flight), group in grouped.items():
        graph, stats, ts_pairs = _build_pool(group, category, flight, division)
        pools[gender][category][division][flight] = {
            "graph":    graph,
            "stats":    stats,
            "ts_pairs": ts_pairs,
        }

    global_matches_by_key = defaultdict(list)
    for match in matches:
        gender   = match.get("gender")
        category = match.get("match_type", "").lower().strip()
        if gender in ("Boys", "Girls") and category in ("singles", "doubles"):
            global_matches_by_key[(gender, category)].append(match)

    for (gender, category), rows in global_ts_rows.items():
        rows.sort(key=lambda x: x[0])
        global_ts_pairs = [(w, l) for _, w, l in rows]

        # Ratings and opponent lists keyed by BARE id.
        global_ts_ratings = compute_trueskill(global_ts_pairs)

        global_opponents_by_entity = defaultdict(list)
        for w, l in global_ts_pairs:
            global_opponents_by_entity[w].append(l)
            global_opponents_by_entity[l].append(w)

        # Global win-graph also keyed by BARE id.
        all_global_matches = sorted(
            global_matches_by_key[(gender, category)],
            key=lambda m: _parse_dt(m.get("match_updated_at"))
        )
        global_latest_by_pair = {}
        for m in all_global_matches:
            bw = _bare_entity_from_match(m, category, "winner")
            bl = _bare_entity_from_match(m, category, "loser")
            if bw is None or bl is None:
                continue
            pk = _pair_key(bw, bl)
            global_latest_by_pair[pk] = {"winner": bw, "loser": bl}

        global_graph = defaultdict(set)
        for rec in global_latest_by_pair.values():
            global_graph[rec["winner"]].add(rec["loser"])

        pools[gender][category]["_global_ts_pairs"]            = global_ts_pairs
        pools[gender][category]["_global_ts_ratings"]          = global_ts_ratings
        pools[gender][category]["_global_opponents_by_entity"] = dict(global_opponents_by_entity)
        pools[gender][category]["_global_graph"]               = global_graph

    return pools


def create_rankings(
    pools,
    category,
    overall_stats=None,
    player_lookup=None,
    pair_lookup=None,
):
    """
    Rank by TGRS inside each gender/division/flight bucket.

    Global metrics (ts_mu, sos, reachability, quality_wins) are computed
    using BARE entity keys so they reflect a player's full cross-bucket
    record.

    Local metrics (local_ts_mu, local_sos, local_reachability) are
    computed using local (flight+division)-scoped keys and reflect only
    performance within this specific bucket.
    """
    player_lookup = player_lookup or {}
    pair_lookup   = pair_lookup   or {}
    overall_stats = overall_stats or {}

    rows = []

    genders = (
        (TARGET_GENDER,)
        if TARGET_GENDER
        else ("Boys", "Girls")
    )

    for gender in genders:
        category_pools = pools[gender][category]

        global_ts_ratings          = category_pools.get("_global_ts_ratings", {})
        global_opponents_by_entity = category_pools.get("_global_opponents_by_entity", {})
        global_graph               = category_pools.get("_global_graph", defaultdict(set))

        for division, division_pools in category_pools.items():
            if str(division).startswith("_"):
                continue
            for flight, pool in division_pools.items():
                graph    = pool["graph"]    # local-scoped keys
                stats    = pool["stats"]    # local-scoped keys
                ts_pairs = pool["ts_pairs"] # local-scoped keys

                eligible = {
                    entity
                    for entity, info in stats.items()
                    if info["raw_matches"] >= MIN_MATCHES
                }
                if not eligible:
                    continue

                graph = {
                    node: {x for x in neighbors if x in eligible}
                    for node, neighbors in graph.items()
                    if node in eligible
                }
                for entity in eligible:
                    graph.setdefault(entity, set())

                if not graph:
                    continue

                # Global reachability: use bare keys, look up via bare id.
                global_reach = {
                    entity: _reachability_score(_bare_from_local(entity), global_graph)
                    for entity in eligible
                }

                # Local reachability: use local graph directly.
                local_reach = {
                    entity: _reachability_score(entity, graph)
                    for entity in graph
                }

                eligible = {e for e in eligible if local_reach.get(e, 0) > 0}
                if not eligible:
                    continue

                # Local TrueSkill trained only on this bucket's matches.
                local_ts_ratings = compute_trueskill(ts_pairs)

                def ts_mu_val(entity):
                    r = local_ts_ratings.get(entity)
                    return r.mu if r is not None else 0.0

                def ts_cons(entity):
                    r = local_ts_ratings.get(entity)
                    return r.conservative if r is not None else 0.0

                eligible_ts_cons = [ts_cons(e) for e in eligible]
                field_avg_ts = _safe_mean(eligible_ts_cons, default=0.0)

                # Local opponent list: built from local ts_pairs keys.
                local_opponents_by_entity = defaultdict(list)
                for w, l in ts_pairs:
                    local_opponents_by_entity[w].append(l)
                    local_opponents_by_entity[l].append(w)

                sos          = {}
                local_sos    = {}
                quality_wins = {}
                tgrs_score   = {}

                for entity in eligible:
                    bare = _bare_from_local(entity)

                    # --- Global SOS: bare opponents → global ratings ---
                    global_opp_strengths = [
                        global_ts_ratings[opp].conservative
                        for opp in global_opponents_by_entity.get(bare, [])
                        if opp in global_ts_ratings
                    ]
                    entity_global_sos = _safe_mean(global_opp_strengths, default=field_avg_ts)
                    sos[entity] = entity_global_sos

                    # --- Local SOS: local opponents → local ratings ---
                    local_opp_strengths = [
                        local_ts_ratings[opp].conservative
                        for opp in local_opponents_by_entity.get(entity, [])
                        if opp in local_ts_ratings
                    ]
                    entity_local_sos = _safe_mean(local_opp_strengths, default=field_avg_ts)
                    local_sos[entity] = entity_local_sos

                    # --- Quality wins: bare win-graph → global ratings ---
                    beaten_ratings = [
                        global_ts_ratings[opp].mu
                        for opp in global_graph.get(bare, set())
                        if opp in global_ts_ratings
                    ]
                    entity_quality_wins = _top_n_average(beaten_ratings, n=3, default=0.0)
                    quality_wins[entity] = entity_quality_wins

                    # --- Global ts_mu via bare key ---
                    g_ts_r = global_ts_ratings.get(bare)
                    global_ts_mu_val = g_ts_r.mu if g_ts_r is not None else 0.0

                    tgrs_score[entity] = (
                        TGRS_TS_MU_WEIGHT         * global_ts_mu_val
                        + TGRS_SOS_WEIGHT          * entity_global_sos
                        + TGRS_REACH_WEIGHT        * global_reach.get(entity, 0)
                        + TGRS_QUALITY_WINS_WEIGHT * entity_quality_wins
                        + TGRS_LOCAL_SOS_WEIGHT    * entity_local_sos
                        + TGRS_LOCAL_TS_MU_WEIGHT  * ts_mu_val(entity)
                        + TGRS_LOCAL_REACH_WEIGHT  * local_reach.get(entity, 0)
                    )

                ordered = sorted(
                    eligible,
                    key=lambda x: (
                        -tgrs_score.get(x, -9999.0),
                        -global_reach.get(x, 0),
                        -local_reach.get(x, 0),
                        -quality_wins.get(x, 0.0),
                        -ts_mu_val(x),
                        -ts_cons(x),
                        _local_entity_label(x),
                    )
                )

                rank = 1
                i = 0
                n = len(ordered)

                while i < n:
                    current_score = tgrs_score[ordered[i]]

                    tie_group = []
                    j = i
                    while j < n and abs(tgrs_score[ordered[j]] - current_score) < 1e-12:
                        tie_group.append(ordered[j])
                        j += 1

                    for entity in tie_group:
                        bucket_info = stats[entity]
                        global_info = overall_stats.get(entity, bucket_info)

                        bare        = _bare_from_local(entity)
                        g_ts_r      = global_ts_ratings.get(bare)
                        local_ts_r  = local_ts_ratings.get(entity)
                        score       = tgrs_score.get(entity, 0.0)

                        row = {
                            "rank":               rank,
                            "gender":             gender,
                            "division":           division,
                            "flight":             flight,
                            "rating":             round(score, 4),
                            "TGRS":               round(score, 4),
                            "reachability":       global_reach.get(entity, 0),
                            "local_reachability": local_reach.get(entity, 0),
                            "sos":                round(sos.get(entity, field_avg_ts), 4),
                            "local_sos":          round(local_sos.get(entity, field_avg_ts), 4),
                            "quality_wins":       round(quality_wins.get(entity, 0.0), 4),
                            "ts_mu":              round(g_ts_r.mu, 4)           if g_ts_r      else None,
                            "ts_sigma":           round(g_ts_r.sigma, 4)        if g_ts_r      else None,
                            "ts_rating":          round(g_ts_r.conservative, 4) if g_ts_r      else None,
                            "local_ts_mu":        round(local_ts_r.mu, 4)       if local_ts_r  else None,
                            "matches_played":     global_info["raw_matches"],
                            "wins":               global_info["raw_wins"],
                            "losses":             global_info["raw_losses"],
                            "last_match_date": (
                                global_info["raw_last_date"].strftime("%Y-%m-%d")
                                if global_info["raw_last_date"] != datetime.min
                                else ""
                            ),
                        }

                        if category == "singles":
                            meta = player_lookup.get(bare, {})
                            row.update({
                                "name":     meta.get("name", bare),
                                "school":   meta.get("school", ""),
                                "division": division,
                            })
                        else:
                            meta = pair_lookup.get(bare, {})
                            row.update({
                                "pair_name": meta.get("pair_name", " / ".join(bare)),
                                "school":    meta.get("school", ""),
                                "division":  division,
                            })

                        rows.append(row)

                    rank += len(tie_group)
                    i = j

    return rows
