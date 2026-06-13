# trueskill_engine.py
#
# Lightweight TrueSkill implementation (Herbrich et al., 2007).
# No external dependencies — uses only `math` and `collections`.
#
# Public API
# ----------
#   compute_trueskill(match_pairs) -> {entity: Rating}
#   Rating.conservative           -> mu - 3 * sigma  (used for ranking)
#
# Design notes
# ------------
# * Uses the standard factor-graph / EP update rules for the win-loss
#   case (no draws; tennis never draws).
# * TAU (dynamics noise) prevents sigma from collapsing to zero so that
#   later matches always carry weight.
# * Iterative mode: replays all matches N times. Each iteration starts
#   every entity from (previous_mu, fresh SIGMA) so the prior is informed
#   by all evidence from the last pass but sigma is reset, making the
#   effective weight of a match depend only on opponent strength — not on
#   when in the season it was played. Stops early if max mu-change < 1e-3.

import math
from collections import defaultdict
from dataclasses import dataclass

# ── Hyperparameters ────────────────────────────────────────────────────────────

MU    = 25.0        # initial mean skill
SIGMA = MU / 3      # initial uncertainty  (~8.33)
BETA  = SIGMA / 2   # performance noise    (~4.17)
TAU   = SIGMA / 5   # dynamics factor      (~1.67)

ITERATIONS = 5      # number of passes; 3-5 is usually enough for convergence

# ── Normal-distribution helpers ───────────────────────────────────────────────

_SQRT2   = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _phi(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _Phi(x: float) -> float:
    """Standard normal CDF via math.erfc for numerical stability at tails."""
    return 0.5 * math.erfc(-x / _SQRT2)


def _v_win(t: float) -> float:
    denom = _Phi(t)
    if denom < 1e-10:
        return max(0.0, -t)
    return _phi(t) / denom


def _w_win(t: float, v: float) -> float:
    return min(max(v * (v + t), 0.0), 1.0 - 1e-10)


# ── Rating dataclass ──────────────────────────────────────────────────────────

@dataclass
class Rating:
    mu: float    = MU
    sigma: float = SIGMA

    @property
    def conservative(self) -> float:
        """Lower-bound estimate used for ranking: mu − 3σ."""
        return self.mu - 3.0 * self.sigma

    def __repr__(self) -> str:
        return f"Rating(mu={self.mu:.2f}, σ={self.sigma:.2f}, cons={self.conservative:.2f})"


# ── Core update ───────────────────────────────────────────────────────────────

def _update(r_win: Rating, r_lose: Rating) -> tuple[Rating, Rating]:
    """Apply one TrueSkill win/loss update. Returns new Rating objects."""
    sw2 = r_win.sigma  ** 2 + TAU ** 2
    sl2 = r_lose.sigma ** 2 + TAU ** 2
    c2  = 2.0 * BETA ** 2 + sw2 + sl2
    c   = math.sqrt(c2)
    t   = (r_win.mu - r_lose.mu) / c
    v   = _v_win(t)
    w   = _w_win(t, v)
    return (
        Rating(mu=r_win.mu  + (sw2 / c) * v,  sigma=math.sqrt(sw2 * (1.0 - (sw2 / c2) * w))),
        Rating(mu=r_lose.mu - (sl2 / c) * v,  sigma=math.sqrt(sl2 * (1.0 - (sl2 / c2) * w))),
    )


# ── Single pass ───────────────────────────────────────────────────────────────

def _run_pass(
    match_pairs: list[tuple],
    prev_ratings: dict | None,
) -> dict:
    """
    Replay all matches once in order.

    If prev_ratings is supplied, each entity starts from
    (prev_mu, fresh SIGMA) instead of the global default.
    Resetting sigma each pass is what makes timing not matter:
    every match is evaluated against a confident prior built from
    ALL previous evidence, so an August win counts the same as
    an October win.
    """
    ratings: dict = defaultdict(Rating)
    if prev_ratings:
        for entity, r in prev_ratings.items():
            ratings[entity] = Rating(mu=r.mu, sigma=SIGMA)

    for winner, loser in match_pairs:
        ratings[winner], ratings[loser] = _update(ratings[winner], ratings[loser])

    return dict(ratings)


# ── Public entry point ────────────────────────────────────────────────────────

def compute_trueskill(
    match_pairs: list[tuple],
    iterations: int = ITERATIONS,
) -> dict:
    """
    Run iterative TrueSkill and return final ratings.

    Parameters
    ----------
    match_pairs : list of (winner_entity, loser_entity)
        Entities can be any hashable type. Order within each pass
        still matters for tie-breaking but does NOT affect which
        matches carry more weight — that is determined by opponent
        strength, not timing.
    iterations : int
        Maximum number of passes. Stops early if max mu-change
        across all entities drops below 0.001 (converged).

    Returns
    -------
    dict mapping entity -> Rating
    """
    ratings = None

    for i in range(iterations):
        new_ratings = _run_pass(match_pairs, ratings)

        # Convergence check after first pass
        if ratings is not None:
            max_delta = max(
                abs(new_ratings.get(e, Rating()).mu - ratings.get(e, Rating()).mu)
                for e in set(list(new_ratings) + list(ratings))
            )
            if max_delta < 0.001:
                return new_ratings

        ratings = new_ratings

    return ratings
