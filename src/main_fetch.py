# main_fetch.py
# Phase 1: crawl → all_matches.csv + school_meta.json
#
# Run this first; the outputs are consumed by main_rank.py.

import asyncio
import json
import logging

import pandas as pd

from config import MAX_SCHOOLS, TARGET_GENDER
from crawler import crawl_school_matches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

SEED_SCHOOL_ID = 3877   # Berrien Springs — change to any valid school


async def main() -> None:
    logging.info("=" * 62)
    logging.info("Tennis Ranking System  —  Fetch Phase  —  seed: %d", SEED_SCHOOL_ID)
    if TARGET_GENDER:
        logging.info("Gender filter active: %s only", TARGET_GENDER)
    logging.info("=" * 62)

    # ── Phase 1: crawl ────────────────────────────────────────────────────────
    matches, school_meta = await crawl_school_matches(
        seed_id=SEED_SCHOOL_ID,
        max_schools=MAX_SCHOOLS,
    )

    if not matches:
        logging.error("No matches collected — check seed ID, year, and network.")
        return

    # ── Phase 2: persist raw data ─────────────────────────────────────────────
    pd.DataFrame(matches).to_csv("all_matches.csv", index=False)
    logging.info("Saved  all_matches.csv          (%d rows)", len(matches))

    # JSON requires string keys; int school IDs are restored by main_rank.py
    with open("school_meta.json", "w") as fh:
        json.dump({str(k): v for k, v in school_meta.items()}, fh)
    logging.info("Saved  school_meta.json          (%d schools)", len(school_meta))

    logging.info("=" * 62)
    logging.info("Fetch phase done — run main_rank.py next.")


if __name__ == "__main__":
    asyncio.run(main())
