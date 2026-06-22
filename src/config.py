# config.py — parameter settings for the tennis ranking system.

# Year (for record-keeping) and filters
import datetime

today = datetime.date.today()
# Stay on the previous year until April 1 (adjust month as needed)
if today.month < 8:
    YEAR = today.year - 1
else:
    YEAR = today.year
IS_NOT_VARSITY = 0           # 0 = varsity only

TARGET_STATE  = "MI"         # or None for no filter
TARGET_GENDER = "Boys"       # or "Girls" or None for both

MAX_SCHOOLS = None           # optional crawl limit

# Minimum matches to appear in rankings
MIN_MATCHES = 5

# Division lookups (not needed for core logic; used in ranking output)
TARGET_DIVISION = None
TARGET_FLIGHT   = None
TARGET_POOL     = None
