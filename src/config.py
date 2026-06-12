# config.py — parameter settings for the tennis ranking system.

# Year (for record-keeping) and filters
YEAR           = 2025
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
