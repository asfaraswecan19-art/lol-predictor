"""
leaguepedia_scraper.py
======================
Fetches pro match data from Leaguepedia's Cargo API.

SETUP (required — anonymous queries are rate-limited to almost nothing):
  1. Create a free Fandom account at https://www.fandom.com/register
  2. Go to https://lol.fandom.com/wiki/Special:BotPasswords
  3. "Create a new bot password" — name it anything, e.g. "predictor"
  4. Tick "Basic rights" and "High-volume editing" under Grants
  5. Click Create — note the generated username (YourName@predictor) and password
  6. Fill in BOT_USERNAME and BOT_PASSWORD below

Install:
    python -m pip install mwclient pandas

Run:
    python leaguepedia_scraper.py

Outputs (in ./data/):
    leaguepedia_games.csv      — one row per game: teams, winner, patch, picks, bans
    leaguepedia_players.csv    — player directory for name normalisation
"""

import time, os, json, random, hashlib
import pandas as pd
import mwclient
import mwclient.errors
from pathlib import Path

# ---------------------------------------------------------------------------
# FILL THESE IN
# ---------------------------------------------------------------------------
BOT_USERNAME = ""   # e.g. "YourFandomName@predictor"
BOT_PASSWORD = ""   # the generated bot password
# ---------------------------------------------------------------------------

OUTPUT_DIR = "data"
CACHE_DIR  = "data/.lp_cache"

PAGE_SIZE      = 500
PAGE_DELAY     = 1.5   # seconds between API pages (logged-in limit is generous)
RATELIMIT_WAIT = 70    # seconds to wait if we still hit a rate limit
MAX_RETRIES    = 6


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _key(tables, fields, where, offset):
    raw = f"{tables}|{fields}|{where}|{offset}"
    return hashlib.md5(raw.encode()).hexdigest()  # always 32 chars, safe on all OS

def _cpath(key):
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    return Path(CACHE_DIR) / f"{key}.json"

def _load(key):
    cp = _cpath(key)
    return json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else None

def _save(key, data):
    _cpath(key).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------

def connect():
    if not BOT_USERNAME or not BOT_PASSWORD:
        print(
            "\nERROR: BOT_USERNAME and BOT_PASSWORD are empty.\n"
            "Open the script and fill in the CONFIG section at the top.\n"
            "See the docstring for setup instructions (~5 minutes).\n"
        )
        raise SystemExit(1)

    print("Connecting to lol.fandom.com ...")
    site = mwclient.Site("lol.fandom.com", path="/", max_retries=0, retry_timeout=0)
    print(f"Logging in as {BOT_USERNAME} ...")
    site.login(BOT_USERNAME, BOT_PASSWORD)
    print(f"Logged in. Groups: {site.groups}\n")
    return site


# ---------------------------------------------------------------------------
# Single page fetch with retry
# ---------------------------------------------------------------------------

def fetch_page(site, tables, fields, where="", order_by="", offset=0):
    key    = _key(tables, fields, where, offset)
    cached = _load(key)
    if cached is not None:
        return cached

    kwargs = dict(tables=tables, fields=fields, limit=PAGE_SIZE,
                  offset=offset, format="json")
    if where:    kwargs["where"]    = where
    if order_by: kwargs["order_by"] = order_by

    wait = RATELIMIT_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = site.api("cargoquery", **kwargs)

            if "error" in result:
                code = result["error"].get("code", "")
                info = result["error"].get("info", "")
                if code == "ratelimited":
                    t = wait + random.uniform(0, 10)
                    print(f"\n  [rate limited] waiting {t:.0f}s (attempt {attempt}/{MAX_RETRIES})...")
                    time.sleep(t); wait = min(wait * 1.5, 300); continue
                raise RuntimeError(f"API error [{code}]: {info}")

            rows = [r["title"] for r in result.get("cargoquery", [])]
            _save(key, rows)
            return rows

        except mwclient.errors.APIError as e:
            code = str(getattr(e, "code", e))
            if "ratelimited" in code.lower():
                t = wait + random.uniform(0, 10)
                print(f"\n  [rate limited] waiting {t:.0f}s (attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(t); wait = min(wait * 1.5, 300); continue
            if attempt < MAX_RETRIES:
                print(f"\n  [APIError attempt {attempt}/{MAX_RETRIES}] {e} — retrying in 20s")
                time.sleep(20); continue
            raise

        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"\n  [error attempt {attempt}/{MAX_RETRIES}] {e} — retrying in {15*attempt}s")
                time.sleep(15 * attempt); continue
            raise

    raise RuntimeError(f"Gave up after {MAX_RETRIES} retries (offset={offset})")


def fetch_all(site, tables, fields, where="", order_by=""):
    all_rows, offset, page = [], 0, 1
    while True:
        src = "cache" if _load(_key(tables, fields, where, offset)) else "API  "
        print(f"  page {page:3d}  offset {offset:5d}  [{src}] ...", end=" ", flush=True)
        batch = fetch_page(site, tables, fields, where=where, order_by=order_by, offset=offset)
        all_rows.extend(batch)
        print(f"{len(batch)} rows  ({len(all_rows):,} total)")
        if len(batch) < PAGE_SIZE: break
        offset += PAGE_SIZE; page += 1
        if src == "API  ": time.sleep(PAGE_DELAY + random.uniform(0, 0.5))
    return all_rows


# ---------------------------------------------------------------------------
# Games  (ScoreboardGames — confirmed field names)
#
# Key facts learned from leaguepedia-parser source:
#   - Winner is "1" (blue) or "2" (red), NOT a team name
#   - Team1Picks / Team2Picks are comma-separated champion strings
#   - Team1Bans  / Team2Bans  are comma-separated champion strings
#   - Team1Players / Team2Players are comma-separated player name strings
#   - Per-slot picks/bans live in PicksAndBansS7, joined via GameId
# ---------------------------------------------------------------------------

def fetch_games(site):
    print("=== Fetching games (ScoreboardGames) ===")

    # Only use fields that actually exist in ScoreboardGames
    fields = ", ".join([
        "GameId", "MatchId", "Tournament", "Team1", "Team2",
        "Winner",             # "1" = blue wins, "2" = red wins
        "Patch", "DateTime_UTC",
        "Team1Score", "Team2Score",
        "Gamelength_Number",
        "Team1Picks",         # comma-separated, e.g. "Azir,Jinx,Thresh,Garen,Vi"
        "Team2Picks",
        "Team1Bans",          # comma-separated
        "Team2Bans",
        "Team1Players",       # comma-separated player names
        "Team2Players",
        "OverviewPage",
        "VOD",
    ])

    rows = fetch_all(
        site,
        tables   = "ScoreboardGames",
        fields   = fields,
        where    = (
            "DateTime_UTC >= '2022-01-01' AND ("
            "Tournament LIKE '%LCK%' OR "
            "Tournament LIKE '%LPL%' OR "
            "Tournament LIKE '%LEC%' OR "
            "Tournament LIKE '%LCS%' OR "
            "Tournament LIKE '%LTA North%' OR "
            "Tournament LIKE '%LTA South%' OR "
            "Tournament LIKE '%CBLOL%' OR "
            "Tournament LIKE '%World Championship%' OR "
            "Tournament LIKE '%MSI%' OR "
            "Tournament LIKE '%Mid-Season Invitational%'"
            ")"
        ),
        order_by = "DateTime_UTC ASC",
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.columns = df.columns.str.strip()

    # Rename for clarity
    df = df.rename(columns={"Gamelength Number": "gamelength_minutes"})

    # Derived columns
    df["DateTime_UTC"] = pd.to_datetime(df["DateTime_UTC"], errors="coerce")
    df["patch_major"]  = df["Patch"].astype(str).str.extract(r"^(\d+\.\d+)")[0]
    df["blue_team"]    = df["Team1"]
    df["red_team"]     = df["Team2"]
    # Winner "1" = blue side, "2" = red side
    df["blue_win"]     = (df["Winner"].astype(str).str.strip() == "1").astype(int)

    # Expand comma-separated picks into individual columns (top/jng/mid/bot/sup order)
    for side, col in [("blue", "Team1Picks"), ("red", "Team2Picks")]:
        if col in df.columns:
            split = df[col].fillna("").str.split(",", expand=True)
            roles = ["top","jng","mid","bot","sup"]
            for i, role in enumerate(roles):
                df[f"{side}_pick_{role}"] = split[i].str.strip() if i < len(split.columns) else ""

    # Expand bans (5 per team)
    for side, col in [("blue", "Team1Bans"), ("red", "Team2Bans")]:
        if col in df.columns:
            split = df[col].fillna("").str.split(",", expand=True)
            for i in range(5):
                df[f"{side}_ban{i+1}"] = split[i].str.strip() if i < len(split.columns) else ""

    # Flag international events
    df["is_international"] = (
        df["Tournament"].str.lower().str.contains("worlds|msi|allstar", na=False)
    ).astype(int)

    # Region
    region_map = {
        "LCK":"Korea","LPL":"China","LEC":"Europe","LCS":"NorthAmerica",
        "VCS":"Vietnam","PCS":"Pacific","CBLOL":"Brazil","LLA":"LatAm",
    }
    df["region"] = "Unknown"
    for abbr, region in region_map.items():
        df.loc[df["Tournament"].str.contains(abbr, case=False, na=False), "region"] = region

    print(f"Games: {len(df):,}\n")
    return df


# ---------------------------------------------------------------------------
# Player directory
# ---------------------------------------------------------------------------

def fetch_players(site):
    print("=== Fetching player directory ===")
    rows = fetch_all(
        site,
        tables = "Players",
        fields = "ID, Name, Country, NationalityPrimary, Role, Team, Residency",
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.columns = df.columns.str.strip()
    print(f"Players: {len(df):,}\n")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR,  exist_ok=True)

    site = connect()

    games = fetch_games(site)
    if not games.empty:
        p = os.path.join(OUTPUT_DIR, "leaguepedia_games.csv")
        games.to_csv(p, index=False)
        print(f"Saved: {p}  ({len(games):,} rows x {len(games.columns)} cols)")

    players = fetch_players(site)
    if not players.empty:
        p = os.path.join(OUTPUT_DIR, "leaguepedia_players.csv")
        players.to_csv(p, index=False)
        print(f"Saved: {p}  ({len(players):,} rows)")

    print("\nDone. Re-run anytime — finished pages load from cache instantly.")

    if not games.empty:
        cols = ["DateTime_UTC","blue_team","red_team","blue_win",
                "patch_major","region","blue_pick_top","red_pick_top"]
        print("\nSample:")
        print(games[[c for c in cols if c in games.columns]].head(4).to_string())


if __name__ == "__main__":
    main()
