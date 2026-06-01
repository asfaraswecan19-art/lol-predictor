"""
LOLESPORTS API — KILL TIMELINE SCRAPER
========================================
Pulls precise kill timestamps for pro matches from the unofficial lolesports
feed. Use to improve the FT5 model's time-to-5-kills feature beyond the proxy
currently in kill_timelines.csv.

USAGE:
    python lolesports_pull.py probe         # Test a single known match
    python lolesports_pull.py schedule      # List recent matches
    python lolesports_pull.py game <id>     # Pull kill events for one game
    python lolesports_pull.py backfill      # Walk schedule + pull all (slow)

NOTES:
- Uses unofficial endpoints with a hardcoded API key public to lolesports.com.
- These endpoints can break / be rate-limited / be revoked at any time.
- LPL data is typically not available (Riot doesn't share LPL feeds publicly).
- Each game = 2 API calls (window + details). Backfill is slow.
"""
import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests

# The hardcoded key the lolesports.com site itself uses
API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
HEADERS_ESPORTS = {"x-api-key": API_KEY}

# Two domains, two roles:
ESPORTS_API = "https://esports-api.lolesports.com/persisted/gw"   # schedule, match metadata
FEED        = "https://feed.lolesports.com/livestats/v1"           # in-game frames/events

OUTPUT_DIR = Path("kill_timelines_v2")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOW-LEVEL HTTP
# ============================================================
def fetch_json(url, headers=None, verbose=True):
    """GET a URL, return parsed JSON or None. Verbose logging on failure."""
    try:
        r = requests.get(url, headers=headers or {}, timeout=20)
        if verbose:
            print(f"  GET {url}")
            print(f"    -> {r.status_code} ({len(r.content)} bytes)")
        if r.status_code != 200:
            if verbose:
                print(f"    body[:300]: {r.text[:300]}")
            return None
        return r.json()
    except Exception as e:
        if verbose:
            print(f"  EXCEPTION: {e}")
        return None


# ============================================================
# SCHEDULE / MATCH METADATA (esports-api)
# ============================================================
def get_schedule(league_ids=None):
    """Return raw schedule events. league_ids is comma-separated league IDs.
    Without filter, returns recent events across all leagues.
    """
    url = f"{ESPORTS_API}/getSchedule?hl=en-US"
    if league_ids:
        url += f"&leagueId={league_ids}"
    data = fetch_json(url, HEADERS_ESPORTS)
    if not data:
        return []
    return data.get('data', {}).get('schedule', {}).get('events', [])


def get_match_details(match_id):
    """Return match metadata including game IDs."""
    url = f"{ESPORTS_API}/getEventDetails?hl=en-US&id={match_id}"
    data = fetch_json(url, HEADERS_ESPORTS)
    return data.get('data', {}).get('event') if data else None


# ============================================================
# LIVE STATS FEED (feed.lolesports.com)
# ------------------------------------------------------------
# These endpoints serve game frames and event lists. Game ID here is the
# RIOT GAME ID (long numeric), NOT the match ID from the schedule.
# Each match has 1-5 game IDs depending on Bo1/3/5.
#
# /window/{gameId}         -> high-level state snapshots (10s frames)
# /details/{gameId}        -> participant-level details (10s frames)
#
# Both accept ?startingTime=<ISO8601> to paginate. Frames span 30s each;
# to get the whole game you walk forward in 30s increments.
# ============================================================
def get_window(game_id, starting_time=None, verbose=True):
    """Get window frames. starting_time is ISO 8601 UTC (e.g. 2024-05-12T12:00:00Z)."""
    url = f"{FEED}/window/{game_id}"
    if starting_time:
        url += f"?startingTime={starting_time}"
    return fetch_json(url, verbose=verbose)


def get_details(game_id, starting_time=None, verbose=True):
    """Get details frames (participant stats)."""
    url = f"{FEED}/details/{game_id}"
    if starting_time:
        url += f"?startingTime={starting_time}"
    return fetch_json(url, verbose=verbose)


# ============================================================
# KILL TIMELINE EXTRACTION
# ============================================================
def _round_to_10s(ts):
    """Round ISO 8601 timestamp DOWN to nearest 10-second boundary.
    API requires startingTime ending in :00/:10/:20/:30/:40/:50 with no millis.
    """
    if not ts:
        return None
    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
    # Floor seconds to nearest 10
    floored_sec = (dt.second // 10) * 10
    dt = dt.replace(second=floored_sec, microsecond=0)
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def extract_kill_events(game_id, verbose=True):
    """Walk through window frames of a game, return list of kill events.

    The API serves ~10 seconds of frames per call. To get the full game we
    step startingTime forward by 10s each iteration.

    Returns: list of dicts {side, kill_number_for_team, t_secs, ts}
    """
    if verbose:
        print(f"\n  Extracting kills for game {game_id}...")
    initial = get_window(game_id, verbose=verbose)
    if not initial:
        return []

    frames = initial.get('frames', []) or []
    if not frames:
        if verbose: print("    No frames available — game data not stored, or LPL/blocked.")
        return []

    # Game start = timestamp of first frame, rounded down to 10s boundary
    game_start_iso = _round_to_10s(frames[0].get('rfc460Timestamp'))
    if verbose:
        print(f"    Game start (10s-floored): {game_start_iso}")

    # We'll step forward 10 seconds at a time from game start.
    # Each call returns ~10s worth of frames (multiple snapshots).
    # We keep ONE frame per 10s window — the last one (most up-to-date kill count).
    all_frames = []
    cursor = game_start_iso
    iterations = 0
    consecutive_empty = 0
    MAX_ITER = 240  # 240 * 10s = 40 min, plenty for any pro game
    while iterations < MAX_ITER and consecutive_empty < 5:
        if iterations == 0:
            page = initial  # already fetched
        else:
            page = get_window(game_id, starting_time=cursor, verbose=False)
            if not page:
                consecutive_empty += 1
                cursor = _step_10s(cursor)
                iterations += 1
                if verbose and iterations % 10 == 0:
                    print(f"    [{iterations}/{MAX_ITER}] cursor={cursor} (empty x{consecutive_empty})", flush=True)
                continue

        page_frames = page.get('frames', []) or []
        if page_frames:
            all_frames.append(page_frames[-1])
            consecutive_empty = 0
            if verbose and iterations % 10 == 0:
                lf = page_frames[-1]
                bk = (lf.get('blueTeam') or {}).get('totalKills', 0)
                rk = (lf.get('redTeam')  or {}).get('totalKills', 0)
                gs = lf.get('gameState','?')
                print(f"    [{iterations}/{MAX_ITER}] cursor={cursor} blue={bk} red={rk} state={gs}", flush=True)
            # If game is finished and we have stable kill counts, we can stop
            if page_frames[-1].get('gameState') == 'finished':
                if verbose:
                    print(f"    Game state = finished at iteration {iterations}, stopping.", flush=True)
                break
        else:
            consecutive_empty += 1

        cursor = _step_10s(cursor)
        iterations += 1
        if iterations > 1:
            time.sleep(0.15)  # be gentle

    if verbose:
        print(f"    Collected {len(all_frames)} 10s snapshots across {iterations} pages")
        if all_frames:
            first_kills_b = (all_frames[0].get('blueTeam') or {}).get('totalKills', 0)
            last_kills_b  = (all_frames[-1].get('blueTeam') or {}).get('totalKills', 0)
            last_kills_r  = (all_frames[-1].get('redTeam')  or {}).get('totalKills', 0)
            print(f"    First snapshot blue kills: {first_kills_b}")
            print(f"    Last snapshot:  blue={last_kills_b}, red={last_kills_r}")
            print(f"    Last gameState: {all_frames[-1].get('gameState')}")

    # Detect kills by tracking totalKills increment between consecutive snapshots
    kills = []
    prev_blue = 0
    prev_red  = 0
    for f in all_frames:
        bt = f.get('blueTeam', {}) or {}
        rt = f.get('redTeam',  {}) or {}
        cur_blue = bt.get('totalKills', 0) or 0
        cur_red  = rt.get('totalKills', 0) or 0
        ts = f.get('rfc460Timestamp')
        try:
            t_secs = (datetime.fromisoformat(ts.replace('Z','+00:00'))
                      - datetime.fromisoformat(game_start_iso.replace('Z','+00:00'))).total_seconds()
        except Exception:
            t_secs = None

        for _ in range(max(0, cur_blue - prev_blue)):
            kills.append({'side':'blue', 'kill_number_for_team':prev_blue+1, 't_secs':t_secs, 'ts':ts})
            prev_blue += 1
        for _ in range(max(0, cur_red - prev_red)):
            kills.append({'side':'red',  'kill_number_for_team':prev_red+1, 't_secs':t_secs, 'ts':ts})
            prev_red += 1

    if verbose:
        print(f"    Total kills extracted: {len(kills)} (blue: {prev_blue}, red: {prev_red})")
    return kills


def _step_10s(iso_ts):
    """Return iso_ts + 10 seconds, formatted same way."""
    dt = datetime.fromisoformat(iso_ts.replace('Z', '+00:00'))
    dt = dt + timedelta(seconds=10)
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def summarize_kills(kills):
    """Compute time to 5th kill for each side, and game-total stats."""
    blue_kills = [k for k in kills if k['side']=='blue']
    red_kills  = [k for k in kills if k['side']=='red']
    summary = {
        'blue_total_kills': len(blue_kills),
        'red_total_kills':  len(red_kills),
        'blue_time_to_5':   None,
        'red_time_to_5':    None,
        'blue_kills_at_10min':  sum(1 for k in blue_kills if (k['t_secs'] or 0) <= 600),
        'red_kills_at_10min':   sum(1 for k in red_kills  if (k['t_secs'] or 0) <= 600),
        'blue_kills_at_15min':  sum(1 for k in blue_kills if (k['t_secs'] or 0) <= 900),
        'red_kills_at_15min':   sum(1 for k in red_kills  if (k['t_secs'] or 0) <= 900),
    }
    if len(blue_kills) >= 5 and blue_kills[4]['t_secs']:
        summary['blue_time_to_5'] = blue_kills[4]['t_secs'] / 60
    if len(red_kills) >= 5 and red_kills[4]['t_secs']:
        summary['red_time_to_5'] = red_kills[4]['t_secs'] / 60
    if summary['blue_time_to_5'] and summary['red_time_to_5']:
        summary['first_to_five'] = 'blue' if summary['blue_time_to_5'] < summary['red_time_to_5'] else 'red'
    elif summary['blue_time_to_5']:
        summary['first_to_five'] = 'blue'
    elif summary['red_time_to_5']:
        summary['first_to_five'] = 'red'
    else:
        summary['first_to_five'] = None  # neither team reached 5
    return summary


# ============================================================
# MODES
# ============================================================
def probe():
    """Smoke-test endpoints and print everything we learn."""
    print("=" * 60)
    print("  PROBE")
    print("=" * 60)

    print("\n[1] Hitting schedule endpoint...")
    events = get_schedule()
    print(f"    Got {len(events)} events")
    if events:
        # Find a recent completed match
        completed = [e for e in events if e.get('state')=='completed' and e.get('match')]
        if not completed:
            print("    No completed matches in schedule window. Try again later.")
            return
        sample = completed[-1]
        print(f"    Sample match: {sample.get('league',{}).get('name')} | "
              f"{sample.get('match',{}).get('teams',[{}])[0].get('code')} vs "
              f"{sample.get('match',{}).get('teams',[{}])[1].get('code')}")
        match_id = sample.get('match',{}).get('id')

        print(f"\n[2] Fetching match details for match_id={match_id}...")
        details = get_match_details(match_id)
        if not details:
            print("    Match details fetch failed")
            return
        games = details.get('match', {}).get('games', [])
        print(f"    Games in match: {len(games)}")
        for i, g in enumerate(games):
            print(f"      Game {i+1}: id={g.get('id')} state={g.get('state')} number={g.get('number')}")

        # Try to pull kills for the first completed game
        completed_games = [g for g in games if g.get('state')=='completed']
        if not completed_games:
            print("\n    No completed games to test kill extraction on.")
            return
        game_id = completed_games[0].get('id')

        print(f"\n[3] Extracting kills for game_id={game_id}...")
        kills = extract_kill_events(game_id)
        if kills:
            summary = summarize_kills(kills)
            print(f"\n    Summary:")
            for k, v in summary.items():
                print(f"      {k}: {v}")
        else:
            print("    No kill data returned (game might be too old or LPL-blocked)")


def cmd_schedule():
    events = get_schedule()
    completed = [e for e in events if e.get('state')=='completed']
    print(f"Schedule events: {len(events)} total, {len(completed)} completed")
    for e in completed[-20:]:
        m = e.get('match', {})
        teams = m.get('teams', [])
        if len(teams) >= 2:
            print(f"  {e.get('startTime','')[:10]} {e.get('league',{}).get('name','?'):10} "
                  f"{teams[0].get('code','?'):4} vs {teams[1].get('code','?'):4}  "
                  f"match_id={m.get('id')}")


def cmd_game(game_id):
    kills = extract_kill_events(game_id)
    summary = summarize_kills(kills)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    # Save raw
    out = OUTPUT_DIR / f"{game_id}_kills.json"
    with open(out, 'w') as f:
        json.dump({'game_id':game_id,'kills':kills,'summary':summary}, f, indent=2)
    print(f"\nSaved: {out}")


def cmd_dump(game_id):
    """Save raw window JSON to disk so we can inspect the schema."""
    print(f"Dumping raw frames for game {game_id}...")
    initial = get_window(game_id)
    if not initial:
        print("No response.")
        return
    # Save initial response
    out1 = OUTPUT_DIR / f"{game_id}_window_initial.json"
    with open(out1, 'w') as f:
        json.dump(initial, f, indent=2)
    print(f"\nSaved initial response: {out1}")

    frames = initial.get('frames', []) or []
    print(f"\nInitial frames count: {len(frames)}")
    if frames:
        first = frames[0]; last = frames[-1]
        print(f"First frame keys: {list(first.keys())}")
        print(f"First frame timestamp: {first.get('rfc460Timestamp')}")
        print(f"Last frame timestamp:  {last.get('rfc460Timestamp')}")
        print(f"\nFirst frame blueTeam keys: {list((first.get('blueTeam') or {}).keys())}")
        print(f"First frame blueTeam totalKills: {(first.get('blueTeam') or {}).get('totalKills')}")
        print(f"Last  frame blueTeam totalKills: {(last.get('blueTeam') or {}).get('totalKills')}")
        print(f"Last  frame redTeam  totalKills: {(last.get('redTeam')  or {}).get('totalKills')}")

    # Try pagination: pass the LAST timestamp and fetch again
    if frames:
        last_ts = frames[-1].get('rfc460Timestamp')
        print(f"\nPaginating with startingTime={last_ts}...")
        nxt = get_window(game_id, starting_time=last_ts)
        if nxt:
            new_frames = nxt.get('frames', []) or []
            print(f"  Got {len(new_frames)} frames in next page")
            if new_frames:
                print(f"  Next page first ts: {new_frames[0].get('rfc460Timestamp')}")
                print(f"  Next page last  ts: {new_frames[-1].get('rfc460Timestamp')}")
                print(f"  Last kills: blue={new_frames[-1].get('blueTeam',{}).get('totalKills')}, "
                      f"red={new_frames[-1].get('redTeam',{}).get('totalKills')}")
            out2 = OUTPUT_DIR / f"{game_id}_window_page2.json"
            with open(out2, 'w') as f:
                json.dump(nxt, f, indent=2)
            print(f"  Saved page 2: {out2}")


def cmd_schedule_back(league_id=None):
    """Walk schedule backward via pageToken to see how far history goes.
    Prints earliest event date reached and total events collected per page."""
    print(f"Walking schedule backward...")
    page = 0
    page_token = None
    earliest_date = None
    total_events = 0
    league_counts = {}
    while page < 50:  # safety cap
        url = f"{ESPORTS_API}/getSchedule?hl=en-US"
        if league_id:
            url += f"&leagueId={league_id}"
        if page_token:
            url += f"&pageToken={page_token}"
        data = fetch_json(url, HEADERS_ESPORTS, verbose=(page < 3))  # quiet after page 3
        if not data:
            print(f"  Stopped at page {page} (fetch failed)")
            break
        sched = data.get('data', {}).get('schedule', {}) or {}
        events = sched.get('events', []) or []
        if not events:
            print(f"  Page {page}: 0 events, stopping")
            break
        total_events += len(events)
        # Get earliest event in this page
        page_dates = [e.get('startTime') for e in events if e.get('startTime')]
        if page_dates:
            page_min = min(page_dates)
            if earliest_date is None or page_min < earliest_date:
                earliest_date = page_min
        # Count leagues
        for e in events:
            lg = (e.get('league') or {}).get('name', '?')
            league_counts[lg] = league_counts.get(lg, 0) + 1
        # Get older token (for walking BACKWARD in time)
        older = sched.get('pages', {}).get('older')
        print(f"  Page {page}: {len(events)} events, earliest in page: {min(page_dates) if page_dates else 'n/a'}, older_token: {bool(older)}")
        if not older:
            print(f"  No 'older' token, reached end of available history")
            break
        page_token = older
        page += 1
        time.sleep(0.3)
    print(f"\nTotal events collected: {total_events}")
    print(f"Earliest event date:    {earliest_date}")
    print(f"\nEvents per league (top 20):")
    for lg, n in sorted(league_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {lg:<30} {n}")


# ============================================================
# ENTRY
# ============================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    mode = sys.argv[1]
    if mode == 'probe':
        probe()
    elif mode == 'schedule':
        cmd_schedule()
    elif mode == 'game' and len(sys.argv) >= 3:
        cmd_game(sys.argv[2])
    elif mode == 'dump' and len(sys.argv) >= 3:
        cmd_dump(sys.argv[2])
    elif mode == 'schedule_back':
        league = sys.argv[2] if len(sys.argv) >= 3 else None
        cmd_schedule_back(league)
    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
