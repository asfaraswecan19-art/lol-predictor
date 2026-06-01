"""
fetch_kills.py — Extract kill timelines for every game in game_ids.csv.
Uses per-game JSON files in kill_data/ as the checkpoint.
Restart-safe: skips any game whose file already exists.

PARALLEL: 4 worker threads. Configurable via --workers N.

USAGE:
    python fetch_kills.py               # all games, 4 workers
    python fetch_kills.py --workers 2   # gentler concurrency
    python fetch_kills.py --tier1       # only LCK/LEC/LCS/etc
    python fetch_kills.py --limit 100   # cap for testing
    python fetch_kills.py --retry       # only retry games in _failed.txt
    python fetch_kills.py --stats       # print progress/coverage stats

OUTPUT:
    kill_data/{game_id}.json      — kill list + summary for one game
    kill_data/_no_data/{game_id}  — empty marker for games with no frames
    kill_data/_failed.txt         — transient errors, retry these later
"""
import argparse, csv, json, os, signal, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
import requests

API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
HEADERS = {"x-api-key": API_KEY}
FEED = "https://feed.lolesports.com/livestats/v1"

GAME_IDS_CSV = Path("game_ids.csv")
OUT_DIR      = Path("kill_data")
NO_DATA_DIR  = OUT_DIR / "_no_data"
FAILED_LOG   = OUT_DIR / "_failed.txt"

OUT_DIR.mkdir(exist_ok=True)
NO_DATA_DIR.mkdir(exist_ok=True)

SLEEP_BETWEEN_PAGES = 0.05
SLEEP_BETWEEN_GAMES = 0.05  # smaller because workers stagger naturally
MAX_PAGES_PER_GAME  = 240
MAX_EMPTY_PAGES     = 5
DEFAULT_WORKERS     = 4

TIER1_LEAGUES = {'LCK','LPL','LEC','LCS','LCP','CBLOL','MSI','WLDs','Worlds',
                 'FST','LTA N','LTA S','LTA','EMEA Masters'}

_STOP_REQUESTED = False
_print_lock = threading.Lock()
_log_lock = threading.Lock()
def _safe_print(msg):
    with _print_lock:
        print(msg, flush=True)

def _handle_sigint(sig, frame):
    global _STOP_REQUESTED
    _safe_print("\n[Ctrl+C — finishing in-progress games then stopping]")
    _STOP_REQUESTED = True
signal.signal(signal.SIGINT, _handle_sigint)


# ============================================================
# API helpers
# ============================================================
def _round_to_10s(ts):
    if not ts: return None
    dt = datetime.fromisoformat(ts.replace('Z','+00:00'))
    dt = dt.replace(second=(dt.second // 10) * 10, microsecond=0)
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def _step_10s(iso):
    dt = datetime.fromisoformat(iso.replace('Z','+00:00')) + timedelta(seconds=10)
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def fetch_window(game_id, starting_time=None, retries=2):
    url = f"{FEED}/window/{game_id}"
    if starting_time:
        url += f"?startingTime={starting_time}"
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=25)
            if r.status_code == 200:
                return r.json(), None
            if r.status_code == 204:
                # 204 = "No Content" — game has no live-stats data on the feed.
                # Treat as no_data (no retry) rather than a transient failure.
                return None, "no_data"
            if r.status_code in (429, 502, 503, 504):
                # transient — back off and retry
                time.sleep(2 + attempt * 3)
                continue
            return None, f"http_{r.status_code}"
        except requests.exceptions.Timeout:
            time.sleep(2 + attempt * 3)
            continue
        except Exception as e:
            return None, f"exception_{type(e).__name__}"
    return None, "retries_exhausted"


# ============================================================
# Extraction
# ============================================================
def extract_kills(game_id):
    """Pull all frames, return (kills_list, error_str_or_None).
    error_str: 'no_data' if game has no frames (LPL etc — don't retry)
               other strings = transient failure (retry later)
    """
    initial, err = fetch_window(game_id)
    if err:
        return None, err
    if not initial:
        return None, "no_response"

    frames = initial.get('frames', []) or []
    if not frames:
        return None, "no_data"  # empty marker, no retry

    game_start_iso = _round_to_10s(frames[0].get('rfc460Timestamp'))
    if not game_start_iso:
        return None, "no_timestamp"

    all_frames = []
    cursor = game_start_iso
    iterations = 0
    consecutive_empty = 0
    finished_seen = False

    while iterations < MAX_PAGES_PER_GAME and consecutive_empty < MAX_EMPTY_PAGES:
        if iterations == 0:
            page = initial
        else:
            page, err = fetch_window(game_id, starting_time=cursor)
            if err:
                # transient mid-walk — give up this game, retry later
                return None, f"midwalk_{err}"

        page_frames = (page.get('frames', []) or []) if page else []
        if page_frames:
            all_frames.append(page_frames[-1])
            consecutive_empty = 0
            if page_frames[-1].get('gameState') == 'finished':
                finished_seen = True
                break
        else:
            consecutive_empty += 1

        cursor = _step_10s(cursor)
        iterations += 1
        if iterations > 1:
            time.sleep(SLEEP_BETWEEN_PAGES)

    if not finished_seen and consecutive_empty >= MAX_EMPTY_PAGES:
        # game data appears truncated, but we may have what we need
        pass

    # Extract kills from snapshots
    kills = []
    prev_blue = prev_red = 0
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
            kills.append({'side':'blue','kill_num':prev_blue+1,'t_secs':t_secs,'ts':ts})
            prev_blue += 1
        for _ in range(max(0, cur_red - prev_red)):
            kills.append({'side':'red','kill_num':prev_red+1,'t_secs':t_secs,'ts':ts})
            prev_red += 1

    if not kills:
        # Game completed but no kills detected — unusual, mark for inspection
        # Treat as no_data so we don't retry forever
        return None, "no_kills_extracted"

    return kills, None


def summarize(kills):
    blue = [k for k in kills if k['side']=='blue']
    red  = [k for k in kills if k['side']=='red']
    s = {
        'blue_total_kills': len(blue),
        'red_total_kills':  len(red),
        'blue_time_to_5':   blue[4]['t_secs']/60 if len(blue)>=5 and blue[4]['t_secs'] else None,
        'red_time_to_5':    red[4]['t_secs']/60  if len(red)>=5  and red[4]['t_secs']  else None,
        'blue_kills_at_10min': sum(1 for k in blue if (k['t_secs'] or 0) <= 600),
        'red_kills_at_10min':  sum(1 for k in red  if (k['t_secs'] or 0) <= 600),
        'blue_kills_at_15min': sum(1 for k in blue if (k['t_secs'] or 0) <= 900),
        'red_kills_at_15min':  sum(1 for k in red  if (k['t_secs'] or 0) <= 900),
    }
    if s['blue_time_to_5'] and s['red_time_to_5']:
        s['first_to_five'] = 'blue' if s['blue_time_to_5'] < s['red_time_to_5'] else 'red'
    elif s['blue_time_to_5']: s['first_to_five'] = 'blue'
    elif s['red_time_to_5']:  s['first_to_five'] = 'red'
    else:                     s['first_to_five'] = None
    return s


# ============================================================
# Driver
# ============================================================
def load_game_list(tier1=False, retry_only=False):
    """Yield (game_id, meta_dict) for all games we should process."""
    if retry_only:
        if not FAILED_LOG.exists():
            print("No _failed.txt to retry."); return
        with FAILED_LOG.open() as f:
            for line in f:
                parts = line.strip().split('\t')
                if parts and parts[0]:
                    yield parts[0], {}
        return

    with GAME_IDS_CSV.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('state') != 'completed': continue
            if tier1 and row.get('league') not in TIER1_LEAGUES: continue
            yield row['game_id'], row


def is_done(game_id):
    """Has this game already been processed (either successfully or marked no_data)?"""
    return (OUT_DIR / f"{game_id}.json").exists() or (NO_DATA_DIR / f"{game_id}").exists()


def write_result(game_id, meta, kills, summary):
    out = OUT_DIR / f"{game_id}.json"
    payload = {
        'game_id': game_id,
        'match_id': meta.get('match_id'),
        'league': meta.get('league'),
        'start_time': meta.get('start_time'),
        'blue_team': meta.get('blue_team'),
        'red_team': meta.get('red_team'),
        'summary': summary,
        'kills': kills,
    }
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, separators=(',',':')))
    os.replace(tmp, out)  # atomic rename


def mark_no_data(game_id):
    (NO_DATA_DIR / game_id).write_text('')


def log_failed(game_id, reason):
    with _log_lock:
        with FAILED_LOG.open('a') as f:
            f.write(f"{game_id}\t{reason}\t{datetime.now().isoformat()}\n")


def stats():
    """Print summary of progress."""
    success = list(OUT_DIR.glob("*.json"))
    no_data = list(NO_DATA_DIR.glob("*"))
    n_failed = 0
    if FAILED_LOG.exists():
        with FAILED_LOG.open() as f:
            n_failed = sum(1 for _ in f)
    total_in_csv = 0
    if GAME_IDS_CSV.exists():
        with GAME_IDS_CSV.open(encoding='utf-8') as f:
            total_in_csv = sum(1 for _ in csv.DictReader(f) if _.get('state')=='completed')
    print(f"Games successfully extracted: {len(success)}")
    print(f"Games marked no_data:         {len(no_data)}")
    print(f"Failure log entries:          {n_failed}")
    print(f"Completed games in CSV:       {total_in_csv}")
    if total_in_csv:
        done = len(success) + len(no_data)
        pct = done / total_in_csv * 100
        print(f"Coverage:                     {done}/{total_in_csv} ({pct:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier1', action='store_true', help='Only tier-1 leagues')
    ap.add_argument('--limit', type=int, default=0, help='Cap games processed (testing)')
    ap.add_argument('--retry', action='store_true', help='Retry games in _failed.txt only')
    ap.add_argument('--stats', action='store_true', help='Print stats and exit')
    ap.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                    help=f'Parallel worker count (default {DEFAULT_WORKERS})')
    args = ap.parse_args()

    if args.stats:
        stats(); return

    if not GAME_IDS_CSV.exists() and not args.retry:
        print(f"ERROR: {GAME_IDS_CSV} not found. Run fetch_game_ids.py first.")
        return

    # If retrying, clear the log so we can rebuild it for genuinely-still-failing games
    if args.retry and FAILED_LOG.exists():
        retry_targets = set()
        with FAILED_LOG.open() as f:
            for line in f:
                gid = line.strip().split('\t')[0]
                if gid: retry_targets.add(gid)
        FAILED_LOG.unlink()
        print(f"Retrying {len(retry_targets)} previously-failed games")

    todo = []
    for game_id, meta in load_game_list(tier1=args.tier1, retry_only=args.retry):
        if is_done(game_id): continue
        todo.append((game_id, meta))
        if args.limit and len(todo) >= args.limit: break

    print(f"Games to process this session: {len(todo)}")
    if not todo:
        stats(); return

    start_time = time.time()
    n_success = n_nodata = n_failed = 0
    completed_count = 0
    counter_lock = threading.Lock()

    _safe_print(f"Starting kill extraction with {args.workers} parallel workers. "
                f"First games take ~30-45 sec each.")

    def process_one(idx, game_id, meta):
        """Worker function: extract one game, return result tuple."""
        if _STOP_REQUESTED:
            return ('skipped', idx, game_id, meta, None)

        kills, err = extract_kills(game_id)
        if err == 'no_data' or err == 'no_kills_extracted':
            mark_no_data(game_id)
            return ('no_data', idx, game_id, meta, err)
        elif err:
            log_failed(game_id, err)
            return ('failed', idx, game_id, meta, err)
        else:
            summary = summarize(kills)
            write_result(game_id, meta, kills, summary)
            return ('success', idx, game_id, meta, summary)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, i, gid, meta): (i, gid, meta)
                   for i, (gid, meta) in enumerate(todo, 1)}

        for fut in as_completed(futures):
            if _STOP_REQUESTED:
                pool.shutdown(wait=False, cancel_futures=True)
                break

            try:
                result_type, idx, game_id, meta, payload = fut.result()
            except Exception as e:
                _safe_print(f"  WORKER CRASHED: {e}")
                continue

            if result_type == 'skipped':
                continue
            elif result_type == 'no_data':
                n_nodata += 1
                status = f"no_data ({payload})"
            elif result_type == 'failed':
                n_failed += 1
                status = f"FAIL ({payload})"
            else:  # success
                n_success += 1
                status = f"OK ({payload.get('blue_total_kills',0)}+{payload.get('red_total_kills',0)} kills)"

            with counter_lock:
                completed_count += 1
                done = completed_count
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(todo) - done) / rate if rate > 0 else 0
            _safe_print(f"  [{done}/{len(todo)}] {meta.get('league','?'):10} "
                        f"{meta.get('blue_team','?')}/{meta.get('red_team','?'):8} "
                        f"game={game_id}... {status}  "
                        f"[{rate:.2f}/s, ETA {remaining/60:.0f}m]")

            time.sleep(SLEEP_BETWEEN_GAMES)

    print(f"\nSession done.")
    print(f"  Success: {n_success}")
    print(f"  No-data: {n_nodata}")
    print(f"  Failed:  {n_failed}")
    print()
    stats()


if __name__ == '__main__':
    main()
