"""
fetch_game_ids.py — For each match in schedule_raw.jsonl, fetch its
game IDs via getEventDetails. Writes game_ids.csv with checkpointing.

USAGE:
    python fetch_game_ids.py             # process all matches
    python fetch_game_ids.py --reset     # delete output + state and redo
    python fetch_game_ids.py --tier1     # only process LCK/LEC/LCS/LPL/LCP/CBLOL/MSI/WLDs/FST

OUTPUT:
    game_ids.csv — columns: match_id, game_id, game_number, state,
                            start_time, league, blue_team, red_team
"""
import argparse, csv, json, os, signal, time
from pathlib import Path
import requests

API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
HEADERS = {"x-api-key": API_KEY}
ESPORTS_API = "https://esports-api.lolesports.com/persisted/gw"
IN_FILE   = Path("schedule_raw.jsonl")
OUT_FILE  = Path("game_ids.csv")
STATE_FILE = Path("game_ids.state.json")
SLEEP = 0.25  # gentler than schedule because per-call is cheap

TIER1_LEAGUES = {'LCK','LPL','LEC','LCS','LCP','CBLOL','MSI','WLDs','Worlds',
                 'FST','LTA N','LTA S','LTA','EMEA Masters'}

_STOP_REQUESTED = False
def _handle_sigint(sig, frame):
    global _STOP_REQUESTED
    print("\n[Ctrl+C — finishing current match then stopping]")
    _STOP_REQUESTED = True
signal.signal(signal.SIGINT, _handle_sigint)


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {'completed_match_ids': []}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def get_event_details(match_id):
    url = f"{ESPORTS_API}/getEventDetails?hl=en-US&id={match_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reset', action='store_true')
    ap.add_argument('--tier1', action='store_true', help='Only tier-1 leagues')
    args = ap.parse_args()

    if not IN_FILE.exists():
        print(f"ERROR: {IN_FILE} not found. Run fetch_schedule.py first.")
        return

    if args.reset:
        for f in [OUT_FILE, STATE_FILE]:
            if f.exists(): f.unlink()
        print("Reset.")

    state = load_state()
    completed = set(state.get('completed_match_ids', []))
    print(f"Already-completed matches: {len(completed)}")

    # Load all matches from schedule
    matches = []
    with IN_FILE.open('r', encoding='utf-8') as f:
        for line in f:
            ev = json.loads(line)
            if ev.get('state') != 'completed':
                continue  # skip ongoing/upcoming
            m = ev.get('match') or {}
            mid = m.get('id')
            if not mid:
                continue
            league = (ev.get('league') or {}).get('name','?')
            if args.tier1 and league not in TIER1_LEAGUES:
                continue
            teams = m.get('teams', [])
            if len(teams) < 2:
                continue
            matches.append({
                'match_id': mid,
                'start_time': ev.get('startTime',''),
                'league': league,
                'blue_team': teams[0].get('code',''),
                'red_team': teams[1].get('code',''),
                'blue_name': teams[0].get('name',''),
                'red_name': teams[1].get('name',''),
            })

    # De-duplicate match_ids (same match can appear if schedule pages overlap)
    seen = set(); unique_matches = []
    for m in matches:
        if m['match_id'] not in seen:
            seen.add(m['match_id']); unique_matches.append(m)
    matches = unique_matches
    print(f"Unique completed matches to process: {len(matches)}")

    # Filter to remaining
    remaining = [m for m in matches if m['match_id'] not in completed]
    print(f"Remaining after checkpoint: {len(remaining)}")

    # Open output (append mode, write header if new)
    write_header = not OUT_FILE.exists() or OUT_FILE.stat().st_size == 0
    with OUT_FILE.open('a', encoding='utf-8', newline='') as out_f:
        writer = csv.writer(out_f)
        if write_header:
            writer.writerow(['match_id','game_id','game_number','state',
                             'start_time','league','blue_team','red_team',
                             'blue_name','red_name'])

        for i, m in enumerate(remaining, 1):
            if _STOP_REQUESTED: break

            details = get_event_details(m['match_id'])
            if not details:
                print(f"  [{i}/{len(remaining)}] {m['league']:10} {m['blue_team']}/{m['red_team']}: details fetch FAILED, skipping")
                time.sleep(1)
                continue

            games = ((details.get('data') or {}).get('event') or {}).get('match', {}).get('games', [])
            wrote = 0
            for g in games:
                gid = g.get('id')
                if not gid: continue
                writer.writerow([m['match_id'], gid, g.get('number',''), g.get('state',''),
                                 m['start_time'], m['league'], m['blue_team'], m['red_team'],
                                 m['blue_name'], m['red_name']])
                wrote += 1
            out_f.flush()
            os.fsync(out_f.fileno())

            completed.add(m['match_id'])
            if i % 25 == 0:
                state['completed_match_ids'] = list(completed)
                save_state(state)
                print(f"  [{i}/{len(remaining)}] {m['league']:10} {m['blue_team']}/{m['red_team']}: {wrote} games. Checkpoint saved.")

            time.sleep(SLEEP)

    # Final checkpoint
    state['completed_match_ids'] = list(completed)
    save_state(state)
    print(f"\nDone. {len(completed)} matches processed. Output: {OUT_FILE}")


if __name__ == '__main__':
    main()
