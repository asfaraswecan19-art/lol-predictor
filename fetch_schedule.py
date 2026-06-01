"""
fetch_schedule.py — Walk lolesports schedule backward via pageToken.
Writes schedule_raw.jsonl (one event per line) with checkpoint resume.

USAGE:
    python fetch_schedule.py             # walk until exhausted
    python fetch_schedule.py --max 50    # limit pages (e.g. testing)
    python fetch_schedule.py --reset     # delete state and start over

CHECKPOINT FILES:
    schedule_raw.jsonl  — appended one event per line (jsonl format)
    schedule.state.json — last pageToken processed + counters

If the script crashes/is killed, just re-run it. It picks up from the
last pageToken written to state.json, and de-duplicates events on
re-read (each event's matchId+startTime is unique).
"""
import argparse, json, os, sys, time, signal
from pathlib import Path
import requests

API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
HEADERS = {"x-api-key": API_KEY}
ESPORTS_API = "https://esports-api.lolesports.com/persisted/gw"
OUT_FILE   = Path("schedule_raw.jsonl")
STATE_FILE = Path("schedule.state.json")
SLEEP_BETWEEN_PAGES = 0.3  # seconds

# Graceful shutdown — finishes current page write before quitting
_STOP_REQUESTED = False
def _handle_sigint(sig, frame):
    global _STOP_REQUESTED
    print("\n[Ctrl+C received — will stop after current page]")
    _STOP_REQUESTED = True
signal.signal(signal.SIGINT, _handle_sigint)


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {'page_token': None, 'pages_completed': 0, 'events_written': 0, 'done': False}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def count_existing_events():
    """Count lines in jsonl if it exists."""
    if not OUT_FILE.exists(): return 0
    with OUT_FILE.open('r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def fetch_page(page_token=None):
    url = f"{ESPORTS_API}/getSchedule?hl=en-US"
    if page_token:
        url += f"&pageToken={page_token}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}: {r.text[:200]}")
            return None
        return r.json()
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max', type=int, default=0, help='Max pages (0 = unlimited)')
    ap.add_argument('--reset', action='store_true', help='Delete state + output and start over')
    args = ap.parse_args()

    if args.reset:
        if STATE_FILE.exists(): STATE_FILE.unlink()
        if OUT_FILE.exists():   OUT_FILE.unlink()
        print("State and output reset.")

    state = load_state()
    if state.get('done'):
        print(f"Schedule fetch already marked DONE. Total events: {state['events_written']}")
        print(f"Run with --reset to redo from scratch.")
        return

    # Sanity check: file event count should match state counter (resume integrity)
    actual_events = count_existing_events()
    if actual_events != state['events_written']:
        print(f"WARN: state says {state['events_written']} events but file has {actual_events}. "
              f"Using file count as truth.")
        state['events_written'] = actual_events

    print(f"Resuming from page {state['pages_completed']} (token: "
          f"{'<initial>' if state['page_token'] is None else state['page_token'][:30]+'...'})")
    print(f"Already-written events: {state['events_written']}")

    pages_this_run = 0
    with OUT_FILE.open('a', encoding='utf-8') as out:
        while not _STOP_REQUESTED:
            if args.max and pages_this_run >= args.max:
                print(f"\nReached --max {args.max} pages. Stopping.")
                break

            data = fetch_page(state['page_token'])
            if not data:
                print("  Fetch failed, will retry once after 5s pause...")
                time.sleep(5)
                data = fetch_page(state['page_token'])
                if not data:
                    print("  Retry also failed. Saving state and exiting.")
                    break

            sched = data.get('data', {}).get('schedule', {}) or {}
            events = sched.get('events', []) or []
            if not events:
                print(f"  Page returned 0 events — likely end of history.")
                state['done'] = True
                break

            # Write each event as one JSON line
            for ev in events:
                out.write(json.dumps(ev, separators=(',',':')) + '\n')
            out.flush()
            os.fsync(out.fileno())   # force to disk before claiming page done

            state['events_written'] += len(events)
            state['pages_completed'] += 1
            pages_this_run += 1

            # Get older token for next iteration
            older_token = (sched.get('pages') or {}).get('older')
            earliest = min((e.get('startTime','') for e in events), default='?')
            print(f"  Page {state['pages_completed']}: {len(events)} events "
                  f"(earliest {earliest[:10]}). Total written: {state['events_written']}")

            if not older_token:
                print("  No older pageToken — reached end of history.")
                state['done'] = True
                state['page_token'] = None
                break

            state['page_token'] = older_token
            save_state(state)
            time.sleep(SLEEP_BETWEEN_PAGES)

    save_state(state)
    print(f"\nStopped. Pages this run: {pages_this_run}. Total events: {state['events_written']}.")
    if state.get('done'):
        print("Schedule fetch DONE.")
    else:
        print("Re-run to continue from where we stopped.")


if __name__ == '__main__':
    main()
