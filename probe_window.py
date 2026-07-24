"""
probe_window.py — how much data does ONE window request actually return?

The scraper currently keeps only the LAST frame of each response and advances
the cursor by 10 seconds, so it re-requests the same window ~60x over. If each
response actually contains many frames spanning several minutes, we can use all
of them and step the cursor to the END of the window instead -- cutting requests
by roughly an order of magnitude.

This probes one real game and reports frames-per-response and the time span.
"""
import json, glob, os, sys
from datetime import datetime
import requests

FEED = "https://feed.lolesports.com/livestats/v1"

# grab a game_id we know has data: pick one from an existing kill_data JSON
gid = None
for fp in glob.glob('kill_data/*.json')[:50]:
    try:
        d = json.load(open(fp))
        if d.get('kills'):
            gid = d.get('game_id') or os.path.splitext(os.path.basename(fp))[0]
            break
    except Exception:
        continue
if not gid:
    print("No usable game found in kill_data/. Pass a game_id as an argument.")
    if len(sys.argv) > 1:
        gid = sys.argv[1]
    else:
        raise SystemExit(1)

print(f"Probing game_id: {gid}")
r = requests.get(f"{FEED}/window/{gid}", timeout=25)
print(f"  HTTP {r.status_code}")
if r.status_code != 200:
    raise SystemExit("Non-200; try another game_id.")

d = r.json()
frames = d.get('frames', []) or []
print(f"\n  FRAMES RETURNED IN ONE RESPONSE: {len(frames)}")
if frames:
    t0 = frames[0].get('rfc460Timestamp')
    t1 = frames[-1].get('rfc460Timestamp')
    print(f"  first frame ts: {t0}")
    print(f"  last  frame ts: {t1}")
    try:
        span = (datetime.fromisoformat(t1.replace('Z','+00:00'))
                - datetime.fromisoformat(t0.replace('Z','+00:00'))).total_seconds()
        print(f"  TIME SPAN COVERED: {span:.0f} seconds ({span/60:.1f} minutes)")
        if len(frames) > 1:
            print(f"  frame interval: {span/(len(frames)-1):.1f}s")
    except Exception as e:
        print(f"  (couldn't compute span: {e})")
    # kill totals across the window — shows how much progress one call gives
    b0 = (frames[0].get('blueTeam',{}) or {}).get('totalKills',0)
    r0 = (frames[0].get('redTeam',{}) or {}).get('totalKills',0)
    b1 = (frames[-1].get('blueTeam',{}) or {}).get('totalKills',0)
    r1 = (frames[-1].get('redTeam',{}) or {}).get('totalKills',0)
    print(f"  kills at window start: blue {b0}, red {r0}")
    print(f"  kills at window end:   blue {b1}, red {r1}")

print("\n  => If FRAMES RETURNED is much greater than 1, the scraper is")
print("     discarding almost every frame and re-requesting the same data.")
print("     Stepping the cursor by the full span instead of 10s would cut")
print("     requests by roughly (frames per response)x.")
