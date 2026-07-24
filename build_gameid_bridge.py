"""
build_gameid_bridge.py — map lolesports JSON game_ids to proplay game_ids
=========================================================================
JSON kill files use lolesports platform IDs; proplay uses Oracle's Elixir IDs;
no shared key. game_ids.csv bridges them: lolesports game_id + a clean team
ABBREVIATION (JDG, GEN, T1) + full names + start_time + league + game_number.

Full-name matching failed (LPL "Beijing JDG Esports" vs proplay "JD Gaming").
Abbreviations are the clean key -- verified essentially unique (only GX
collides: GIANTX/LEC vs GIANTX ITERO/LES, resolved by including league; ITERO
is LES, a non-target league, filtered out anyway).

APPROACH
  1. LEARN (abbr, league) -> proplay_team from co-occurrence on matching dates.
     Self-validating; printed for review.
  2. JOIN JSON -> game_ids (exact game_id) -> proplay (date +-1d, learned
     team-pair, game_number).
  3. DETECT side-swaps -> flip_label (reconcile FT10 label to proplay blue/red).
  4. REPORT match rate, learned map, swap %, unresolved.

OUTPUT: game_id_map.csv (json_game_id, proplay_game_id, flip_label, league)
Requires: game_ids.csv, proplay_matches.csv, kill_data/*.json
"""

import json, glob, os, re
from collections import defaultdict, Counter
from datetime import timedelta
import pandas as pd

def basic(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower()) if s is not None else ''

def abbr_matches(abbr, team):
    """Does abbr look like an acronym/prefix of a proplay team name?
    e.g. JDG ~ 'JD Gaming' (acronym), T1 ~ 'T1' (prefix)."""
    a = re.sub(r'[^a-z0-9]', '', str(abbr).lower())
    if not a: return False
    words = re.sub(r'[^a-z0-9 ]', ' ', str(team).lower()).split()
    if not words: return False
    acro = ''.join(w[0] for w in words)
    joined = ''.join(words)
    return a == acro or joined.startswith(a) or acro.startswith(a) or a in joined

# =================================================================
# Load
# =================================================================
print("="*70)
print("  GAME-ID BRIDGE (abbreviation-based)")
print("="*70)

gi = pd.read_csv('game_ids.csv', dtype=str)
gi['game_number'] = pd.to_numeric(gi['game_number'], errors='coerce')
gi['start_dt'] = pd.to_datetime(gi['start_time'], errors='coerce', utc=True)
gi['gdate'] = gi['start_dt'].dt.date
json_ids = set(os.path.splitext(os.path.basename(p))[0] for p in glob.glob('kill_data/*.json'))
gi['has_json'] = gi['game_id'].isin(json_ids)

pp = pd.read_csv('proplay_matches.csv')
pp['date'] = pd.to_datetime(pp['date'], errors='coerce')
pp['pdate'] = pp['date'].dt.date

# proplay uses TWO game_id formats:
#   - LPL:  '9691-9691_game_2'  -> game number is in the suffix
#   - all others: 'LOLTMNT02_414569' / 'ESPORTSTMNT04_...' -> NO suffix.
# For the suffix-less format, game order within a series is chronological
# (verified: trailing id + timestamp both increment through a Bo3/Bo5). So we
# derive game_number by sorting games within each (date, team-pair) group by
# time and numbering 1,2,3... This is what unblocks every non-LPL league.
def _suffix_num(gid):
    m = re.search(r'_game_(\d+)$', str(gid)); return int(m.group(1)) if m else None
pp['game_number'] = pp['game_id'].apply(_suffix_num)

# unordered team-pair key for grouping a series
pp['_pair'] = pp.apply(lambda r: tuple(sorted([basic(r['blue_team']), basic(r['red_team'])])), axis=1)

# fill missing game_number chronologically within (date, pair)
_missing = pp['game_number'].isna()
if _missing.any():
    # sort so cumcount follows real game order; secondary sort on game_id's
    # trailing number as a tiebreak when timestamps are identical
    pp['_idnum'] = pp['game_id'].apply(lambda g: int(m.group(1)) if (m:=re.search(r'(\d+)$', str(g))) else 0)
    pp_sorted = pp.sort_values(['pdate','_pair','date','_idnum'])
    derived = pp_sorted.groupby(['pdate','_pair']).cumcount() + 1
    pp.loc[pp_sorted.index[_missing.loc[pp_sorted.index].values], 'game_number'] = \
        derived[_missing.loc[pp_sorted.index].values].values
    pp['game_number'] = pp['game_number'].astype('Int64')
    print(f"  Derived game_number for {int(_missing.sum())} suffix-less proplay rows "
          f"(chronological within series).")

print(f"  game_ids rows: {len(gi)} | with JSON: {gi['has_json'].sum()} | proplay rows: {len(pp)}")

giJ = gi[gi['has_json'] & gi['game_number'].notna()].copy()

# =================================================================
# STEP 1 — learn (abbr, league) -> proplay_team
# =================================================================
print("\n" + "-"*70)
print("  STEP 1: learning (abbreviation, league) -> proplay team")
print("-"*70)

pp_by_date_league = defaultdict(set)
for _, r in pp.iterrows():
    if r['pdate'] is not None:
        k = (r['pdate'], str(r['league']))
        pp_by_date_league[k].add(r['blue_team']); pp_by_date_league[k].add(r['red_team'])

votes = defaultdict(Counter)   # (abbr, league) -> Counter(proplay_team)
for _, g in giJ.iterrows():
    if g['gdate'] is None: continue
    cands = set()
    for delta in (-1,0,1):
        cands |= pp_by_date_league.get((g['gdate']+timedelta(days=delta), str(g['league'])), set())
    for abbr, full in [(g['blue_team'], g['blue_name']), (g['red_team'], g['red_name'])]:
        key = (str(abbr), str(g['league']))
        bfull = basic(full)
        ftoks = set(re.sub(r'[^a-z0-9 ]',' ',str(full).lower()).split())
        for ppt in cands:
            bppt = basic(ppt); score = 0
            ptoks = set(re.sub(r'[^a-z0-9 ]',' ',str(ppt).lower()).split())
            if bppt and bfull and bppt == bfull:
                score = 6                                   # exact (normalized) name
            elif bppt and bfull and (bppt in bfull or bfull in bppt):
                score = 4                                   # substring either way
            else:
                # token overlap: how many words do the names share?
                shared = ftoks & ptoks
                # ignore generic filler words that don't identify a team
                shared -= {'esports','esport','gaming','team','the','red','force'}
                if len(shared) >= 1:
                    score = 3
                elif abbr_matches(abbr, ppt):
                    score = 2
            if score: votes[key][ppt] += score

# resolve each (abbr,league) to its top-voted proplay team
abbr_map = {}
weak = []
for key, ctr in votes.items():
    if not ctr: continue
    top, topn = ctr.most_common(1)[0]
    total = sum(ctr.values())
    abbr_map[key] = top
    if topn/total < 0.6:   # low consensus -> flag for review
        weak.append((key, dict(ctr.most_common(3))))

print(f"  Learned {len(abbr_map)} (abbr,league) mappings.")
print("  Sample (first 25):")
for key in sorted(abbr_map.keys())[:25]:
    print(f"    {key[0]:<6} [{key[1]:<6}] -> {abbr_map[key]}")
if weak:
    print(f"\n  LOW-CONSENSUS mappings to eyeball ({len(weak)}):")
    for key, top3 in weak[:15]:
        print(f"    {key} -> {top3}")

def resolve(abbr, league):
    return abbr_map.get((str(abbr), str(league)))

# =================================================================
# STEP 2/3 — join + side-swap detection
# =================================================================
print("\n" + "-"*70)
print("  STEP 2: joining through the learned map")
print("-"*70)

# index proplay by (unordered proplay-team pair, game_number) -> rows
pp_index = defaultdict(list)
for idx, r in pp.iterrows():
    pair = tuple(sorted([basic(r['blue_team']), basic(r['red_team'])]))
    pp_index[(pair, r['game_number'])].append(r)

records=[]; unmatched=[]; ambiguous=[]; unresolved_abbr=Counter()
for _, g in giJ.iterrows():
    bt = resolve(g['blue_team'], g['league'])
    rt = resolve(g['red_team'], g['league'])
    if bt is None or rt is None:
        if bt is None: unresolved_abbr[(str(g['blue_team']), str(g['league']))]+=1
        if rt is None: unresolved_abbr[(str(g['red_team']), str(g['league']))]+=1
        unmatched.append((g['game_id'], g['league'], str(g['gdate']), g['blue_name'], g['red_name']))
        continue
    pair = tuple(sorted([basic(bt), basic(rt)]))
    gn = int(g['game_number'])
    cands = pp_index.get((pair, gn), [])
    if g['gdate'] is not None:
        cands = [c for c in cands if c['pdate'] is not None and abs((c['pdate']-g['gdate']).days)<=1]
    if len(cands)==0:
        unmatched.append((g['game_id'], g['league'], str(g['gdate']), g['blue_name'], g['red_name']))
        continue
    if len(cands)>1:
        ambiguous.append((g['game_id'], pair, gn, len(cands))); continue
    c = cands[0]
    # side-swap: is game_ids' blue the same as proplay's blue?
    if basic(bt) == basic(c['blue_team']):
        flip = 0
    elif basic(bt) == basic(c['red_team']):
        flip = 1
    else:
        unmatched.append((g['game_id'], g['league'], str(g['gdate']), g['blue_name'], g['red_name'])); continue
    records.append({'json_game_id':g['game_id'], 'proplay_game_id':c['game_id'],
                    'flip_label':flip, 'league':g['league']})

mp = pd.DataFrame(records)

# =================================================================
# REPORT
# =================================================================
print("\n" + "="*70)
print("  RESULT")
print("="*70)
total = len(giJ)
print(f"  JSON-backed games:        {total}")
print(f"  Cleanly mapped:           {len(mp)}  ({len(mp)/max(total,1)*100:.1f}%)")
print(f"  Unmatched:                {len(unmatched)}")
print(f"  Ambiguous (>1 candidate): {len(ambiguous)}")
if len(mp):
    print(f"  Side-swapped (flip=1):    {mp['flip_label'].sum()} ({mp['flip_label'].mean()*100:.0f}%)")
    print("\n  Mapped by league:")
    for lg,n in mp['league'].value_counts().items():
        print(f"    {lg:<8} {n}")
if unresolved_abbr:
    print(f"\n  UNRESOLVED (abbr, league) — no proplay team learned ({len(unresolved_abbr)}):")
    for (ab,lg),n in unresolved_abbr.most_common(20):
        print(f"    {n:>4}x  {ab} [{lg}]")
if unmatched:
    print(f"\n  Sample unmatched (first 8 of {len(unmatched)}):")
    for gid,lg,dt,bn,rn in unmatched[:8]:
        print(f"    {gid} | {lg} | {dt} | {bn} vs {rn}")

if len(mp):
    mp.to_csv('game_id_map.csv', index=False)
    print(f"\n  Wrote game_id_map.csv ({len(mp)} rows).")
print("\n" + "="*70)
print("  Check: match rate high? side-swap ~40-50%? learned map sane?")
print("  Unresolved abbreviations tell us which teams need attention.")
print("="*70)
