"""
build_precise_labels.py — precise FT5 + FT10 labels from JSON kill data
=======================================================================
Replaces the Oracle 5-min-snapshot proxy (kill_timelines.csv) with PRECISE
labels from the 10-second lolesports kill data. We proved (kills@10 adjudication,
77-23) that precise labels are CORRECT and the proxy is wrong ~27% of the time.

Emits BOTH thresholds in one file so FT5 and FT10 train from the same
trustworthy source:
  first_to_five / ft5_ambiguous   (blue[4] vs red[4])
  first_to_ten  / ft10_ambiguous  (blue[9] vs red[9])

Labels are reconciled to proplay's blue/red frame via the bridge's flip_label
(teams swap sides between series games -- the polarity trap). Picks/players
come from proplay so they line up with the reconciled labels.

OUTPUT: precise_labels.csv with columns matching what train_and_save.py needs:
  game_id (proplay), tournament (=league), blue_team, red_team,
  blue_picks, red_picks, blue_players, red_players,
  first_to_five, blue_time, red_time, is_ambiguous  (FT5, proxy-compatible names)
  first_to_ten, blue_time10, red_time10, ft10_ambiguous

NOTE: 'is_ambiguous' / 'first_to_five' / 'blue_time' keep the proxy's column
names so the existing FT5 code reads this file with zero changes when the
source switch is flipped to 'precise'.

Requires: kill_data/*.json, game_id_map.csv (bridge), proplay_matches.csv
Run from the dataset builder folder (after build_gameid_bridge.py).
"""

import json, glob, os
import pandas as pd
import numpy as np

KILL_DIR = 'kill_data'
AMBIG_SECS = 10        # both sides' Nth kill within this window = ambiguous
TEAM_ALIASES = {'Team BDS':'Team Shifters','BDS':'Team Shifters'}
def norm(n): return TEAM_ALIASES.get(str(n).strip(), str(n).strip())

print("="*66)
print("  BUILD PRECISE LABELS (FT5 + FT10) from JSON kill data")
print("="*66)

for req in (KILL_DIR, 'game_id_map.csv', 'proplay_matches.csv'):
    if not os.path.exists(req):
        raise SystemExit(f"Missing '{req}'. Run fetch_kills.py + build_gameid_bridge.py first.")

def nth_kill_time(kills, side, n):
    s = [k for k in kills if k.get('side')==side]
    if len(s) >= n and s[n-1].get('t_secs') is not None:
        return s[n-1]['t_secs']
    return None

# -----------------------------------------------------------------
# 1. extract both thresholds from every JSON
# -----------------------------------------------------------------
files = glob.glob(os.path.join(KILL_DIR,'*.json'))
print(f"  JSON files: {len(files)}")
recs=[]
for fp in files:
    try:
        with open(fp) as f: d=json.load(f)
    except Exception:
        continue
    kills = d.get('kills') or []
    gid = str(d.get('game_id') or os.path.splitext(os.path.basename(fp))[0])
    b5,r5   = nth_kill_time(kills,'blue',5),  nth_kill_time(kills,'red',5)
    b10,r10 = nth_kill_time(kills,'blue',10), nth_kill_time(kills,'red',10)
    bk10 = sum(1 for k in kills if k.get('side')=='blue' and (k.get('t_secs') or 1e9)<=600)
    rk10 = sum(1 for k in kills if k.get('side')=='red'  and (k.get('t_secs') or 1e9)<=600)

    # FT5
    if b5 is None and r5 is None:
        ft5=None; ft5_amb=1; bt5=rt5=None
    else:
        if b5 is not None and r5 is not None:
            ft5_amb = 1 if abs(b5-r5) < AMBIG_SECS else 0
            ft5 = 'blue' if b5 < r5 else 'red'
        elif b5 is not None: ft5='blue'; ft5_amb=0
        else: ft5='red'; ft5_amb=0
        bt5, rt5 = (b5/60 if b5 else None), (r5/60 if r5 else None)

    # FT10
    if b10 is None and r10 is None:
        ft10=None; ft10_amb=1; bt10=rt10=None
    else:
        if b10 is not None and r10 is not None:
            ft10_amb = 1 if abs(b10-r10) < AMBIG_SECS else 0
            ft10 = 'blue' if b10 < r10 else 'red'
        elif b10 is not None: ft10='blue'; ft10_amb=0
        else: ft10='red'; ft10_amb=0
        bt10, rt10 = (b10/60 if b10 else None), (r10/60 if r10 else None)

    recs.append({'json_game_id':gid,
                 'first_to_five':ft5,'blue_time':bt5,'red_time':rt5,'is_ambiguous':ft5_amb,
                 'first_to_ten':ft10,'blue_time10':bt10,'red_time10':rt10,'ft10_ambiguous':ft10_amb,
                 'blue_kills10':bk10,'red_kills10':rk10})
lab = pd.DataFrame(recs)
print(f"  Extracted labels from {len(lab)} games")
print(f"    FT5 usable:  {lab['first_to_five'].notna().sum()}  (ambiguous {int((lab['is_ambiguous']==1).sum())})")
print(f"    FT10 usable: {lab['first_to_ten'].notna().sum()}  (ambiguous {int((lab['ft10_ambiguous']==1).sum())})")

# -----------------------------------------------------------------
# 2. join through the bridge, apply flip to BOTH labels
# -----------------------------------------------------------------
bridge = pd.read_csv('game_id_map.csv', dtype={'json_game_id':str,'proplay_game_id':str})
bridge['json_game_id'] = bridge['json_game_id'].astype(str)
lab['json_game_id'] = lab['json_game_id'].astype(str)
lab = lab.merge(bridge, on='json_game_id', how='inner')
print(f"  Linked via bridge: {len(lab)}")

flip = lab['flip_label']==1
swap = {'blue':'red','red':'blue'}
for col in ('first_to_five','first_to_ten'):
    lab.loc[flip, col] = lab.loc[flip, col].map(swap)
# swap the side-specific time + kill columns too, to stay in proplay frame
lab.loc[flip, ['blue_time','red_time']]     = lab.loc[flip, ['red_time','blue_time']].values
lab.loc[flip, ['blue_time10','red_time10']] = lab.loc[flip, ['red_time10','blue_time10']].values
lab.loc[flip, ['blue_kills10','red_kills10']] = lab.loc[flip, ['red_kills10','blue_kills10']].values
print(f"  Applied side-swap flip to {int(flip.sum())} games")

# -----------------------------------------------------------------
# 3. attach proplay picks/players/teams/league (the model's frame)
# -----------------------------------------------------------------
pp = pd.read_csv('proplay_matches.csv')
pp['game_id'] = pp['game_id'].astype(str)
pp['blue_team'] = pp['blue_team'].apply(norm)
pp['red_team']  = pp['red_team'].apply(norm)
keep = ['game_id','league','blue_team','red_team','blue_picks','red_picks','blue_players','red_players']
# bridge already carried a 'league' column; drop it so the proplay merge's
# 'league' lands cleanly (avoids league_x/league_y suffixing).
lab = lab.drop(columns=[c for c in ['league'] if c in lab.columns])
out = lab.merge(pp[keep], left_on='proplay_game_id', right_on='game_id', how='inner')
print(f"  Joined proplay picks: {len(out)}")

# shape to what train_and_save expects (tournament = league; game_id = proplay id)
out = out.drop(columns=['game_id'])            # proplay's game_id col; use proplay_game_id instead
out = out.rename(columns={'league':'tournament'})
out['game_id'] = out['proplay_game_id']
cols = ['game_id','tournament','blue_team','red_team',
        'blue_picks','red_picks','blue_players','red_players',
        'first_to_five','blue_time','red_time','is_ambiguous',
        'first_to_ten','blue_time10','red_time10','ft10_ambiguous',
        'blue_kills10','red_kills10']
out = out[cols]

# drop rows with no FT5 label at all (keep FT10-only? no -- FT5 model needs FT5).
# We keep every row; each model filters its own ambiguous/missing at train time.
out.to_csv('precise_labels.csv', index=False)
print(f"\n  Wrote precise_labels.csv: {len(out)} rows")
print(f"    FT5 non-ambiguous:  {int(((out['is_ambiguous']==0) & out['first_to_five'].notna()).sum())}")
print(f"    FT10 non-ambiguous: {int(((out['ft10_ambiguous']==0) & out['first_to_ten'].notna()).sum())}")
print(f"    Blue FT5 rate:  {(out['first_to_five']=='blue').mean()*100:.1f}%")
print(f"    Blue FT10 rate: {(out['first_to_ten']=='blue').mean()*100:.1f}%")
print("="*66)
