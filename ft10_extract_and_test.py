"""
FT10 — EXTRACT LABELS, VALIDATE, TEST PREDICTABILITY
====================================================
We established the precise JSON kill data produces CORRECT labels (FT5 precise
beat proxy 77-23 on independent kills@10 evidence). FT10 must use this data --
there is no 5-min-snapshot proxy for 10 kills.

This script does three things in sequence:

  STEP 1  Extract FT10 labels from kill_data/*.json (who hit 10 kills first,
          at what time). Mirrors the FT5 summarize() logic at index 9 instead
          of index 4. Flags ambiguous games (both hit 10 in the same 10s frame)
          and games that never reached 10 total kills for a side.

  STEP 2  Validate the labels the same way we validated FT5: on games where
          the team leading kills@10 is clear, does the FT10 winner usually
          match? (Weaker check than for FT5 -- 10th kill is later -- but a
          gross mismatch would reveal an extraction bug.)

  STEP 3  Predictability backtest: FT10 from FT5-STYLE features (champ
          aggression, team early rate, h2h, form -- NO kill_speed, the
          feature that overfit), precise labels, train<=2025 / test 2026.
          Also reports correlation with the WIN model's target, because if
          FT10 just mirrors "who wins", it's redundant, not a new market.

DECISION:
  - If FT10 backtests meaningfully above its baseline AND isn't ~fully
    redundant with the winner -> build the full pipeline.
  - If near-baseline or ~= winner -> FT10 isn't a viable separate product;
    stop before integrating.

Requires: kill_data/ (JSON files), proplay_matches.csv.
Run from the dataset builder folder.
"""

import warnings; warnings.filterwarnings('ignore')
import json, glob, os
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score

KILL_DIR = 'kill_data'
FORM_WINDOW = 8
H2H_CAP = 0.60
TARGET_LEAGUES = ['LCK','LPL','LEC','LCS','CBLOL','MSI','WLDs','EWC','LTA N','LTA S','LTA','FST']
AMBIG_FRAME_SECS = 10   # both teams' 10th kill within this window = ambiguous
TEAM_ALIASES = {'Team BDS':'Team Shifters','BDS':'Team Shifters'}
def norm(n): return TEAM_ALIASES.get(str(n).strip(), str(n).strip())
def cap_h2h(r): return max(1-H2H_CAP, min(H2H_CAP, r))
def weighted_form(hist,w):
    h=hist[-w:] if hist else []
    if not h: return 0.5
    wt=list(range(1,len(h)+1)); return sum(v*x for v,x in zip(h,wt))/sum(wt)

# =================================================================
# STEP 1 — extract FT10 labels from JSON
# =================================================================
print("="*66)
print("  STEP 1 — extracting FT10 labels from kill_data/*.json")
print("="*66)

if not os.path.isdir(KILL_DIR):
    raise SystemExit(f"'{KILL_DIR}/' not found. Run fetch_kills.py first.")

files = glob.glob(os.path.join(KILL_DIR,'*.json'))
print(f"  JSON files found: {len(files)}")

def nth_kill_time(kills, side, n):
    """t_secs of this side's n-th kill (1-indexed), or None."""
    s = [k for k in kills if k.get('side')==side]
    if len(s) >= n and s[n-1].get('t_secs') is not None:
        return s[n-1]['t_secs']
    return None

recs=[]
n_ok=n_no10=n_ambig=n_bad=0
for fp in files:
    try:
        with open(fp) as f: d=json.load(f)
    except Exception:
        n_bad+=1; continue
    kills = d.get('kills') or []
    gid = d.get('game_id') or os.path.splitext(os.path.basename(fp))[0]
    b10 = nth_kill_time(kills,'blue',10)
    r10 = nth_kill_time(kills,'red',10)
    # FT5 too (for validation cross-check + since we're already here)
    b5  = nth_kill_time(kills,'blue',5)
    r5  = nth_kill_time(kills,'red',5)
    # kills by 10 min, independent adjudicator
    bk10 = sum(1 for k in kills if k.get('side')=='blue' and (k.get('t_secs') or 1e9)<=600)
    rk10 = sum(1 for k in kills if k.get('side')=='red'  and (k.get('t_secs') or 1e9)<=600)

    if b10 is None and r10 is None:
        n_no10+=1; continue   # neither side reached 10 kills
    if b10 is not None and r10 is not None:
        ambiguous = 1 if abs(b10-r10) < AMBIG_FRAME_SECS else 0
        ft10 = 'blue' if b10 < r10 else 'red'
    elif b10 is not None:
        ambiguous=0; ft10='blue'
    else:
        ambiguous=0; ft10='red'
    if ambiguous: n_ambig+=1
    n_ok+=1
    recs.append({'game_id':gid,'league':d.get('league'),
                 'blue_team':norm(d.get('blue_team')),'red_team':norm(d.get('red_team')),
                 'first_to_ten':ft10,'blue_time10':b10,'red_time10':r10,'is_ambiguous':ambiguous,
                 'ft5': ('blue' if (b5 is not None and (r5 is None or b5<r5)) else ('red' if r5 is not None else None)),
                 'blue_kills10':bk10,'red_kills10':rk10})

ft10 = pd.DataFrame(recs)
print(f"  Reached 10 kills (usable):     {n_ok}")
print(f"  Never reached 10 (dropped):    {n_no10}")
print(f"  Ambiguous (both within {AMBIG_FRAME_SECS}s):  {n_ambig}")
print(f"  Unreadable files:              {n_bad}")
if len(ft10)==0:
    raise SystemExit("No usable FT10 games extracted.")

# attach date/picks/players/league from proplay — VIA THE BRIDGE MAP.
# JSON game_ids and proplay game_ids are different systems; game_id_map.csv
# links them and flags side-swaps (flip_label). We remap the JSON game_id to
# the proplay game_id, and where flip_label==1 we FLIP the FT10 winner so it's
# expressed in proplay's blue/red frame (the frame the champion features use).
if not os.path.exists('game_id_map.csv'):
    raise SystemExit("game_id_map.csv not found. Run build_gameid_bridge.py first.")
bridge = pd.read_csv('game_id_map.csv', dtype={'json_game_id':str, 'proplay_game_id':str})
print(f"\n  Bridge map rows: {len(bridge)}")
# JSON game_id is a string; ensure the join key matches type
ft10['game_id'] = ft10['game_id'].astype(str)
bridge['json_game_id'] = bridge['json_game_id'].astype(str)

ft10 = ft10.merge(bridge, left_on='game_id', right_on='json_game_id', how='inner')
print(f"  FT10 games linked via bridge: {len(ft10)}")
# apply the side-swap flip to the label
_flip = ft10['flip_label'] == 1
ft10.loc[_flip, 'first_to_ten'] = ft10.loc[_flip, 'first_to_ten'].map({'blue':'red','red':'blue'})
# also flip the kills@10 columns so the validation check stays in the same frame
ft10.loc[_flip, ['blue_kills10','red_kills10']] = ft10.loc[_flip, ['red_kills10','blue_kills10']].values
print(f"  Applied side-swap flip to {int(_flip.sum())} games.")

# now join proplay on the PROPLAY game_id
pp = pd.read_csv('proplay_matches.csv')
pp['game_id'] = pp['game_id'].astype(str)
pp['blue_team']=pp['blue_team'].apply(norm); pp['red_team']=pp['red_team'].apply(norm)
pp['date']=pd.to_datetime(pp['date'],errors='coerce'); pp['year']=pp['date'].dt.year
for c in ['blue_picks','red_picks','blue_players','red_players']:
    pp[c]=pp[c].apply(lambda x:[s.strip() for s in str(x).split(',')])
ft10 = ft10.drop(columns=[c for c in ['league','date','year'] if c in ft10.columns]).merge(
    pp[['game_id','date','year','league','blue_picks','red_picks','blue_players','red_players','blue_win']],
    left_on='proplay_game_id', right_on='game_id', how='inner', suffixes=('','_pp'))
print(f"  Merged with proplay (have picks): {len(ft10)}")
ft10 = ft10[ft10['league'].isin(TARGET_LEAGUES)].sort_values('date').reset_index(drop=True)
print(f"  In target leagues: {len(ft10)}")
print(f"  Blue FT10 rate: {(ft10['first_to_ten']=='blue').mean()*100:.1f}%")
print(f"\n  FT10 games by year (this determines the train/test split):")
_yc = ft10['year'].value_counts().sort_index()
for _yr,_n in _yc.items():
    _tag = '  <- TEST' if _yr==2026 else ('  <- train' if _yr<=2025 else '')
    print(f"    {int(_yr) if pd.notna(_yr) else '?'}: {_n}{_tag}")
_n_missing_year = ft10['year'].isna().sum()
if _n_missing_year:
    print(f"    (missing/unparseable date: {_n_missing_year} games)")

# =================================================================
# STEP 2 — validate FT10 labels against kills@10
# =================================================================
print("\n" + "="*66)
print("  STEP 2 — validating FT10 labels vs kills@10 evidence")
print("="*66)
v = ft10[ft10['is_ambiguous']==0].copy()
v['k10_leader']=np.where(v['blue_kills10']>v['red_kills10'],'blue',
                 np.where(v['red_kills10']>v['blue_kills10'],'red','tie'))
vv=v[v['k10_leader'].isin(['blue','red'])]
concord=(vv['first_to_ten']==vv['k10_leader']).mean()
print(f"  Games with clear kills@10 leader: {len(vv)}")
print(f"  FT10 winner matches kills@10 leader: {concord*100:.1f}%")
print("  (expect high but not perfect -- 10th kill comes after the 10-min mark")
print("   sometimes. A LOW number here would signal an extraction bug.)")

# =================================================================
# STEP 3 — predictability backtest (FT5-style features, no kill_speed)
# =================================================================
print("\n" + "="*66)
print("  STEP 3 — FT10 predictability (train<=2025, test 2026)")
print("="*66)
data = ft10[ft10['is_ambiguous']==0].copy()
data['y']=(data['first_to_ten']=='blue').astype(int)

cw=defaultdict(int); cg=defaultdict(int)
tw=defaultdict(int); tg=defaultdict(int)
h2h=defaultdict(lambda: defaultdict(int)); trec=defaultdict(list)
rows=[]
for _,row in data.iterrows():
    b,r=row['blue_team'],row['red_team']; bp,rp=row['blue_picks'],row['red_picks']; res=row['y']
    def cr(c):
        g=cg[c]; return cw[c]/g if g>0 else 0.5
    ba=sum(cr(c) for c in bp)/len(bp) if bp else 0.5
    ra=sum(cr(c) for c in rp)/len(rp) if rp else 0.5
    btg=tg[b]; rtg=tg[r]
    be=tw[b]/btg if btg>0 else 0.5; re=tw[r]/rtg if rtg>0 else 0.5
    mk=tuple(sorted([b,r])); hr=h2h[mk]; ht=sum(hr.values())
    h2h_r=cap_h2h(hr[b]/ht) if ht>0 else 0.5
    bf=weighted_form(trec[b],FORM_WINDOW); rf=weighted_form(trec[r],FORM_WINDOW)
    rows.append({'ba':ba,'ra':ra,'ad':ba-ra,'be':be,'re':re,'ed':be-re,
                 'h2h':h2h_r,'bf':bf,'rf':rf,'fmd':bf-rf})
    for c in bp: cg[c]+=1; cw[c]+=res
    for c in rp: cg[c]+=1; cw[c]+=(1-res)
    tg[b]+=1; tg[r]+=1; tw[b]+=res; tw[r]+=(1-res)
    h2h[mk][b]+=res; h2h[mk][r]+=(1-res)
    trec[b].append(res); trec[r].append(1-res)

feat=pd.DataFrame(rows)
tr=(data['year']<=2025).values; te=(data['year']==2026).values

# Fallback: if the year-based split leaves either side empty (e.g. the JSON
# kill data only covers 2026, or only older years), a temporal 2025/2026 split
# is impossible. Fall back to a chronological 80/20 split so we can still get a
# read, and say so loudly -- the result is then "can FT10 be predicted at all",
# not "out-of-sample 2026 accuracy".
_split_note = "train<=2025 / test=2026 (temporal holdout)"
if tr.sum()==0 or te.sum()==0:
    n=len(data); cut=int(n*0.8)
    tr=np.zeros(n,dtype=bool); te=np.zeros(n,dtype=bool)
    tr[:cut]=True; te[cut:]=True
    _split_note = f"chronological 80/20 split (data doesn't span the 2025/2026 boundary)"
    print(f"\n  NOTE: year-based split had an empty side. Falling back to:")
    print(f"        {_split_note}")

if tr.sum()<20 or te.sum()<10:
    print(f"\n  Train {int(tr.sum())} | Test {int(te.sum())} — TOO FEW to model reliably.")
    print(f"  You need more FT10 games (more JSON coverage across years).")
    print(f"  Extraction + validation above are still valid; only the backtest")
    print(f"  is blocked by sample size.")
    raise SystemExit(0)
mlb=MultiLabelBinarizer(); mlb.fit((data[tr]['blue_picks']+data[tr]['red_picks']).tolist())
be=pd.DataFrame(mlb.transform(data['blue_picks']),columns=['b_'+c for c in mlb.classes_]).reset_index(drop=True)
re=pd.DataFrame(mlb.transform(data['red_picks']),columns=['r_'+c for c in mlb.classes_]).reset_index(drop=True)
X=pd.concat([be,re,feat.reset_index(drop=True)],axis=1); y=data['y'].reset_index(drop=True)
print(f"  Split: {_split_note}")
print(f"  Train {tr.sum()} | Test {te.sum()} | Features {X.shape[1]}")
m=CalibratedClassifierCV(GradientBoostingClassifier(n_estimators=125,max_depth=1,learning_rate=0.03,random_state=42),method='isotonic',cv=5)
m.fit(X[tr],y[tr])
p=m.predict_proba(X[te])[:,1]; yt=y[te].reset_index(drop=True)
acc=accuracy_score(yt,(p>=0.5).astype(int)); auc=roc_auc_score(yt,p)
base=max(yt.mean(),1-yt.mean())
print(f"\n  FT10 accuracy: {acc*100:.2f}%   AUC {auc:.4f}")
print(f"  Always-majority baseline: {base*100:.2f}%   edge {(acc-base)*100:+.2f}%")

# redundancy with the winner: how often does FT10 == game winner?
if 'blue_win' in data.columns:
    dm = data.dropna(subset=['blue_win'])
    same = (dm['y']==dm['blue_win'].astype(int)).mean()
    print(f"\n  FT10 == game winner: {same*100:.1f}% of games")
    print("  (if very high, FT10 is largely redundant with the win model)")

print("\n" + "="*66)
print("  DECISION")
print("="*66)
edge=(acc-base)
if te.sum()<20:
    print("  Test sample too small to decide -- need more 2026 FT10 games.")
elif edge < 0.02:
    print("  FT10 is ~baseline -- not predictable from pre-game features.")
    print("  NOT worth building as a product.")
else:
    print(f"  FT10 beats baseline by {edge*100:.1f}%. Predictable to some degree.")
    print("  Check redundancy above: if FT10==winner is very high (>~80%), it's")
    print("  mostly the win model in disguise. If moderate, it's a real market.")
    print("  If both look good -> build the full FT10 pipeline.")
print("="*66)
