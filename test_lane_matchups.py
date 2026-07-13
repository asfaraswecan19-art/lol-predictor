"""
STAGE 1 — LANE-SEPARATED MATCHUP DIFFS  (standalone A/B backtest)
================================================================
Hypothesis: giving the model FIVE separate per-lane matchup signals
beats mushing all five champions into one averaged 'winrate_diff'.

Each lane diff = blue_lane_role_champ_rate - red_lane_role_champ_rate,
using the SAME shrunk role_champ_rate the production model already uses.
No new data, no archetype labels. This is a pure re-slicing test.

We compare three feature sets on the identical 2026 holdout:
  A) V8 baseline                 (current production feature set)
  B) V8 + 5 lane diffs           (does lane granularity add signal?)
  C) lane diffs ALONE            (do they carry standalone signal at all?)

Plus we count 'losing lanes' and print its correlation with the outcome
as a preview of Stage 2 (not added to the model yet).

Train 2023-2025, test 2026 — mirrors backtester.py exactly.
Run from the dataset builder folder. Paste the output back.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import glob, os
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ---- config (matches backtester.py) ----
DATA_FILE='proplay_matches.csv'; FT5_DATA='kill_timelines.csv'
FORM_WINDOW=8; RECENT_WINDOW=20; GOLD_WINDOW=15
BLUE_SIDE_WINRATE=0.5312; MIN_ROLE_GAMES=5; MIN_PC_GAMES=12
PC_WEIGHT=0.10; RC_WEIGHT=0.90; H2H_CAP=0.60
POSITIONS=['top','jng','mid','adc','sup']
TARGET_LEAGUES=['LCK','LPL','LEC','LCS','CBLOL','MSI','WLDs','LTA N','LTA S','LTA','FST']
K_CHAMP=8; K_ROLE=MIN_ROLE_GAMES; K_PC=MIN_PC_GAMES
TEAM_ALIASES={'Team BDS':'Team Shifters','BDS':'Team Shifters'}
def normalize_team(n): return TEAM_ALIASES.get(str(n),str(n))
def cap_h2h(r): return max(1-H2H_CAP, min(H2H_CAP, r))
def weighted_form(hist,w):
    h=hist[-w:] if hist else []
    if not h: return 0.5
    wts=list(range(1,len(h)+1)); return sum(v*x for v,x in zip(h,wts))/sum(wts)
def shrunk_rate(wins,games,prior,k): return (wins+k*prior)/(games+k)

# ---- load ----
df=pd.read_csv(DATA_FILE)
df['blue_team']=df['blue_team'].apply(normalize_team)
df['red_team']=df['red_team'].apply(normalize_team)
for c in ['blue_picks','red_picks','blue_players','red_players']:
    df[c]=df[c].apply(lambda x:[s.strip() for s in str(x).split(',')])
if 'date' in df.columns:
    df['date']=pd.to_datetime(df['date'],errors='coerce')
    df=df.sort_values('date').reset_index(drop=True)
if 'year' not in df.columns: df['year']=df['date'].dt.year
df['date_str']=df['date'].astype(str).str[:10] if 'date' in df.columns else ''
print(f"Loaded {len(df)} games | train(<=2025): {(df['year']<=2025).sum()} | test(2026): {(df['year']==2026).sum()}")

# ---- gold lookup (same as backtester) ----
raw_files=sorted([f for f in glob.glob('*LoL_esports_match_data_from_OraclesElixir*.csv') if '2022' not in f])
raw_dfs=[pd.read_csv(f,low_memory=False) for f in raw_files if os.path.exists(f)]
raw=pd.concat(raw_dfs,ignore_index=True) if raw_dfs else pd.DataFrame()
gold_lookup={}
if not raw.empty:
    raw['date']=pd.to_datetime(raw['date'],errors='coerce')
    raw=raw[raw['league'].isin(TARGET_LEAGUES)].copy()
    raw['teamname']=raw['teamname'].apply(lambda x: normalize_team(str(x)) if pd.notna(x) else x)
    tr=raw[raw['position']=='team'].sort_values('date').reset_index(drop=True)
    for col in ['golddiffat10','golddiffat20']:
        tr[col]=pd.to_numeric(tr[col],errors='coerce').fillna(0)
    g10=defaultdict(list); g20=defaultdict(list)
    for _,r in tr.iterrows():
        t=r['teamname']; ds=str(r['date'])[:10]; w=GOLD_WINDOW
        a20=sum(g20[t][-w:])/len(g20[t][-w:]) if g20[t] else 0.0
        a10=sum(g10[t][-w:])/len(g10[t][-w:]) if g10[t] else 0.0
        gold_lookup[(ds,t)]={'avg_gd20':a20,'late_scaling':a20-a10}
        g10[t].append(float(r['golddiffat10'])); g20[t].append(float(r['golddiffat20']))

# ---- role_champ_rate (shrunk, same as production) ----
rc_w={}; rc_g={}
for _,row in df.iterrows():
    res=row['blue_win']
    for i,c in enumerate(row['blue_picks']):
        k=(POSITIONS[i] if i<5 else 'unknown', c); rc_g[k]=rc_g.get(k,0)+1; rc_w[k]=rc_w.get(k,0)+res
    for i,c in enumerate(row['red_picks']):
        k=(POSITIONS[i] if i<5 else 'unknown', c); rc_g[k]=rc_g.get(k,0)+1; rc_w[k]=rc_w.get(k,0)+(1-res)
role_champ_rate={k:shrunk_rate(rc_w[k],rc_g[k],0.5,K_ROLE) for k in rc_g}

def lane_rate(pos, champ):
    return role_champ_rate.get((pos, champ), 0.5)

# ---- build features (V8 + lane diffs), incremental like backtester ----
team_wins={}; team_games={}; champ_wins={}; champ_games={}
h2h={}; team_recent={}; pc_wins_w={}; pc_games_w={}
rows=[]
for _,row in df.iterrows():
    blue,red=row['blue_team'],row['red_team']
    bp,rp=row['blue_picks'],row['red_picks']
    bpl,rpl=row['blue_players'],row['red_players']
    ds=row.get('date_str','')

    b_games=team_games.get(blue,0); r_games=team_games.get(red,0)
    b_wr=team_wins.get(blue,0)/b_games if b_games>0 else 0.5
    r_wr=team_wins.get(red,0)/r_games if r_games>0 else 0.5
    b_cwr=sum(shrunk_rate(champ_wins.get(c,0),champ_games.get(c,0),0.5,K_CHAMP) for c in bp)/len(bp)
    r_cwr=sum(shrunk_rate(champ_wins.get(c,0),champ_games.get(c,0),0.5,K_CHAMP) for c in rp)/len(rp)
    mk=tuple(sorted([blue,red])); hr=h2h.get(mk,{}); ht=sum(hr.values())
    h2h_rate=cap_h2h(hr.get(blue,0)/ht) if ht>0 else 0.5
    bh=team_recent.get(blue,[]); rh=team_recent.get(red,[])
    b_form=weighted_form(bh,FORM_WINDOW); r_form=weighted_form(rh,FORM_WINDOW)
    b_rwr=sum(bh[-RECENT_WINDOW:])/len(bh[-RECENT_WINDOW:]) if bh else 0.5
    r_rwr=sum(rh[-RECENT_WINDOW:])/len(rh[-RECENT_WINDOW:]) if rh else 0.5
    b_pc=[PC_WEIGHT*shrunk_rate(pc_wins_w.get((pl,c),0),pc_games_w.get((pl,c),0),
              role_champ_rate.get((POSITIONS[i] if i<5 else 'unknown',c),0.5),K_PC)
          +RC_WEIGHT*role_champ_rate.get((POSITIONS[i] if i<5 else 'unknown',c),0.5)
          for i,(pl,c) in enumerate(zip(bpl,bp))]
    r_pc=[PC_WEIGHT*shrunk_rate(pc_wins_w.get((pl,c),0),pc_games_w.get((pl,c),0),
              role_champ_rate.get((POSITIONS[i] if i<5 else 'unknown',c),0.5),K_PC)
          +RC_WEIGHT*role_champ_rate.get((POSITIONS[i] if i<5 else 'unknown',c),0.5)
          for i,(pl,c) in enumerate(zip(rpl,rp))]
    b_pca=sum(b_pc)/len(b_pc); r_pca=sum(r_pc)/len(r_pc)
    bf=gold_lookup.get((ds,blue),{}); rf=gold_lookup.get((ds,red),{})

    # ---- NEW: per-lane matchup diffs ----
    # positional order confirmed, so index i => POSITIONS[i]
    lane_diffs={}
    for i,pos in enumerate(POSITIONS):
        bc = bp[i] if i < len(bp) else None
        rc = rp[i] if i < len(rp) else None
        b_lane = lane_rate(pos, bc) if bc else 0.5
        r_lane = lane_rate(pos, rc) if rc else 0.5
        lane_diffs[f'{pos}_diff'] = b_lane - r_lane

    # Stage-2 preview (not fed to model yet): count losing lanes for blue
    num_losing = sum(1 for v in lane_diffs.values() if v < 0)

    rows.append({
        # ----- V8 baseline features -----
        'blue_team_winrate':b_wr,'red_team_winrate':r_wr,'team_winrate_diff':b_wr-r_wr,
        'blue_team_games':b_games,'red_team_games':r_games,
        'blue_avg_winrate':b_cwr,'red_avg_winrate':r_cwr,'winrate_diff':b_cwr-r_cwr,
        'h2h_winrate':h2h_rate,
        'blue_form':b_form,'red_form':r_form,'form_diff':b_form-r_form,
        'blue_recent_wr':b_rwr,'red_recent_wr':r_rwr,'recent_wr_diff':b_rwr-r_rwr,
        'blue_side_advantage':BLUE_SIDE_WINRATE,
        'blue_pc_avg':b_pca,'red_pc_avg':r_pca,'pc_avg_diff':b_pca-r_pca,
        'blue_avg_gd20':bf.get('avg_gd20',0.0),'red_avg_gd20':rf.get('avg_gd20',0.0),
        'gd20_diff':bf.get('avg_gd20',0.0)-rf.get('avg_gd20',0.0),
        'blue_late_scaling':bf.get('late_scaling',0.0),'red_late_scaling':rf.get('late_scaling',0.0),
        'late_scaling_diff':bf.get('late_scaling',0.0)-rf.get('late_scaling',0.0),
        # ----- NEW lane diffs -----
        **lane_diffs,
        'num_losing_lanes': num_losing,
        'year':row['year'],
    })

    # update state AFTER building features
    res=row['blue_win']
    team_games[blue]=team_games.get(blue,0)+1; team_games[red]=team_games.get(red,0)+1
    team_wins[blue]=team_wins.get(blue,0)+res; team_wins[red]=team_wins.get(red,0)+(1-res)
    for c in bp: champ_games[c]=champ_games.get(c,0)+1; champ_wins[c]=champ_wins.get(c,0)+res
    for c in rp: champ_games[c]=champ_games.get(c,0)+1; champ_wins[c]=champ_wins.get(c,0)+(1-res)
    if mk not in h2h: h2h[mk]={}
    h2h[mk][blue]=h2h[mk].get(blue,0)+res; h2h[mk][red]=h2h[mk].get(red,0)+(1-res)
    if blue not in team_recent: team_recent[blue]=[]
    if red not in team_recent: team_recent[red]=[]
    team_recent[blue].append(1 if res==1 else 0); team_recent[red].append(0 if res==1 else 1)
    for pl,c in zip(bpl,bp): pc_games_w[(pl,c)]=pc_games_w.get((pl,c),0)+1; pc_wins_w[(pl,c)]=pc_wins_w.get((pl,c),0)+res
    for pl,c in zip(rpl,rp): pc_games_w[(pl,c)]=pc_games_w.get((pl,c),0)+1; pc_wins_w[(pl,c)]=pc_wins_w.get((pl,c),0)+(1-res)

feat=pd.DataFrame(rows).reset_index(drop=True)

# ---- champion one-hot (same as V8) ----
mlb=MultiLabelBinarizer()
mlb.fit((df['blue_picks']+df['red_picks']).tolist())
be=pd.DataFrame(mlb.transform(df['blue_picks']),columns=['blue_'+c for c in mlb.classes_]).reset_index(drop=True)
re_=pd.DataFrame(mlb.transform(df['red_picks']),columns=['red_'+c for c in mlb.classes_]).reset_index(drop=True)

y=df['blue_win'].reset_index(drop=True)
train=feat['year']<=2025
test=feat['year']==2026

V8_COLS=['blue_team_winrate','red_team_winrate','team_winrate_diff',
    'blue_team_games','red_team_games',
    'blue_avg_winrate','red_avg_winrate','winrate_diff','h2h_winrate',
    'blue_form','red_form','form_diff',
    'blue_recent_wr','red_recent_wr','recent_wr_diff','blue_side_advantage',
    'blue_pc_avg','red_pc_avg','pc_avg_diff',
    'blue_avg_gd20','red_avg_gd20','gd20_diff',
    'blue_late_scaling','red_late_scaling','late_scaling_diff']
LANE_COLS=['top_diff','jng_diff','mid_diff','adc_diff','sup_diff']

def run(label, extra_cols, include_champs=True):
    parts=[]
    if include_champs: parts=[be,re_]
    X=pd.concat(parts+[feat[extra_cols]],axis=1)
    m=GradientBoostingClassifier(n_estimators=125,max_depth=2,learning_rate=0.1,random_state=42)
    m.fit(X[train.values],y[train.values])
    pred=m.predict(X[test.values]); prob=m.predict_proba(X[test.values])[:,1]
    acc=accuracy_score(y[test.values],pred); auc=roc_auc_score(y[test.values],prob)
    print(f"  {label:<34} Acc:{acc*100:.2f}%  AUC:{auc:.4f}")
    return acc, auc, m, X

print(f"\n{'='*60}")
print(f"  STAGE 1 — LANE MATCHUP DIFF TEST (train<=2025, test 2026)")
print(f"{'='*60}")
accA,aucA,_,_        = run("A) V8 baseline",            V8_COLS)
accB,aucB,mB,XB      = run("B) V8 + 5 lane diffs",      V8_COLS+LANE_COLS)
accC,aucC,_,_        = run("C) lane diffs ALONE (+champs)", LANE_COLS)
accD,aucD,_,_        = run("D) lane diffs, NO champs",  LANE_COLS, include_champs=False)

print(f"\n  {'-'*56}")
print(f"  Uplift B vs A:   Acc {(accB-accA)*100:+.2f}%   AUC {aucB-aucA:+.4f}")
print(f"  (positive = lane granularity beats averaging)")

# ---- feature importance of the lane diffs within model B ----
print(f"\n  LANE-DIFF FEATURE IMPORTANCE (within model B)")
importances=pd.Series(mB.feature_importances_, index=XB.columns)
lane_imp=importances[LANE_COLS].sort_values(ascending=False)
total_imp=importances.sum()
for name,imp in lane_imp.items():
    print(f"    {name:<12} {imp/total_imp*100:5.2f}% of total importance")
print(f"    {'ALL LANES':<12} {lane_imp.sum()/total_imp*100:5.2f}% combined")

# ---- Stage 2 preview: does 'num_losing_lanes' correlate with outcome? ----
print(f"\n  STAGE 2 PREVIEW — num_losing_lanes vs blue_win (2026 holdout)")
test_feat=feat[test.values]
corr=np.corrcoef(test_feat['num_losing_lanes'], y[test.values])[0,1]
print(f"    Correlation: {corr:+.4f}  (negative expected: more losing lanes -> blue loses)")
for n in range(6):
    mask=test_feat['num_losing_lanes']==n
    if mask.sum()>0:
        wr=y[test.values][mask.values].mean()
        print(f"    {n} losing lanes: blue wins {wr*100:5.1f}%  ({mask.sum()} games)")

print(f"\n{'='*60}")
print("  Read: if B clearly beats A AND lanes carry real importance,")
print("  Stage 1 is validated -> proceed to Stage 2 (losing-lane")
print("  interaction). If B ~ A, lane granularity doesn't help on")
print("  top of what averaging already captures.")
print(f"{'='*60}\n")
