"""
META AWARENESS — STANDALONE A/B BACKTEST
========================================
Two distinct ideas, deliberately separated because they measure different things:

  A) META PICKS (stable)   — what's standard in a role RIGHT NOW.
     The existing role_champ_rate pools 2023-2026 equally, so a champ that was
     meta in 2023 and is now unplayed still carries full weight. This tests a
     TIME-SCOPED version (rolling window) instead.

  B) EMERGENCE (unstable)  — what's SPIKING right now (the "Viktor ADC" case).
     Measured by CHANGE IN PLAY RATE, not win rate. A champ going 0 -> 8 games
     in a role in two weeks is informative even though its win rate is
     statistically meaningless on 8 games. Play rate needs far fewer games to
     measure than win rate, which sidesteps the sparsity problem that killed
     "patch-specific champ WR" (already in the doesn't-work list).

WHY THIS MIGHT STILL FAIL:
  "Patch-specific champ WR: no improvement" is already documented as tested.
  That's adjacent. The differences: (1) we use a rolling WINDOW pooling multiple
  patches, not a single-patch slice, so more data; (2) emergence measures PLAY
  rate (dense) not WIN rate (sparse). If this comes back flat too, that's a
  strong signal the whole meta-timing family is exhausted.

ALL FEATURES ARE BUILT CAUSALLY: for each game, meta stats are computed ONLY
from games strictly BEFORE it. No look-ahead.

Configs tested on the same walk-forward out-of-sample predictions:
  1) V8 baseline
  2) V8 + meta win rate (time-scoped role_champ_rate)
  3) V8 + emergence features (play-rate spike)
  4) V8 + both

Run from the dataset builder folder. Paste output back.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import glob, os
from collections import defaultdict, deque
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_FILE='proplay_matches.csv'
FORM_WINDOW=8; RECENT_WINDOW=20; GOLD_WINDOW=15
BLUE_SIDE_WINRATE=0.5312
H2H_CAP=0.60
K_CHAMP=8; K_ROLE=5; K_PC=12
POSITIONS=['top','jng','mid','adc','sup']
TARGET_LEAGUES=['LCK','LPL','LEC','LCS','CBLOL','MSI','WLDs','EWC','LTA N','LTA S','LTA','FST']

# --- meta window config ---
META_WINDOW_DAYS   = 60    # "what's meta now" lookback
RECENT_SPIKE_DAYS  = 30    # emergence: recent period
BASE_SPIKE_DAYS    = 90    # emergence: baseline period to compare against
K_META             = 5     # shrinkage for the time-scoped meta win rate

TEAM_ALIASES={'Team BDS':'Team Shifters','BDS':'Team Shifters'}
def normalize_team(n): return TEAM_ALIASES.get(str(n),str(n))
def cap_h2h(r): return max(1-H2H_CAP, min(H2H_CAP, r))
def weighted_form(hist,w):
    h=hist[-w:] if hist else []
    if not h: return 0.5
    wts=list(range(1,len(h)+1)); return sum(v*x for v,x in zip(h,wts))/sum(wts)
def shrunk_rate(wins,games,prior,k): return (wins+k*prior)/(games+k)

# ---------------- load ----------------
df=pd.read_csv(DATA_FILE)
df['blue_team']=df['blue_team'].apply(normalize_team)
df['red_team']=df['red_team'].apply(normalize_team)
for c in ['blue_picks','red_picks','blue_players','red_players']:
    df[c]=df[c].apply(lambda x:[s.strip() for s in str(x).split(',')])
df['date']=pd.to_datetime(df['date'],errors='coerce')
df=df.sort_values('date').reset_index(drop=True)
if 'year' not in df.columns: df['year']=df['date'].dt.year
df['date_str']=df['date'].dt.strftime('%Y-%m-%d')
print(f"Loaded {len(df)} games | {df['date'].min().date()} -> {df['date'].max().date()}")

# ---------------- gold lookup ----------------
raw_files=sorted([f for f in glob.glob('*LoL_esports_match_data_from_OraclesElixir*.csv') if '2022' not in f])
raw_dfs=[pd.read_csv(f,low_memory=False) for f in raw_files if os.path.exists(f)]
gold_lookup={}
if raw_dfs:
    raw=pd.concat(raw_dfs,ignore_index=True)
    raw['date']=pd.to_datetime(raw['date'],errors='coerce')
    raw=raw[raw['league'].isin(TARGET_LEAGUES)].copy()
    raw['teamname']=raw['teamname'].apply(lambda x: normalize_team(str(x)) if pd.notna(x) else x)
    tr=raw[raw['position']=='team'].sort_values('date').reset_index(drop=True)
    for col in ['golddiffat10','golddiffat20']:
        tr[col]=pd.to_numeric(tr[col],errors='coerce').fillna(0) if col in tr.columns else 0.0
    g10=defaultdict(list); g20=defaultdict(list)
    for _,r in tr.iterrows():
        t=r['teamname']; ds=str(r['date'])[:10]; w=GOLD_WINDOW
        a20=sum(g20[t][-w:])/len(g20[t][-w:]) if g20[t] else 0.0
        a10=sum(g10[t][-w:])/len(g10[t][-w:]) if g10[t] else 0.0
        gold_lookup[(ds,t)]={'avg_gd20':a20,'late_scaling':a20-a10}
        g10[t].append(float(r['golddiffat10'])); g20[t].append(float(r['golddiffat20']))

# ---------------- all-time role_champ_rate (V8 baseline uses this) ----------------
rc_w={}; rc_g={}
for _,row in df.iterrows():
    res=row['blue_win']
    for i,c in enumerate(row['blue_picks']):
        k=(POSITIONS[i] if i<5 else 'unknown', c); rc_g[k]=rc_g.get(k,0)+1; rc_w[k]=rc_w.get(k,0)+res
    for i,c in enumerate(row['red_picks']):
        k=(POSITIONS[i] if i<5 else 'unknown', c); rc_g[k]=rc_g.get(k,0)+1; rc_w[k]=rc_w.get(k,0)+(1-res)
role_champ_rate={k:shrunk_rate(rc_w[k],rc_g[k],0.5,K_ROLE) for k in rc_g}

# ---------------- CAUSAL meta history ----------------
# For each (role, champ) keep a deque of (date, won). For each game we prune to
# the lookback window using ONLY games strictly before the current date.
meta_hist  = defaultdict(deque)   # (role,champ) -> deque[(date, won)]
role_hist  = defaultdict(deque)   # role -> deque[date]  (total picks in that role)

def prune(dq, cutoff, has_result=True):
    while dq and (dq[0][0] if has_result else dq[0]) < cutoff:
        dq.popleft()

def meta_stats(role, champ, now, role_totals):
    """Causal: only uses games already appended (strictly before `now`).

    PERF: the deques are pruned to BASE_SPIKE_DAYS on access, so they stay
    small instead of growing to the full dataset. Per-role totals are
    computed ONCE per game by the caller and passed in via `role_totals`,
    rather than rescanning the entire role history for all 10 champions in
    every game (which was quadratic and took ~forever on 8.5k games).
    """
    cutoff_meta   = now - pd.Timedelta(days=META_WINDOW_DAYS)
    cutoff_recent = now - pd.Timedelta(days=RECENT_SPIKE_DAYS)
    cutoff_base   = now - pd.Timedelta(days=BASE_SPIKE_DAYS)

    dq = meta_hist[(role,champ)]
    # prune anything older than the widest window we care about
    while dq and dq[0][0] < cutoff_base:
        dq.popleft()

    n_meta=0; w_meta=0; n_recent=0; n_base=0
    for (d,w) in dq:
        if d >= cutoff_meta:
            n_meta += 1; w_meta += w
        if d >= cutoff_recent:
            n_recent += 1
        elif d >= cutoff_base:
            n_base += 1

    role_recent, role_base = role_totals

    meta_wr = shrunk_rate(w_meta, n_meta, 0.5, K_META)
    pr_recent = (n_recent / role_recent) if role_recent > 0 else 0.0
    pr_base   = (n_base   / role_base)   if role_base   > 0 else 0.0
    emergence = pr_recent - pr_base
    is_new    = 1.0 if (n_base == 0 and n_recent > 0) else 0.0

    return meta_wr, pr_recent, emergence, is_new, n_meta

def role_window_totals(role, now):
    """Prune the role deque and return (recent_count, base_count).
    Called ONCE per role per game instead of once per champion."""
    cutoff_recent = now - pd.Timedelta(days=RECENT_SPIKE_DAYS)
    cutoff_base   = now - pd.Timedelta(days=BASE_SPIKE_DAYS)
    rq = role_hist[role]
    while rq and rq[0] < cutoff_base:
        rq.popleft()
    recent=0; base=0
    for d in rq:
        if d >= cutoff_recent: recent += 1
        else: base += 1
    return recent, base

# ---------------- build features ----------------
rows=[]
team_wins={}; team_games={}; champ_wins={}; champ_games={}
h2h={}; team_recent={}; pc_wins_w={}; pc_games_w={}

import time as _time
_t0=_time.time()
_total=len(df)
print(f"\nBuilding features for {_total} games (this takes a minute)...")

for _i,(_,row) in enumerate(df.iterrows()):
    if _i and _i % 1000 == 0:
        _el=_time.time()-_t0
        _eta=_el/_i*(_total-_i)
        print(f"  {_i}/{_total} games ({_i/_total*100:.0f}%) — {_el:.0f}s elapsed, ~{_eta:.0f}s left")
    blue,red=row['blue_team'],row['red_team']
    bp,rp=row['blue_picks'],row['red_picks']
    bpl,rpl=row['blue_players'],row['red_players']
    ds=row['date_str']; now=row['date']

    # ---- V8 features ----
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
    b_pc=[0.10*shrunk_rate(pc_wins_w.get((pl,c),0),pc_games_w.get((pl,c),0),
              role_champ_rate.get((POSITIONS[i] if i<5 else 'unknown',c),0.5),K_PC)
          +0.90*role_champ_rate.get((POSITIONS[i] if i<5 else 'unknown',c),0.5)
          for i,(pl,c) in enumerate(zip(bpl,bp))]
    r_pc=[0.10*shrunk_rate(pc_wins_w.get((pl,c),0),pc_games_w.get((pl,c),0),
              role_champ_rate.get((POSITIONS[i] if i<5 else 'unknown',c),0.5),K_PC)
          +0.90*role_champ_rate.get((POSITIONS[i] if i<5 else 'unknown',c),0.5)
          for i,(pl,c) in enumerate(zip(rpl,rp))]
    b_pca=sum(b_pc)/len(b_pc); r_pca=sum(r_pc)/len(r_pc)
    bf=gold_lookup.get((ds,blue),{}); rf=gold_lookup.get((ds,red),{})

    # ---- META features (causal: meta_hist only holds prior games) ----
    # Compute each role's window totals ONCE for this game, then reuse across
    # both teams' champions in that role (was being recomputed 10x per game).
    role_totals_cache = {pos: role_window_totals(pos, now) for pos in POSITIONS}

    b_meta=[]; b_pr=[]; b_emg=[]; b_new=[]
    for i,c in enumerate(bp):
        pos=POSITIONS[i] if i<5 else 'unknown'
        rt_ = role_totals_cache.get(pos, (0,0))
        mw,pr,em,nw,_=meta_stats(pos,c,now,rt_)
        b_meta.append(mw); b_pr.append(pr); b_emg.append(em); b_new.append(nw)
    r_meta=[]; r_pr=[]; r_emg=[]; r_new=[]
    for i,c in enumerate(rp):
        pos=POSITIONS[i] if i<5 else 'unknown'
        rt_ = role_totals_cache.get(pos, (0,0))
        mw,pr,em,nw,_=meta_stats(pos,c,now,rt_)
        r_meta.append(mw); r_pr.append(pr); r_emg.append(em); r_new.append(nw)

    rows.append({
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
        # --- A) meta win rate (time-scoped) ---
        'blue_meta_wr':np.mean(b_meta),'red_meta_wr':np.mean(r_meta),
        'meta_wr_diff':np.mean(b_meta)-np.mean(r_meta),
        # --- B) emergence / play rate ---
        'blue_playrate':np.mean(b_pr),'red_playrate':np.mean(r_pr),
        'playrate_diff':np.mean(b_pr)-np.mean(r_pr),
        'blue_emergence':np.mean(b_emg),'red_emergence':np.mean(r_emg),
        'emergence_diff':np.mean(b_emg)-np.mean(r_emg),
        'blue_new_picks':np.sum(b_new),'red_new_picks':np.sum(r_new),
        'new_picks_diff':np.sum(b_new)-np.sum(r_new),
        'year':row['year'],'league':row['league'],
    })

    # ---- update state AFTER building features ----
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
    # meta history
    for i,c in enumerate(bp):
        pos=POSITIONS[i] if i<5 else 'unknown'
        meta_hist[(pos,c)].append((now, res)); role_hist[pos].append(now)
    for i,c in enumerate(rp):
        pos=POSITIONS[i] if i<5 else 'unknown'
        meta_hist[(pos,c)].append((now, 1-res)); role_hist[pos].append(now)

feat=pd.DataFrame(rows).reset_index(drop=True)
print(f"  Features built in {_time.time()-_t0:.0f}s")

V8_COLS=['blue_team_winrate','red_team_winrate','team_winrate_diff',
    'blue_team_games','red_team_games',
    'blue_avg_winrate','red_avg_winrate','winrate_diff','h2h_winrate',
    'blue_form','red_form','form_diff',
    'blue_recent_wr','red_recent_wr','recent_wr_diff','blue_side_advantage',
    'blue_pc_avg','red_pc_avg','pc_avg_diff',
    'blue_avg_gd20','red_avg_gd20','gd20_diff',
    'blue_late_scaling','red_late_scaling','late_scaling_diff']
META_COLS=['blue_meta_wr','red_meta_wr','meta_wr_diff']
EMERGE_COLS=['blue_playrate','red_playrate','playrate_diff',
             'blue_emergence','red_emergence','emergence_diff',
             'blue_new_picks','red_new_picks','new_picks_diff']

mlb=MultiLabelBinarizer(); mlb.fit((df['blue_picks']+df['red_picks']).tolist())
be=pd.DataFrame(mlb.transform(df['blue_picks']),columns=['blue_'+c for c in mlb.classes_]).reset_index(drop=True)
re_=pd.DataFrame(mlb.transform(df['red_picks']),columns=['red_'+c for c in mlb.classes_]).reset_index(drop=True)
y=df['blue_win'].reset_index(drop=True)

def walk_forward(extra_cols, label):
    X=pd.concat([be,re_,feat[V8_COLS+extra_cols]],axis=1)
    accs=[]; probs_all=[]; y_all=[]
    for yr in [2024,2025,2026]:
        tr=(feat['year']<yr).values; te=(feat['year']==yr).values
        if te.sum()==0 or tr.sum()<300: continue
        m=GradientBoostingClassifier(n_estimators=125,max_depth=2,learning_rate=0.05,random_state=42)
        m.fit(X[tr],y[tr])
        pb=m.predict_proba(X[te])[:,1]
        probs_all.extend(pb); y_all.extend(y[te].values)
    y_all=np.array(y_all); probs_all=np.array(probs_all)
    acc=accuracy_score(y_all,(probs_all>0.5).astype(int))
    auc=roc_auc_score(y_all,probs_all)
    print(f"  {label:<34} Acc:{acc*100:5.2f}%  AUC:{auc:.4f}")
    return acc,auc,X

print(f"\n{'='*64}")
print(f"  META FEATURES — WALK-FORWARD A/B (train<yr, test=yr)")
print(f"{'='*64}")
accA,aucA,_  = walk_forward([], "1) V8 baseline")
accB,aucB,_  = walk_forward(META_COLS, "2) V8 + meta win rate (time-scoped)")
accC,aucC,_  = walk_forward(EMERGE_COLS, "3) V8 + emergence (play-rate spike)")
accD,aucD,XD = walk_forward(META_COLS+EMERGE_COLS, "4) V8 + both")

print(f"\n  {'-'*60}")
print(f"  Uplift vs baseline:")
for lbl,a,u in [("meta wr",accB,aucB),("emergence",accC,aucC),("both",accD,aucD)]:
    print(f"    {lbl:<12} Acc {(a-accA)*100:+5.2f}%   AUC {u-aucA:+.4f}")

# importance in the full model
print(f"\n  FEATURE IMPORTANCE (model 4, meta+emergence features only)")
tr=(feat['year']<2026).values
m=GradientBoostingClassifier(n_estimators=125,max_depth=2,learning_rate=0.05,random_state=42)
m.fit(XD[tr],y[tr])
imp=pd.Series(m.feature_importances_,index=XD.columns); tot=imp.sum()
for c in META_COLS+EMERGE_COLS:
    print(f"    {c:<18} {imp[c]/tot*100:5.2f}%")
print(f"    {'ALL META/EMERGE':<18} {imp[META_COLS+EMERGE_COLS].sum()/tot*100:5.2f}% combined")

print(f"\n{'='*64}")
print("  READ: if uplift is <0.3% acc and importance is low, the meta")
print("  family is exhausted (consistent with 'patch-specific champ WR:")
print("  no improvement' already in the doesn't-work list). The meta")
print("  REPORT may still be useful to you as information even if it")
print("  doesn't help the model -- that's a separate question.")
print(f"{'='*64}\n")
