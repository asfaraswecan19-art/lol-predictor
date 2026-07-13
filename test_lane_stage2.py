"""
STAGE 2 — LOSING-LANE COUNT + JUNGLE INTERACTION  (standalone A/B backtest)
==========================================================================
Stage 1 result: flat per-lane diffs added ~nothing (+0.21% acc, 3% importance)
because "who's losing lanes" is already implicit in V8's other features.

BUT the losing-lane WIN-RATE LADDER was clean and steep (72.7% at 0 losing
lanes -> 27.5% at 5), and the biggest drop was 2->3 losing lanes -- exactly
the "collapsed position" threshold. Stage 1 never actually fed the model the
lane COUNT, and never tested the specific thing you described: a strong
jungler with many losing lanes should be discounted (a NON-LINEAR interaction
that flat features can't express).

This tests that narrow question. Configs on the identical 2026 holdout:
  A) V8 baseline
  B) V8 + num_losing_lanes                      (the count as a feature)
  C) V8 + num_losing_lanes + jng_diff           (count + jungle strength)
  D) V8 + jng_x_losing interaction term         (explicit product)
  E) V8 + all of the above (let GBM sort it)

We also try a 'severity' version (sum of negative lane diffs, not just a
count) since a lane losing by a little != losing by a lot.

Train 2023-2025, test 2026. Run from dataset builder folder. Paste output.
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

DATA_FILE='proplay_matches.csv'
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

# gold lookup
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

# role_champ_rate
rc_w={}; rc_g={}
for _,row in df.iterrows():
    res=row['blue_win']
    for i,c in enumerate(row['blue_picks']):
        k=(POSITIONS[i] if i<5 else 'unknown', c); rc_g[k]=rc_g.get(k,0)+1; rc_w[k]=rc_w.get(k,0)+res
    for i,c in enumerate(row['red_picks']):
        k=(POSITIONS[i] if i<5 else 'unknown', c); rc_g[k]=rc_g.get(k,0)+1; rc_w[k]=rc_w.get(k,0)+(1-res)
role_champ_rate={k:shrunk_rate(rc_w[k],rc_g[k],0.5,K_ROLE) for k in rc_g}
def lane_rate(pos,champ): return role_champ_rate.get((pos,champ),0.5)

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

    # per-lane diffs
    lane_diffs=[]
    for i,pos in enumerate(POSITIONS):
        bc=bp[i] if i<len(bp) else None; rc=rp[i] if i<len(rp) else None
        lane_diffs.append((lane_rate(pos,bc) if bc else 0.5) - (lane_rate(pos,rc) if rc else 0.5))
    jng_diff=lane_diffs[1]

    # blue's losing-lane count + severity (severity = sum of how much each losing lane is behind)
    num_losing = sum(1 for v in lane_diffs if v < 0)
    losing_severity = -sum(v for v in lane_diffs if v < 0)   # >=0, bigger = deeper deficits
    # jungle interaction: a strong jungle is worth LESS with more losing lanes.
    # encode as jng_diff * (winning-lane count) so strong jungle + winning lanes compounds,
    # and strong jungle + losing lanes is damped.
    jng_x_winning = jng_diff * (5 - num_losing)
    jng_x_losing  = jng_diff * num_losing

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
        # Stage 2 features
        'num_losing_lanes':num_losing,
        'losing_severity':losing_severity,
        'jng_diff':jng_diff,
        'jng_x_winning':jng_x_winning,
        'jng_x_losing':jng_x_losing,
        'year':row['year'],
    })

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

mlb=MultiLabelBinarizer(); mlb.fit((df['blue_picks']+df['red_picks']).tolist())
be=pd.DataFrame(mlb.transform(df['blue_picks']),columns=['blue_'+c for c in mlb.classes_]).reset_index(drop=True)
re_=pd.DataFrame(mlb.transform(df['red_picks']),columns=['red_'+c for c in mlb.classes_]).reset_index(drop=True)
y=df['blue_win'].reset_index(drop=True)
train=feat['year']<=2025; test=feat['year']==2026

V8_COLS=['blue_team_winrate','red_team_winrate','team_winrate_diff',
    'blue_team_games','red_team_games',
    'blue_avg_winrate','red_avg_winrate','winrate_diff','h2h_winrate',
    'blue_form','red_form','form_diff',
    'blue_recent_wr','red_recent_wr','recent_wr_diff','blue_side_advantage',
    'blue_pc_avg','red_pc_avg','pc_avg_diff',
    'blue_avg_gd20','red_avg_gd20','gd20_diff',
    'blue_late_scaling','red_late_scaling','late_scaling_diff']

def run(label, extra_cols):
    X=pd.concat([be,re_,feat[V8_COLS+extra_cols]],axis=1)
    m=GradientBoostingClassifier(n_estimators=125,max_depth=2,learning_rate=0.1,random_state=42)
    m.fit(X[train.values],y[train.values])
    pred=m.predict(X[test.values]); prob=m.predict_proba(X[test.values])[:,1]
    acc=accuracy_score(y[test.values],pred); auc=roc_auc_score(y[test.values],prob)
    print(f"  {label:<44} Acc:{acc*100:.2f}%  AUC:{auc:.4f}")
    return acc,auc,m,X

print(f"\n{'='*64}")
print(f"  STAGE 2 — LOSING-LANE COUNT + JUNGLE INTERACTION")
print(f"{'='*64}")
accA,aucA,_,_       = run("A) V8 baseline", [])
accB,aucB,_,_       = run("B) V8 + num_losing_lanes", ['num_losing_lanes'])
accC,aucC,_,_       = run("C) V8 + num_losing + jng_diff", ['num_losing_lanes','jng_diff'])
accD,aucD,mD,XD     = run("D) V8 + jng_x_losing + jng_x_winning", ['jng_diff','jng_x_losing','jng_x_winning'])
accE,aucE,mE,XE     = run("E) V8 + severity + jng interactions", ['num_losing_lanes','losing_severity','jng_diff','jng_x_losing','jng_x_winning'])

print(f"\n  {'-'*60}")
best_label,best_acc,best_auc = max(
    [('B',accB,aucB),('C',accC,aucC),('D',accD,aucD),('E',accE,aucE)],
    key=lambda t:t[2])
print(f"  Baseline A:            Acc {accA*100:.2f}%  AUC {aucA:.4f}")
print(f"  Best variant ({best_label}):      Acc {best_acc*100:.2f}%  AUC {best_auc:.4f}")
print(f"  Best uplift vs A:      Acc {(best_acc-accA)*100:+.2f}%  AUC {best_auc-aucA:+.4f}")

# importance of the new features in the richest model (E)
print(f"\n  NEW-FEATURE IMPORTANCE (within model E)")
imp=pd.Series(mE.feature_importances_, index=XE.columns)
tot=imp.sum()
for name in ['num_losing_lanes','losing_severity','jng_diff','jng_x_losing','jng_x_winning']:
    print(f"    {name:<18} {imp[name]/tot*100:5.2f}% of total importance")

print(f"\n{'='*64}")
print("  Read: if best variant clearly beats A (and the interaction")
print("  terms carry real importance), the jungle×lane-state idea is")
print("  real and worth wiring into production. If flat like Stage 1,")
print("  V8 already contains this signal -> close out the lane line.")
print(f"{'='*64}\n")
