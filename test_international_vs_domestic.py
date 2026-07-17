"""
INTERNATIONAL vs DOMESTIC PREDICTION ACCURACY
=============================================
QUESTION: does the model predict cross-region matches (MSI / Worlds / EWC)
worse than domestic-league matches?

WHY THIS MIGHT BE TRUE (the structural argument):
  team_winrate, form, recent_wr, avg_gd20 are all earned against DOMESTIC
  opponents. A 65% LCK win rate and a 65% LEC win rate look identical to
  the model but represent different achievements against different opponent
  pools. Nothing in the feature set encodes region strength. h2h is also
  usually empty for cross-region pairs.

WHY A SINGLE-YEAR TEST CAN'T ANSWER IT:
  International events are short (a few dozen games/year). The 2026 holdout
  alone may contain little or no MSI/Worlds/EWC. So this uses a WALK-FORWARD
  evaluation across multiple years to pool enough international games.

METHOD (walk-forward, no leakage):
  For each test year Y in {2024, 2025, 2026}:
    - train on all games with year < Y
    - predict games with year == Y
  Then pool all those out-of-sample predictions and split them into
  INTERNATIONAL vs DOMESTIC to compare accuracy, AUC, and calibration.

  Features are built incrementally in date order (a game's features only use
  prior games), mirroring backtester.py.

IMPORTANT CAVEAT ON INTERPRETATION:
  International fields are stacked with the best teams in the world, so games
  are closer on paper. A lower accuracy there might reflect GENUINELY CLOSER
  MATCHES rather than a model failure. So we also report the always-favourite
  baseline for each group -- EDGE OVER BASELINE is the fair comparison,
  not raw accuracy. If international edge ~= domestic edge, the model is
  handling cross-region fine and lower raw accuracy is just harder games.

Run from the dataset builder folder. Paste output back.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import glob, os
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

DATA_FILE='proplay_matches.csv'
FORM_WINDOW=8; RECENT_WINDOW=20; GOLD_WINDOW=15
BLUE_SIDE_WINRATE=0.5312
H2H_CAP=0.60
K_CHAMP=8; K_ROLE=5; K_PC=12
POSITIONS=['top','jng','mid','adc','sup']
TARGET_LEAGUES=['LCK','LPL','LEC','LCS','CBLOL','MSI','WLDs','EWC','LTA N','LTA S','LTA','FST']

# Which leagues are cross-region international events
INTERNATIONAL = {'MSI', 'WLDs', 'EWC'}

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
df['is_intl']=df['league'].isin(INTERNATIONAL)

print("="*68)
print("  INTERNATIONAL vs DOMESTIC — DATA INVENTORY")
print("="*68)
print(f"Total games: {len(df)}")
print(f"\nGames per league per year:")
pivot=df.pivot_table(index='league',columns='year',values='game_id',aggfunc='count',fill_value=0)
print(pivot.to_string())
print(f"\nInternational leagues found: {sorted(set(df[df['is_intl']]['league'].unique()))}")
print(f"International games total:   {df['is_intl'].sum()}")
print(f"Domestic games total:        {(~df['is_intl']).sum()}")

if df['is_intl'].sum() == 0:
    raise SystemExit("\nNo international games found. Did you rebuild with EWC/MSI/WLDs in "
                     "TARGET_LEAGUES_T1 and rerun build_dataset.py?")

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

# ---------------- role_champ_rate ----------------
rc_w={}; rc_g={}
for _,row in df.iterrows():
    res=row['blue_win']
    for i,c in enumerate(row['blue_picks']):
        k=(POSITIONS[i] if i<5 else 'unknown', c); rc_g[k]=rc_g.get(k,0)+1; rc_w[k]=rc_w.get(k,0)+res
    for i,c in enumerate(row['red_picks']):
        k=(POSITIONS[i] if i<5 else 'unknown', c); rc_g[k]=rc_g.get(k,0)+1; rc_w[k]=rc_w.get(k,0)+(1-res)
role_champ_rate={k:shrunk_rate(rc_w[k],rc_g[k],0.5,K_ROLE) for k in rc_g}

# ---------------- build features incrementally ----------------
team_wins={}; team_games={}; champ_wins={}; champ_games={}
h2h={}; team_recent={}; pc_wins_w={}; pc_games_w={}
rows=[]
for _,row in df.iterrows():
    blue,red=row['blue_team'],row['red_team']
    bp,rp=row['blue_picks'],row['red_picks']
    bpl,rpl=row['blue_players'],row['red_players']
    ds=row['date_str']

    b_games=team_games.get(blue,0); r_games=team_games.get(red,0)
    b_wr=team_wins.get(blue,0)/b_games if b_games>0 else 0.5
    r_wr=team_wins.get(red,0)/r_games if r_games>0 else 0.5
    b_cwr=sum(shrunk_rate(champ_wins.get(c,0),champ_games.get(c,0),0.5,K_CHAMP) for c in bp)/len(bp)
    r_cwr=sum(shrunk_rate(champ_wins.get(c,0),champ_games.get(c,0),0.5,K_CHAMP) for c in rp)/len(rp)
    mk=tuple(sorted([blue,red])); hr=h2h.get(mk,{}); ht=sum(hr.values())
    h2h_rate=cap_h2h(hr.get(blue,0)/ht) if ht>0 else 0.5
    h2h_games=ht   # tracked so we can report how often cross-region h2h is empty
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
        'year':row['year'],'is_intl':row['is_intl'],'league':row['league'],
        '_h2h_games':h2h_games,
        '_gold_missing': (0 if bf else 1) + (0 if rf else 1),
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

FEAT_COLS=['blue_team_winrate','red_team_winrate','team_winrate_diff',
    'blue_team_games','red_team_games',
    'blue_avg_winrate','red_avg_winrate','winrate_diff','h2h_winrate',
    'blue_form','red_form','form_diff',
    'blue_recent_wr','red_recent_wr','recent_wr_diff','blue_side_advantage',
    'blue_pc_avg','red_pc_avg','pc_avg_diff',
    'blue_avg_gd20','red_avg_gd20','gd20_diff',
    'blue_late_scaling','red_late_scaling','late_scaling_diff']

mlb=MultiLabelBinarizer(); mlb.fit((df['blue_picks']+df['red_picks']).tolist())
be=pd.DataFrame(mlb.transform(df['blue_picks']),columns=['blue_'+c for c in mlb.classes_]).reset_index(drop=True)
re_=pd.DataFrame(mlb.transform(df['red_picks']),columns=['red_'+c for c in mlb.classes_]).reset_index(drop=True)
X=pd.concat([be,re_,feat[FEAT_COLS]],axis=1)
y=df['blue_win'].reset_index(drop=True)

# ---------------- feature-quality check: is h2h actually emptier for intl? ----------------
print("\n" + "="*68)
print("  FEATURE QUALITY — is the model actually blinder on international?")
print("="*68)
for label, mask in [('DOMESTIC', ~feat['is_intl']), ('INTERNATIONAL', feat['is_intl'])]:
    sub=feat[mask]
    if len(sub)==0: continue
    no_h2h=(sub['_h2h_games']==0).mean()*100
    gold_miss=(sub['_gold_missing']>0).mean()*100
    print(f"  {label:<14} games={len(sub):>5}  no prior h2h: {no_h2h:5.1f}%  "
          f"missing gold feat: {gold_miss:5.1f}%")
print("  (higher 'no prior h2h' on international = model has less to work with)")

# ---------------- walk-forward evaluation ----------------
print("\n" + "="*68)
print("  WALK-FORWARD OUT-OF-SAMPLE EVALUATION")
print("="*68)

all_years=sorted(df['year'].unique())
test_years=[yr for yr in all_years if (df['year']<yr).sum() > 300]
if not test_years:
    raise SystemExit("Not enough history to walk forward.")
print(f"  Test years: {test_years} (train on all prior years each time)")

preds=[]
for yr in test_years:
    tr=(feat['year']<yr).values
    te=(feat['year']==yr).values
    if te.sum()==0 or tr.sum()<300: continue
    m=GradientBoostingClassifier(n_estimators=125,max_depth=2,learning_rate=0.05,random_state=42)
    m.fit(X[tr],y[tr])
    prob=m.predict_proba(X[te])[:,1]
    sub=feat[te].copy()
    sub['prob']=prob
    sub['actual']=y[te].values
    sub['pred']=(prob>0.5).astype(int)
    sub['test_year']=yr
    preds.append(sub)
    n_intl=sub['is_intl'].sum()
    print(f"    {yr}: trained on {tr.sum():>5} games, tested {te.sum():>4} "
          f"({n_intl} international)")

res=pd.concat(preds,ignore_index=True)

# ---------------- the comparison ----------------
def summarize(sub, label):
    if len(sub)<10:
        print(f"  {label:<16} only {len(sub)} games -- TOO FEW to judge")
        return None
    acc=accuracy_score(sub['actual'],sub['pred'])
    try: auc=roc_auc_score(sub['actual'],sub['prob'])
    except ValueError: auc=float('nan')
    brier=brier_score_loss(sub['actual'],sub['prob'])
    # always-favourite baseline: side with higher team winrate
    fav=(sub['team_winrate_diff']>0).astype(int)
    fav_acc=accuracy_score(sub['actual'],fav)
    edge=acc-fav_acc
    print(f"  {label:<16} n={len(sub):>5}  acc={acc*100:5.2f}%  auc={auc:.4f}  "
          f"brier={brier:.4f}  always-fav={fav_acc*100:5.2f}%  EDGE={edge*100:+5.2f}%")
    return {'label':label,'n':len(sub),'acc':acc,'auc':auc,'brier':brier,
            'fav_acc':fav_acc,'edge':edge}

print("\n" + "="*68)
print("  RESULTS — pooled out-of-sample predictions")
print("="*68)
dom = summarize(res[~res['is_intl']], 'DOMESTIC')
intl= summarize(res[res['is_intl']],  'INTERNATIONAL')

print("\n  Breakdown by international event:")
for lg in sorted(res[res['is_intl']]['league'].unique()):
    summarize(res[res['league']==lg], f'  {lg}')

print("\n  Breakdown by domestic league:")
for lg in sorted(res[~res['is_intl']]['league'].unique()):
    summarize(res[res['league']==lg], f'  {lg}')

# ---------------- verdict ----------------
print("\n" + "="*68)
print("  VERDICT")
print("="*68)
if dom and intl:
    acc_gap  = intl['acc']  - dom['acc']
    edge_gap = intl['edge'] - dom['edge']
    auc_gap  = intl['auc']  - dom['auc']
    print(f"  Raw accuracy gap (intl - dom):  {acc_gap*100:+.2f}%")
    print(f"  AUC gap:                        {auc_gap:+.4f}")
    print(f"  EDGE gap (the fair comparison): {edge_gap*100:+.2f}%")
    print()
    if intl['n'] < 100:
        print(f"  ⚠️  Only {intl['n']} international games -- treat this as directional,")
        print(f"      not conclusive. Gaps under ~5% are within noise at this size.")
    print()
    if edge_gap < -0.03:
        print("  -> International edge is MEANINGFULLY WORSE than domestic.")
        print("     Consistent with the structural argument: the model can't")
        print("     compare win rates across regions. Strength-of-schedule /")
        print("     opponent adjustment becomes a real candidate to test.")
    elif edge_gap > 0.03:
        print("  -> International edge is BETTER than domestic. The structural")
        print("     argument does NOT hold here -- aggregate features are")
        print("     handling cross-region fine.")
    else:
        print("  -> International edge is ~SIMILAR to domestic.")
        print("     Any lower RAW accuracy is explained by international fields")
        print("     being stacked (closer games), not by model failure.")
        print("     Cross-region adjustment is NOT obviously worth pursuing.")
print("="*68 + "\n")
