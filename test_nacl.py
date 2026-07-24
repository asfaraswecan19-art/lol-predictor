"""
test_nacl.py — should NACL be in the T2 model?
==============================================
NACL (North American Challengers League) has been excluded from T2 for a while,
but the original reasoning isn't documented. This re-tests it empirically, the
same way we tested EWC and the meta features: build the T2 win model WITHOUT
NACL (current behavior) and WITH NACL, and compare out-of-sample accuracy/edge.

It answers three questions:
  1. Does adding NACL change overall T2 performance (help / hurt / flat)?
  2. Are NACL games THEMSELVES predictable, or are they noise?
  3. Is NACL roster-unstable (teams with little history) -- the usual reason to
     exclude a challenger league?

Reads the Oracle's Elixir CSVs directly; touches no pipeline files.
Temporal split: train <=2025, test 2026.
"""

import pandas as pd, numpy as np, os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

FILES = [f'{y}_LoL_esports_match_data_from_OraclesElixir.csv' for y in (2023,2024,2025,2026)]
T2_BASE = ['LCKC','LFL','EM','PRM']          # current T2 (OE league codes may differ)
# OE sometimes labels these differently; map common variants
LEAGUE_ALIASES = {'NA Challengers':'NACL','NACL':'NACL','North American Challengers League':'NACL'}

print("="*64)
print("  NACL INCLUSION TEST (T2)")
print("="*64)

frames=[]
for f in FILES:
    if os.path.exists(f):
        d = pd.read_csv(f, low_memory=False)
        frames.append(d)
    else:
        print(f"  (missing {f})")
if not frames:
    raise SystemExit("No OE CSVs found in this folder.")
raw = pd.concat(frames, ignore_index=True)
raw['league'] = raw['league'].replace(LEAGUE_ALIASES)
raw['year'] = pd.to_datetime(raw['date'], errors='coerce').dt.year

# what does NACL look like in the data?
nacl_names = [l for l in raw['league'].dropna().unique() if 'NACL' in str(l) or 'Challeng' in str(l)]
print(f"  NACL-like league labels present: {nacl_names}")

def build_games(df, leagues):
    """Collapse OE player rows into one row per game with picks + winner."""
    df = df[df['league'].isin(leagues)].copy()
    team_df = df[df['position']=='team'].copy()
    players = df[df['position']!='team'].copy()
    games=[]
    picks_by = {}
    for (gid, side), grp in players.groupby(['gameid','side']):
        picks_by[(gid, side)] = [str(c) for c in grp['champion'].tolist()]
    for gid, grp in team_df.groupby('gameid'):
        b = grp[grp['side']=='Blue']; r = grp[grp['side']=='Red']
        if len(b)!=1 or len(r)!=1: continue
        b=b.iloc[0]; r=r.iloc[0]
        bp = picks_by.get((gid,'Blue'),[]); rp = picks_by.get((gid,'Red'),[])
        if len(bp)!=5 or len(rp)!=5: continue
        games.append({'gameid':gid,'league':b['league'],'year':b['year'],
                      'blue_team':b['teamname'],'red_team':r['teamname'],
                      'blue_win':int(b['result']),'blue_picks':bp,'red_picks':rp})
    return pd.DataFrame(games)

def featurize_and_score(games, label):
    games = games.dropna(subset=['year']).sort_values('year').reset_index(drop=True)
    # team early/overall winrate (simple, leak-free-ish: computed on train only)
    tr = games['year']<=2025; te = games['year']==2026
    if te.sum() < 20 or tr.sum() < 100:
        print(f"  [{label}] not enough data (train {tr.sum()}, test {te.sum()})"); return None
    # team winrate from training games
    tw={}; tg={}
    for _,r in games[tr].iterrows():
        tg[r['blue_team']]=tg.get(r['blue_team'],0)+1; tg[r['red_team']]=tg.get(r['red_team'],0)+1
        tw[r['blue_team']]=tw.get(r['blue_team'],0)+r['blue_win']
        tw[r['red_team']]=tw.get(r['red_team'],0)+(1-r['blue_win'])
    rate=lambda t: tw.get(t,0)/tg[t] if tg.get(t,0)>0 else 0.5
    games['b_wr']=games['blue_team'].map(rate); games['r_wr']=games['red_team'].map(rate)
    games['wr_diff']=games['b_wr']-games['r_wr']
    mlb=MultiLabelBinarizer(); mlb.fit(games['blue_picks']+games['red_picks'])
    B=pd.DataFrame(mlb.transform(games['blue_picks']),columns=['b_'+c for c in mlb.classes_])
    R=pd.DataFrame(mlb.transform(games['red_picks']),columns=['r_'+c for c in mlb.classes_])
    X=pd.concat([B,R,games[['b_wr','r_wr','wr_diff']].reset_index(drop=True)],axis=1)
    y=games['blue_win'].reset_index(drop=True)
    tr=tr.values; te=te.values
    m=GradientBoostingClassifier(n_estimators=125,max_depth=1,learning_rate=0.03,random_state=42)
    m.fit(X[tr],y[tr])
    p=m.predict_proba(X[te])[:,1]; yt=y[te]
    acc=accuracy_score(yt,(p>=0.5).astype(int)); auc=roc_auc_score(yt,p)
    base=max(yt.mean(),1-yt.mean())
    print(f"  [{label}] test games {te.sum()}  acc {acc*100:.2f}%  AUC {auc:.4f}  "
          f"baseline {base*100:.2f}%  edge {(acc-base)*100:+.2f}%")
    return {'acc':acc,'auc':auc,'edge':acc-base,'games':games,'test_mask':te,
            'pred':(p>=0.5).astype(int),'yt':yt.values}

# 1) current T2 (no NACL)
print("\n--- T2 WITHOUT NACL (current) ---")
g_base = build_games(raw, T2_BASE)
print(f"  built {len(g_base)} games across {T2_BASE}")
r_base = featurize_and_score(g_base, "no NACL")

# 2) T2 + NACL
print("\n--- T2 WITH NACL ---")
g_nacl = build_games(raw, T2_BASE+['NACL'])
print(f"  built {len(g_nacl)} games across {T2_BASE+['NACL']}")
r_with = featurize_and_score(g_nacl, "with NACL")

# 3) NACL games alone — are they predictable at all?
print("\n--- NACL ALONE (are its games even predictable?) ---")
g_only = build_games(raw, ['NACL'])
print(f"  built {len(g_only)} NACL games")
if len(g_only)>0:
    r_only = featurize_and_score(g_only, "NACL only")
    # roster stability: what fraction of NACL test teams have >=10 training games?
    if r_only:
        g=r_only['games']; tr=g['year']<=2025; te=g['year']==2026
        tg={}
        for _,r in g[tr].iterrows():
            tg[r['blue_team']]=tg.get(r['blue_team'],0)+1; tg[r['red_team']]=tg.get(r['red_team'],0)+1
        test_teams=set(g[te]['blue_team'])|set(g[te]['red_team'])
        seen=[t for t in test_teams if tg.get(t,0)>=10]
        print(f"  Roster stability: {len(seen)}/{len(test_teams)} NACL 2026 teams "
              f"have >=10 training games ({len(seen)/max(len(test_teams),1)*100:.0f}%)")
        print("  (low % = unstable rosters = weak team-history signal = reason to exclude)")

# verdict
print("\n" + "="*64)
print("  READING THE RESULT")
print("="*64)
if r_base and r_with:
    d_acc=(r_with['acc']-r_base['acc'])*100
    d_auc=(r_with['auc']-r_base['auc'])
    print(f"  Adding NACL changes overall T2:  acc {d_acc:+.2f}pts,  AUC {d_auc:+.4f}")
    if d_acc < -0.5 or d_auc < -0.01:
        print("  -> NACL HURTS. Keep it excluded.")
    elif d_acc > 0.5 and d_auc > 0.005:
        print("  -> NACL HELPS. Consider adding it back.")
    else:
        print("  -> ~FLAT. Exclusion is a wash; decide on other grounds (coverage, noise).")
print("="*64)
