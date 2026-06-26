"""
verify_v81.py — Replicate exact v8.1 win + FT5 models with temporal holdout.
Train on 2023-2025, test on 2026. Reports TRUE out-of-sample accuracy.

FT5 model is tested twice: once with proxy data (what v8.1 uses), once with
v2 precise data (from the lolesports scrape).

USAGE:
    python verify_v81.py
"""
import warnings; warnings.filterwarnings('ignore')
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score

TEAM_ALIASES = {'Team BDS': 'Team Shifters', 'BDS': 'Team Shifters'}
def normalize_team(name): return TEAM_ALIASES.get(str(name), str(name))
FORM_WINDOW = 8; RECENT_WINDOW = 20; GOLD_WINDOW = 15
BLUE_SIDE_WINRATE = 0.5312; MIN_PC_GAMES = 12; MIN_ROLE_GAMES = 5
PC_WEIGHT = 0.10; RC_WEIGHT = 0.90; H2H_CAP = 0.60
POSITIONS = ['top','jng','mid','adc','sup']
TARGET_LEAGUES = ['LCK','LPL','LEC','LCS','CBLOL','MSI','WLDs','LTA N','LTA S','LTA','FST']

def cap_h2h(r): return max(1-H2H_CAP, min(H2H_CAP, r))
def weighted_form(hist, window):
    h = hist[-window:] if hist else []
    if not h: return 0.5
    w = list(range(1, len(h)+1))
    return sum(v*x for v,x in zip(h,w)) / sum(w)
def cur_rate(wins, games, key, default=0.5, min_games=1):
    g = games.get(key, 0)
    return wins.get(key, 0)/g if g >= min_games else default


def build_gold_lookup():
    raw_files = sorted([f for f in glob.glob('*LoL_esports_match_data_from_OraclesElixir*.csv')
                        if '2022' not in f])
    print(f"  Oracle files: {raw_files}")
    if not raw_files:
        print("  WARNING: No Oracle files found. Gold features = 0.")
        return {}, False
    raw = pd.concat([pd.read_csv(f, low_memory=False) for f in raw_files], ignore_index=True)
    raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
    raw = raw[raw['league'].isin(TARGET_LEAGUES)].copy()
    raw['teamname'] = raw['teamname'].apply(lambda x: normalize_team(str(x)) if pd.notna(x) else x)
    tr = raw[raw['position']=='team'].sort_values('date').reset_index(drop=True)
    for c in ['golddiffat10','golddiffat20']:
        tr[c] = pd.to_numeric(tr[c], errors='coerce').fillna(0)
    gd10 = defaultdict(list); gd20 = defaultdict(list); lookup = {}
    for _, row in tr.iterrows():
        t = row['teamname']; ds = str(row['date'])[:10]
        a20 = sum(gd20[t][-GOLD_WINDOW:])/len(gd20[t][-GOLD_WINDOW:]) if gd20[t] else 0.0
        a10 = sum(gd10[t][-GOLD_WINDOW:])/len(gd10[t][-GOLD_WINDOW:]) if gd10[t] else 0.0
        lookup[(ds, t)] = {'avg_gd20': a20, 'late_scaling': a20-a10}
        gd10[t].append(float(row['golddiffat10'])); gd20[t].append(float(row['golddiffat20']))
    print(f"  Gold entries: {len(lookup)}")
    return lookup, True


def run_win_model(gold_lookup, has_gold):
    print("\n" + "="*60)
    print("  WIN MODEL — exact v8.1 replication")
    print("="*60)
    df = pd.read_csv('proplay_matches.csv')
    df['blue_team'] = df['blue_team'].apply(normalize_team)
    df['red_team'] = df['red_team'].apply(normalize_team)
    df['blue_picks'] = df['blue_picks'].apply(lambda x: [c.strip() for c in str(x).split(',')])
    df['red_picks'] = df['red_picks'].apply(lambda x: [c.strip() for c in str(x).split(',')])
    df['blue_players'] = df['blue_players'].apply(lambda x: [p.strip() for p in str(x).split(',')])
    df['red_players'] = df['red_players'].apply(lambda x: [p.strip() for p in str(x).split(',')])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date','blue_win']).sort_values('date').reset_index(drop=True)
    df['date_str'] = df['date'].astype(str).str[:10]
    df['year'] = df['date'].dt.year
    def gg(ds,t,f): return gold_lookup.get((ds,t),{}).get(f,0.0)

    rc_w={}; rc_g={}; pc_w={}; pc_g={}
    tw={}; tg={}; cw={}; cg={}; h2h={}; trec={}
    rows = []
    for _, row in df.iterrows():
        b,r = row['blue_team'],row['red_team']
        bp=[c.strip() for c in row['blue_picks']]; rp=[c.strip() for c in row['red_picks']]
        bpl=[p.strip() for p in row['blue_players']]; rpl=[p.strip() for p in row['red_players']]
        ds=row['date_str']; res=row['blue_win']
        btg=tg.get(b,0); rtg=tg.get(r,0)
        btw=tw.get(b,0)/btg if btg>0 else 0.5; rtw=tw.get(r,0)/rtg if rtg>0 else 0.5
        bcw=sum(cur_rate(cw,cg,c) for c in bp)/len(bp)
        rcw=sum(cur_rate(cw,cg,c) for c in rp)/len(rp)
        mk=tuple(sorted([b,r])); hr=h2h.get(mk,{}); ht=sum(hr.values())
        h2h_r=cap_h2h(hr.get(b,0)/ht) if ht>0 else 0.5
        bh=trec.get(b,[]); rh=trec.get(r,[])
        bf=weighted_form(bh,FORM_WINDOW); rf=weighted_form(rh,FORM_WINDOW)
        brwr=sum(bh[-RECENT_WINDOW:])/len(bh[-RECENT_WINDOW:]) if bh else 0.5
        rrwr=sum(rh[-RECENT_WINDOW:])/len(rh[-RECENT_WINDOW:]) if rh else 0.5
        def blended(pls,pks):
            rates=[]
            for i,(pl,c) in enumerate(zip(pls,pks)):
                pos=POSITIONS[i] if i<len(POSITIONS) else 'unknown'
                pcv=cur_rate(pc_w,pc_g,(pl,c),0.5,MIN_PC_GAMES)
                rcv=cur_rate(rc_w,rc_g,(pos,c),0.5,MIN_ROLE_GAMES)
                rates.append(PC_WEIGHT*pcv+RC_WEIGHT*rcv)
            return sum(rates)/len(rates) if rates else 0.5
        bpca=blended(bpl,bp); rpca=blended(rpl,rp)
        bg20=gg(ds,b,'avg_gd20'); rg20=gg(ds,r,'avg_gd20')
        bl=gg(ds,b,'late_scaling'); rl=gg(ds,r,'late_scaling')
        rows.append({
            'btwr':btw,'rtwr':rtw,'twd':btw-rtw,'btg':btg,'rtg':rtg,
            'bcwr':bcw,'rcwr':rcw,'wrd':bcw-rcw,'h2h':h2h_r,
            'bf':bf,'rf':rf,'fd':bf-rf,'brwr':brwr,'rrwr':rrwr,'rwrd':brwr-rrwr,
            'bsa':BLUE_SIDE_WINRATE,
            'bpca':bpca,'rpca':rpca,'pcad':bpca-rpca,
            'bg20':bg20,'rg20':rg20,'g20d':bg20-rg20,
            'bl':bl,'rl':rl,'ld':bl-rl,
        })
        tg[b]=btg+1; tg[r]=rtg+1
        tw[b]=tw.get(b,0)+res; tw[r]=tw.get(r,0)+(1-res)
        for c in bp: cg[c]=cg.get(c,0)+1; cw[c]=cw.get(c,0)+res
        for c in rp: cg[c]=cg.get(c,0)+1; cw[c]=cw.get(c,0)+(1-res)
        if mk not in h2h: h2h[mk]={}
        h2h[mk][b]=h2h[mk].get(b,0)+res; h2h[mk][r]=h2h[mk].get(r,0)+(1-res)
        trec.setdefault(b,[]).append(1 if res==1 else 0)
        trec.setdefault(r,[]).append(0 if res==1 else 1)
        for i,c in enumerate(bp):
            pos=POSITIONS[i] if i<len(POSITIONS) else 'unknown'
            rc_g[(pos,c)]=rc_g.get((pos,c),0)+1; rc_w[(pos,c)]=rc_w.get((pos,c),0)+res
        for i,c in enumerate(rp):
            pos=POSITIONS[i] if i<len(POSITIONS) else 'unknown'
            rc_g[(pos,c)]=rc_g.get((pos,c),0)+1; rc_w[(pos,c)]=rc_w.get((pos,c),0)+(1-res)
        for pl,c in zip(bpl,bp): pc_g[(pl,c)]=pc_g.get((pl,c),0)+1; pc_w[(pl,c)]=pc_w.get((pl,c),0)+res
        for pl,c in zip(rpl,rp): pc_g[(pl,c)]=pc_g.get((pl,c),0)+1; pc_w[(pl,c)]=pc_w.get((pl,c),0)+(1-res)

    feat = pd.DataFrame(rows)
    tr_m = (df['year']<=2025).values; te_m = (df['year']==2026).values
    mlb = MultiLabelBinarizer()
    mlb.fit((df[tr_m]['blue_picks']+df[tr_m]['red_picks']).tolist())
    be = pd.DataFrame(mlb.transform(df['blue_picks']), columns=['b_'+c for c in mlb.classes_]).reset_index(drop=True)
    re = pd.DataFrame(mlb.transform(df['red_picks']),  columns=['r_'+c for c in mlb.classes_]).reset_index(drop=True)
    X = pd.concat([be, re, feat.reset_index(drop=True)], axis=1)
    y = df['blue_win'].astype(int).reset_index(drop=True)
    print(f"  Train: {tr_m.sum()}  Test: {te_m.sum()}")
    print(f"  Features: {X.shape[1]}  (gold={'included' if has_gold else 'MISSING'})")
    print("  Training...")
    m = CalibratedClassifierCV(GradientBoostingClassifier(
        n_estimators=125,max_depth=2,learning_rate=0.1,random_state=42), method='isotonic',cv=5)
    m.fit(X[tr_m], y[tr_m])
    p = m.predict_proba(X[te_m])[:,1]
    yt = y[te_m].reset_index(drop=True)
    acc = accuracy_score(yt, (p>=0.5).astype(int))
    auc = roc_auc_score(yt, p)
    base = max(yt.mean(), 1-yt.mean())
    return {'n': te_m.sum(), 'acc': acc, 'auc': auc, 'base': base, 'edge': acc-base, 'gold': has_gold}


def run_ft5_model(ft5_file, label):
    print(f"\n{'='*60}")
    print(f"  FT5 MODEL — {label}")
    print(f"{'='*60}")
    ft5 = pd.read_csv(ft5_file)
    ft5['blue_team'] = ft5['blue_team'].apply(normalize_team)
    ft5['red_team'] = ft5['red_team'].apply(normalize_team)
    ft5['blue_picks'] = ft5['blue_picks'].astype(str).apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
    ft5['red_picks'] = ft5['red_picks'].astype(str).apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
    def to_bin(v):
        if isinstance(v, str):
            v = v.strip().lower()
            if v == 'blue': return 1
            if v == 'red': return 0
        return None
    ft5['ft5_bin'] = ft5['first_to_five'].apply(to_bin)
    ft5['is_ambiguous'] = pd.to_numeric(ft5['is_ambiguous'], errors='coerce').fillna(1).astype(int)
    ft5['blue_time'] = pd.to_numeric(ft5['blue_time'], errors='coerce')
    ft5['red_time'] = pd.to_numeric(ft5['red_time'], errors='coerce')
    ft5 = ft5[(ft5['ft5_bin'].notna()) & (ft5['is_ambiguous']==0)].copy()
    ft5['ft5_bin'] = ft5['ft5_bin'].astype(int)
    pp = pd.read_csv('proplay_matches.csv', usecols=['game_id','date','league'])
    pp['date'] = pd.to_datetime(pp['date'], errors='coerce')
    for c in ('date','league','year'):
        if c in ft5.columns: ft5 = ft5.drop(columns=[c])
    ft5 = ft5.merge(pp, on='game_id', how='inner')
    ft5['year'] = ft5['date'].dt.year
    ft5 = ft5[ft5['league'].isin(TARGET_LEAGUES)].copy()
    ft5 = ft5.sort_values('date').reset_index(drop=True)
    print(f"  Games: {len(ft5)}")

    cw=defaultdict(int); cg=defaultdict(int)
    tw=defaultdict(int); tg_d=defaultdict(int)
    at=defaultdict(float); tc=defaultdict(int)
    h2h=defaultdict(lambda: defaultdict(int)); trec=defaultdict(list)
    rows = []
    for _, row in ft5.iterrows():
        b,r = row['blue_team'], row['red_team']
        bp,rp = row['blue_picks'], row['red_picks']
        res = row['ft5_bin']
        def cr(c):
            g=cg[c]; return cw[c]/g if g>0 else 0.5
        ba=sum(cr(c) for c in bp)/len(bp) if bp else 0.5
        ra=sum(cr(c) for c in rp)/len(rp) if rp else 0.5
        btg=tg_d[b]; rtg=tg_d[r]
        be=tw[b]/btg if btg>0 else 0.5; re=tw[r]/rtg if rtg>0 else 0.5
        bs=at[b]/tc[b] if tc[b]>0 else None; rs=at[r]/tc[r] if tc[r]>0 else None
        mk=tuple(sorted([b,r])); hr=h2h[mk]; ht=sum(hr.values())
        h2h_r=cap_h2h(hr[b]/ht) if ht>0 else 0.5
        bf=weighted_form(trec[b],FORM_WINDOW); rf=weighted_form(trec[r],FORM_WINDOW)
        rows.append({
            'ba':ba,'ra':ra,'ad':ba-ra,'be':be,'re':re,'ed':be-re,
            'bs':bs,'rs':rs,'h2h':h2h_r,'bf':bf,'rf':rf,'fmd':bf-rf,
        })
        for c in bp: cg[c]+=1; cw[c]+=res
        for c in rp: cg[c]+=1; cw[c]+=(1-res)
        tg_d[b]+=1; tg_d[r]+=1; tw[b]+=res; tw[r]+=(1-res)
        bt=row['blue_time']; rt=row['red_time']
        if pd.notna(bt) and bt>0: at[b]+=bt; tc[b]+=1
        if pd.notna(rt) and rt>0: at[r]+=rt; tc[r]+=1
        h2h[mk][b]+=res; h2h[mk][r]+=(1-res)
        trec[b].append(res); trec[r].append(1-res)

    KSD = sum(at.values())/sum(tc.values()) if tc else 22.0
    feat = pd.DataFrame(rows)
    feat['bs'] = feat['bs'].fillna(KSD); feat['rs'] = feat['rs'].fillna(KSD)
    feat['sd'] = feat['rs'] - feat['bs']
    tr_m = (ft5['year']<=2025).values; te_m = (ft5['year']==2026).values
    mlb = MultiLabelBinarizer()
    mlb.fit((ft5[tr_m]['blue_picks']+ft5[tr_m]['red_picks']).tolist())
    be = pd.DataFrame(mlb.transform(ft5['blue_picks']), columns=['b_'+c for c in mlb.classes_]).reset_index(drop=True)
    re = pd.DataFrame(mlb.transform(ft5['red_picks']),  columns=['r_'+c for c in mlb.classes_]).reset_index(drop=True)
    X = pd.concat([be, re, feat.reset_index(drop=True)], axis=1)
    y = ft5['ft5_bin'].astype(int).reset_index(drop=True)
    print(f"  Train: {tr_m.sum()}  Test: {te_m.sum()}  Features: {X.shape[1]}")
    print("  Training...")
    m = CalibratedClassifierCV(GradientBoostingClassifier(
        n_estimators=125,max_depth=1,learning_rate=0.03,random_state=42), method='isotonic',cv=5)
    m.fit(X[tr_m], y[tr_m])
    p = m.predict_proba(X[te_m])[:,1]
    yt = y[te_m].reset_index(drop=True)
    acc = accuracy_score(yt, (p>=0.5).astype(int))
    auc = roc_auc_score(yt, p)
    base = max(yt.mean(), 1-yt.mean())
    return {'n': te_m.sum(), 'acc': acc, 'auc': auc, 'base': base, 'edge': acc-base}


def main():
    print("Building gold trajectory lookup...")
    gold, has_gold = build_gold_lookup()
    win = run_win_model(gold, has_gold)

    ft5_proxy = run_ft5_model('kill_timelines.csv',   'PROXY (v8.1 live)')
    ft5_v2 = None
    if Path('kill_timelines_v2.csv').exists():
        ft5_v2 = run_ft5_model('kill_timelines_v2.csv', 'PRECISE (v2)')

    print("\n" + "="*70)
    print("  V8.1 TRUE OUT-OF-SAMPLE ACCURACY")
    print("  Train: 2023-2025  |  Test: 2026")
    print("="*70)
    print(f"\n  WIN MODEL")
    print(f"    Test games:       {win['n']}")
    print(f"    Always-fav base:  {win['base']*100:.2f}%")
    print(f"    Accuracy:         {win['acc']*100:.2f}%")
    print(f"    AUC:              {win['auc']:.4f}")
    print(f"    Edge over base:   {win['edge']*100:+.2f}%")
    print(f"    Gold features:    {'included' if win['gold'] else 'MISSING'}")

    print(f"\n  FT5 MODEL — PROXY (what v8.1 actually uses)")
    print(f"    Test games:       {ft5_proxy['n']}")
    print(f"    Always-fav base:  {ft5_proxy['base']*100:.2f}%")
    print(f"    Accuracy:         {ft5_proxy['acc']*100:.2f}%")
    print(f"    AUC:              {ft5_proxy['auc']:.4f}")
    print(f"    Edge over base:   {ft5_proxy['edge']*100:+.2f}%")

    if ft5_v2:
        print(f"\n  FT5 MODEL — PRECISE (v2 lolesports data)")
        print(f"    Test games:       {ft5_v2['n']}")
        print(f"    Always-fav base:  {ft5_v2['base']*100:.2f}%")
        print(f"    Accuracy:         {ft5_v2['acc']*100:.2f}%")
        print(f"    AUC:              {ft5_v2['auc']:.4f}")
        print(f"    Edge over base:   {ft5_v2['edge']*100:+.2f}%")
        d = ft5_v2['acc'] - ft5_proxy['acc']
        print(f"    Delta vs proxy:   {d*100:+.2f}%")

    print(f"\n  WHAT THIS MEANS:")
    print(f"    These are what the models ACTUALLY achieve on games they've never seen.")
    print(f"    Compare against the CV estimates: ~65.4% win, ~57.9% FT5.")
    print(f"    If close to CV: model generalizes well. CV numbers are trustworthy.")
    print(f"    If much lower: model overfits to past. CV numbers were inflated.")


if __name__ == '__main__':
    main()
