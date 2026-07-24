"""
FT5 PRECISE-vs-PROXY RETEST — isolating kill_speed
==================================================
The earlier verdict was "precise data is worse" (proxy 57%, precise 49%).
But 49% is BELOW chance, which overfitting alone can't produce, and the
diagnosis pinned it on the kill_speed feature overfitting when it has real
variance from precise timestamps.

kill_speed = a team's historical average time-to-5-kills. It's built FROM
the same event being predicted, and it's noisy. With proxy data it's nearly
constant (5-min buckets) so the model ignores it. With precise data it varies
and the GBM chases it.

That means the never-run test is: does the PRECISE LABEL help once you drop
the kill_speed FEATURE? This script runs THREE arms on the identical
train(<=2025)/test(2026) split so we can separate label quality from the
feature artifact:

  A) PROXY  labels + full features (kill_speed included)  = current v8.1
  B) PRECISE labels + full features (kill_speed included) = the "49%" test
  C) PRECISE labels, kill_speed feature REMOVED           = the missing test

If C >= A, precise labels are fine (or better) and the whole "precise is
worse" conclusion was really "kill_speed-as-a-feature is worse." That would
mean we should switch FT5 to precise labels + drop kill_speed.

Requires: proplay_matches.csv, kill_timelines.csv (proxy),
kill_timelines_v2.csv (precise), and the OE raw files (for gold, optional).
Run from the dataset builder folder. Paste the full output.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score

FORM_WINDOW = 8
H2H_CAP = 0.60
TARGET_LEAGUES = ['LCK','LPL','LEC','LCS','CBLOL','MSI','WLDs','EWC','LTA N','LTA S','LTA','FST']

def cap_h2h(r): return max(1-H2H_CAP, min(H2H_CAP, r))
def weighted_form(hist, w):
    h = hist[-w:] if hist else []
    if not h: return 0.5
    wt = list(range(1, len(h)+1))
    return sum(v*x for v,x in zip(h,wt))/sum(wt)

def run_ft5(ft5_file, label, drop_kill_speed):
    print(f"\n--- {label} ---")
    ft5 = pd.read_csv(ft5_file)
    ft5['blue_picks'] = ft5['blue_picks'].astype(str).apply(lambda x:[c.strip() for c in x.split(',') if c.strip()])
    ft5['red_picks']  = ft5['red_picks'].astype(str).apply(lambda x:[c.strip() for c in x.split(',') if c.strip()])
    def to_bin(v):
        if isinstance(v,str):
            v=v.strip().lower()
            if v=='blue': return 1
            if v=='red':  return 0
        return None
    ft5['ft5_bin'] = ft5['first_to_five'].apply(to_bin)
    ft5['is_ambiguous'] = pd.to_numeric(ft5['is_ambiguous'],errors='coerce').fillna(1).astype(int)
    ft5['blue_time'] = pd.to_numeric(ft5['blue_time'],errors='coerce')
    ft5['red_time']  = pd.to_numeric(ft5['red_time'],errors='coerce')
    ft5 = ft5[(ft5['ft5_bin'].notna()) & (ft5['is_ambiguous']==0)].copy()
    ft5['ft5_bin'] = ft5['ft5_bin'].astype(int)

    pp = pd.read_csv('proplay_matches.csv', usecols=['game_id','date','league'])
    pp['date'] = pd.to_datetime(pp['date'],errors='coerce')
    for c in ('date','league','year'):
        if c in ft5.columns: ft5 = ft5.drop(columns=[c])
    ft5 = ft5.merge(pp, on='game_id', how='inner')
    ft5['year'] = ft5['date'].dt.year
    ft5 = ft5[ft5['league'].isin(TARGET_LEAGUES)].copy()
    ft5 = ft5.sort_values('date').reset_index(drop=True)

    cw=defaultdict(int); cg=defaultdict(int)
    tw=defaultdict(int); tg_d=defaultdict(int)
    at=defaultdict(float); tc=defaultdict(int)
    h2h=defaultdict(lambda: defaultdict(int)); trec=defaultdict(list)
    rows=[]
    for _,row in ft5.iterrows():
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
        rows.append({'ba':ba,'ra':ra,'ad':ba-ra,'be':be,'re':re,'ed':be-re,
                     'bs':bs,'rs':rs,'h2h':h2h_r,'bf':bf,'rf':rf,'fmd':bf-rf})
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

    # THE KEY KNOB: drop the kill_speed-derived features entirely
    if drop_kill_speed:
        feat = feat.drop(columns=['bs','rs','sd'])

    tr_m = (ft5['year']<=2025).values; te_m = (ft5['year']==2026).values
    mlb = MultiLabelBinarizer()
    mlb.fit((ft5[tr_m]['blue_picks']+ft5[tr_m]['red_picks']).tolist())
    be = pd.DataFrame(mlb.transform(ft5['blue_picks']), columns=['b_'+c for c in mlb.classes_]).reset_index(drop=True)
    re = pd.DataFrame(mlb.transform(ft5['red_picks']),  columns=['r_'+c for c in mlb.classes_]).reset_index(drop=True)
    X = pd.concat([be, re, feat.reset_index(drop=True)], axis=1)
    y = ft5['ft5_bin'].astype(int).reset_index(drop=True)
    print(f"  Games: {len(ft5)} | Train {tr_m.sum()} Test {te_m.sum()} | Features {X.shape[1]} | kill_speed {'DROPPED' if drop_kill_speed else 'included'}")
    m = CalibratedClassifierCV(GradientBoostingClassifier(
        n_estimators=125,max_depth=1,learning_rate=0.03,random_state=42), method='isotonic',cv=5)
    m.fit(X[tr_m], y[tr_m])
    p = m.predict_proba(X[te_m])[:,1]
    yt = y[te_m].reset_index(drop=True)
    acc = accuracy_score(yt,(p>=0.5).astype(int))
    auc = roc_auc_score(yt,p)
    base = max(yt.mean(), 1-yt.mean())
    print(f"  Acc {acc*100:.2f}%  AUC {auc:.4f}  base {base*100:.2f}%  edge {(acc-base)*100:+.2f}%")
    return {'label':label,'n':int(te_m.sum()),'acc':acc,'auc':auc,'base':base}

print("="*64)
print("  FT5 RETEST — isolating the kill_speed feature")
print("="*64)
A = run_ft5('kill_timelines.csv',    'A) PROXY labels + kill_speed',  drop_kill_speed=False)
B = run_ft5('kill_timelines_v2.csv', 'B) PRECISE labels + kill_speed', drop_kill_speed=False)
C = run_ft5('kill_timelines_v2.csv', 'C) PRECISE labels, NO kill_speed', drop_kill_speed=True)
# and a fair 4th: proxy WITHOUT kill_speed, to be sure the comparison is clean
D = run_ft5('kill_timelines.csv',    'D) PROXY labels, NO kill_speed', drop_kill_speed=True)

print("\n" + "="*64)
print("  SUMMARY")
print("="*64)
for r in (A,B,C,D):
    print(f"  {r['label']:<36} Acc {r['acc']*100:5.2f}%  AUC {r['auc']:.4f}")
print()
print(f"  B - A (precise vs proxy, both w/ kill_speed): {(B['acc']-A['acc'])*100:+.2f}%")
print(f"  C - A (precise NO-speed vs proxy current):    {(C['acc']-A['acc'])*100:+.2f}%")
print(f"  C - B (dropping kill_speed on precise):       {(C['acc']-B['acc'])*100:+.2f}%")
print()
print("  READ:")
print("   - If B is ~49% (below chance) but C jumps up near/above A, then")
print("     the '49%' was the kill_speed FEATURE overfitting, NOT the precise")
print("     labels. Precise labels are fine; drop kill_speed and use them.")
print("   - If C is still well below A, precise labels genuinely underperform")
print("     and the original conclusion holds -- keep the proxy.")
print("   - Compare C vs D: precise-no-speed vs proxy-no-speed is the cleanest")
print("     label-quality comparison (same feature set, only labels differ).")
print("="*64)
