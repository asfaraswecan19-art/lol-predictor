"""
test_win_leakage.py — how much does role_champ_rate leakage inflate the win model?
==================================================================================
FOUND: in backtester.py, role_champ_rate is computed by looping over the ENTIRE
dataset (including 2026 test games) BEFORE the walk-forward feature loop. Since
RC_WEIGHT = 0.90, that leaked statistic dominates the player-champion feature
(blue_pc_avg / red_pc_avg / pc_avg_diff) -- a core win-model input.

Every other win feature checked out clean (team winrate, champ winrate, h2h,
form, recent WR, player-champ rates, and gold_lookup are all properly
walk-forward: stats are read, THEN the current game's result is folded in).

This script runs the win backtest TWICE:
  A) LEAKED   — role_champ_rate from all years (what backtester.py does now)
  B) CLEAN    — role_champ_rate from train years only (<=2025)
The gap between them is the inflation.

Reads proplay_matches.csv directly; changes nothing.
"""
import pandas as pd, numpy as np
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_FILE='proplay_matches.csv'
POSITIONS=['top','jng','mid','bot','sup']
PC_WEIGHT=0.10; RC_WEIGHT=0.90
K_CHAMP=30; K_PC=12; K_ROLE=25; H2H_CAP=0.60
FORM_WINDOW=8; RECENT_WINDOW=20

def shrunk_rate(w,g,prior,k): return (w + k*prior)/(g+k) if (g+k)>0 else prior
def cap_h2h(r): return max(1-H2H_CAP, min(H2H_CAP, r))
def weighted_form(hist,window):
    h=hist[-window:]
    if not h: return 0.5
    wts=[i+1 for i in range(len(h))]
    return sum(v*w for v,w in zip(h,wts))/sum(wts)

print("="*66)
print("  WIN MODEL LEAKAGE TEST (role_champ_rate)")
print("="*66)

df=pd.read_csv(DATA_FILE)
df['date']=pd.to_datetime(df['date'],errors='coerce')
df=df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
df['year']=df['date'].dt.year
for c in ['blue_picks','red_picks','blue_players','red_players']:
    df[c]=df[c].apply(lambda x:[s.strip() for s in str(x).split(',')])
print(f"  games: {len(df)} | train<=2025: {(df['year']<=2025).sum()} | test 2026: {(df['year']==2026).sum()}")

def build_role_champ(scope_df):
    rw=defaultdict(int); rg=defaultdict(int)
    for _,row in scope_df.iterrows():
        res=row['blue_win']
        for i,ch in enumerate(row['blue_picks']):
            pos=POSITIONS[i] if i<len(POSITIONS) else 'unknown'
            rg[(pos,ch)]+=1; rw[(pos,ch)]+=res
        for i,ch in enumerate(row['red_picks']):
            pos=POSITIONS[i] if i<len(POSITIONS) else 'unknown'
            rg[(pos,ch)]+=1; rw[(pos,ch)]+=(1-res)
    return {k: shrunk_rate(rw[k],rg[k],0.5,K_ROLE) for k in rg}

def run(role_champ_rate, label):
    tw={}; tg={}; cw={}; cg={}; h2h={}; trec={}; pcw={}; pcg={}
    rows=[]
    for _,row in df.iterrows():
        b,r=row['blue_team'],row['red_team']
        bp,rp=row['blue_picks'],row['red_picks']
        bpl,rpl=row['blue_players'],row['red_players']
        bg=tg.get(b,0); rg_=tg.get(r,0)
        bwr=tw.get(b,0)/bg if bg>0 else 0.5
        rwr=tw.get(r,0)/rg_ if rg_>0 else 0.5
        bcwr=sum(shrunk_rate(cw.get(c,0),cg.get(c,0),0.5,K_CHAMP) for c in bp)/len(bp)
        rcwr=sum(shrunk_rate(cw.get(c,0),cg.get(c,0),0.5,K_CHAMP) for c in rp)/len(rp)
        mk=tuple(sorted([b,r])); hr=h2h.get(mk,{}); ht=sum(hr.values())
        h2hr=cap_h2h(hr.get(b,0)/ht) if ht>0 else 0.5
        bh=trec.get(b,[]); rh=trec.get(r,[])
        bform=weighted_form(bh,FORM_WINDOW); rform=weighted_form(rh,FORM_WINDOW)
        brwr=sum(bh[-RECENT_WINDOW:])/len(bh[-RECENT_WINDOW:]) if bh else 0.5
        rrwr=sum(rh[-RECENT_WINDOW:])/len(rh[-RECENT_WINDOW:]) if rh else 0.5
        bpc=[PC_WEIGHT*shrunk_rate(pcw.get((pl,c),0),pcg.get((pl,c),0),
              role_champ_rate.get((POSITIONS[i] if i<len(POSITIONS) else 'unknown',c),0.5),K_PC)
             +RC_WEIGHT*role_champ_rate.get((POSITIONS[i] if i<len(POSITIONS) else 'unknown',c),0.5)
             for i,(pl,c) in enumerate(zip(bpl,bp))]
        rpc=[PC_WEIGHT*shrunk_rate(pcw.get((pl,c),0),pcg.get((pl,c),0),
              role_champ_rate.get((POSITIONS[i] if i<len(POSITIONS) else 'unknown',c),0.5),K_PC)
             +RC_WEIGHT*role_champ_rate.get((POSITIONS[i] if i<len(POSITIONS) else 'unknown',c),0.5)
             for i,(pl,c) in enumerate(zip(rpl,rp))]
        bpca=sum(bpc)/len(bpc); rpca=sum(rpc)/len(rpc)
        rows.append({'b_wr':bwr,'r_wr':rwr,'wr_diff':bwr-rwr,'b_g':bg,'r_g':rg_,
                     'b_cwr':bcwr,'r_cwr':rcwr,'cwr_diff':bcwr-rcwr,'h2h':h2hr,
                     'b_form':bform,'r_form':rform,'form_diff':bform-rform,
                     'b_rwr':brwr,'r_rwr':rrwr,'rwr_diff':brwr-rrwr,
                     'b_pca':bpca,'r_pca':rpca,'pca_diff':bpca-rpca,'year':row['year']})
        res=row['blue_win']
        tg[b]=bg+1; tg[r]=rg_+1
        tw[b]=tw.get(b,0)+res; tw[r]=tw.get(r,0)+(1-res)
        for c in bp: cg[c]=cg.get(c,0)+1; cw[c]=cw.get(c,0)+res
        for c in rp: cg[c]=cg.get(c,0)+1; cw[c]=cw.get(c,0)+(1-res)
        h2h.setdefault(mk,{}); h2h[mk][b]=h2h[mk].get(b,0)+res; h2h[mk][r]=h2h[mk].get(r,0)+(1-res)
        trec.setdefault(b,[]); trec.setdefault(r,[])
        trec[b].append(res); trec[r].append(1-res)
        for pl,c in zip(bpl,bp): pcg[(pl,c)]=pcg.get((pl,c),0)+1; pcw[(pl,c)]=pcw.get((pl,c),0)+res
        for pl,c in zip(rpl,rp): pcg[(pl,c)]=pcg.get((pl,c),0)+1; pcw[(pl,c)]=pcw.get((pl,c),0)+(1-res)

    feat=pd.DataFrame(rows)
    tr=(feat['year']<=2025).values; te=(feat['year']==2026).values
    mlb=MultiLabelBinarizer(); mlb.fit((df[tr]['blue_picks']+df[tr]['red_picks']).tolist())
    be=pd.DataFrame(mlb.transform(df['blue_picks']),columns=['b_'+c for c in mlb.classes_]).reset_index(drop=True)
    re_=pd.DataFrame(mlb.transform(df['red_picks']),columns=['r_'+c for c in mlb.classes_]).reset_index(drop=True)
    X=pd.concat([be,re_,feat.drop(columns=['year'])],axis=1)
    y=df['blue_win'].reset_index(drop=True)
    m=GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,random_state=42)
    m.fit(X[tr],y[tr])
    p=m.predict_proba(X[te])[:,1]
    acc=accuracy_score(y[te],(p>=0.5).astype(int)); auc=roc_auc_score(y[te],p)
    fav=df[te].copy(); base=max((y[te]).mean(),1-(y[te]).mean())
    print(f"  [{label}] acc {acc*100:.2f}%  AUC {auc:.4f}  baseline {base*100:.2f}%  edge {(acc-base)*100:+.2f}%")
    return acc,auc

print("\n--- A) LEAKED: role_champ_rate from ALL years (current backtester) ---")
rc_leak = build_role_champ(df)
a_acc,a_auc = run(rc_leak, "leaked")

print("\n--- B) CLEAN: role_champ_rate from train years only (<=2025) ---")
rc_clean = build_role_champ(df[df['year']<=2025])
b_acc,b_auc = run(rc_clean, "clean ")

print("\n"+"="*66)
print(f"  INFLATION FROM LEAKAGE:  acc {(a_acc-b_acc)*100:+.2f} pts,  AUC {a_auc-b_auc:+.4f}")
if abs(a_acc-b_acc) < 0.005:
    print("  -> NEGLIGIBLE. The reported win-model number is essentially real.")
elif (a_acc-b_acc) >= 0.02:
    print("  -> SIGNIFICANT. The reported win accuracy is materially inflated;")
    print("     the CLEAN number is the honest one. Fix backtester.py.")
else:
    print("  -> SMALL but real. Worth fixing, but the model still works.")
print("="*66)
