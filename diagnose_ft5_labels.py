"""
WHY DO PRECISE FT5 LABELS UNDERPERFORM? — label diagnostic (no model)
=====================================================================
Symptom: identical features, precise labels 49.8% vs proxy 56.9%. Below chance
from labels alone means the labels must be wrong INCONSISTENTLY between train
and test (consistent-wrong would score ~50%, not below).

This trains NOTHING. It cross-examines the two label sources on shared games
and checks each hypothesis for the below-chance behavior:

  H1  blue/red TEAM ASSIGNMENT differs between the precise scrape and
      proplay_matches.csv (so 'blue' means opposite things) -- top suspect.
  H2  game_id merge misalignment (ids don't correspond 1:1).
  H3  precise first_to_five genuinely disagrees with proxy on many games.
  H4  precise first_to_five contradicts its OWN blue_time/red_time columns
      (internal polarity flip).

It prints which hypothesis fits, and whether disagreement is systematic
(by year / tournament) -- which is what makes it hurt train vs test.

Requires: proplay_matches.csv, kill_timelines.csv (proxy),
kill_timelines_v2.csv (precise). Run from dataset builder folder.
"""

import pandas as pd
import numpy as np

TEAM_ALIASES = {'Team BDS':'Team Shifters','BDS':'Team Shifters'}
def norm(n): return TEAM_ALIASES.get(str(n).strip(), str(n).strip())

def load_labels(path):
    df = pd.read_csv(path)
    df['first_to_five'] = df['first_to_five'].astype(str).str.strip().str.lower()
    df['blue_time'] = pd.to_numeric(df.get('blue_time'), errors='coerce')
    df['red_time']  = pd.to_numeric(df.get('red_time'),  errors='coerce')
    if 'is_ambiguous' in df.columns:
        df['is_ambiguous'] = pd.to_numeric(df['is_ambiguous'],errors='coerce').fillna(1).astype(int)
    else:
        df['is_ambiguous'] = 0
    if 'blue_team' in df.columns: df['blue_team']=df['blue_team'].apply(norm)
    if 'red_team'  in df.columns: df['red_team'] =df['red_team'].apply(norm)
    return df

print("="*66)
print("  FT5 LABEL DIAGNOSTIC — why does precise underperform?")
print("="*66)

proxy   = load_labels('kill_timelines.csv')
precise = load_labels('kill_timelines_v2.csv')
pp = pd.read_csv('proplay_matches.csv', usecols=['game_id','date','league','blue_team','red_team'])
pp['blue_team']=pp['blue_team'].apply(norm); pp['red_team']=pp['red_team'].apply(norm)
pp['date']=pd.to_datetime(pp['date'],errors='coerce'); pp['year']=pp['date'].dt.year

print(f"\nRow counts: proxy={len(proxy)}  precise={len(precise)}  proplay={len(pp)}")

# -----------------------------------------------------------------
# H4 first: is the PRECISE label internally consistent with its own times?
# -----------------------------------------------------------------
print("\n" + "-"*66)
print("H4: does precise first_to_five agree with its own blue_time/red_time?")
print("-"*66)
pc = precise[(precise['blue_time']>0)&(precise['red_time']>0)&
             (precise['first_to_five'].isin(['blue','red']))].copy()
pc['time_says'] = np.where(pc['blue_time']<pc['red_time'],'blue','red')
agree_self = (pc['time_says']==pc['first_to_five']).mean()
print(f"  Games with both times + a label: {len(pc)}")
print(f"  first_to_five matches (blue_time<red_time): {agree_self*100:.1f}%")
if agree_self < 0.90:
    print("  *** INTERNAL POLARITY BUG: the label contradicts its own timing.")
    print("      first_to_five is likely flipped relative to blue_time/red_time.")
else:
    print("  OK -- precise label is internally consistent with its timings.")

# -----------------------------------------------------------------
# H1/H2: do the two sources describe the SAME game the SAME WAY?
# -----------------------------------------------------------------
print("\n" + "-"*66)
print("H1/H2: team assignment + game_id alignment vs proplay_matches")
print("-"*66)
prec_m = precise.merge(pp, on='game_id', how='inner', suffixes=('_prec','_pp'))
print(f"  precise rows merged onto proplay by game_id: {len(prec_m)} / {len(precise)}")
if len(prec_m)==0:
    print("  *** game_id does not align at all -- H2 confirmed (merge failure).")
else:
    if 'blue_team_prec' in prec_m.columns and 'blue_team_pp' in prec_m.columns:
        same_blue = (prec_m['blue_team_prec']==prec_m['blue_team_pp']).mean()
        swapped   = ((prec_m['blue_team_prec']==prec_m['red_team_pp']) &
                     (prec_m['red_team_prec']==prec_m['blue_team_pp'])).mean()
        print(f"  blue_team matches proplay:  {same_blue*100:.1f}%")
        print(f"  blue/red SWAPPED vs proplay: {swapped*100:.1f}%")
        if swapped > 0.05:
            print("  *** TEAM-ASSIGNMENT MISMATCH (H1): on some games the precise")
            print("      scrape has blue/red swapped vs proplay. Since champion")
            print("      features come from proplay's blue/red, the label points")
            print("      the wrong way on exactly those games -> anti-signal.")
        elif same_blue < 0.90:
            print("  *** Teams don't line up even though ids merged -- suspicious.")
        else:
            print("  OK -- blue/red assignment agrees with proplay.")

# -----------------------------------------------------------------
# H3: how often do PRECISE and PROXY labels agree on shared games?
# -----------------------------------------------------------------
print("\n" + "-"*66)
print("H3: precise vs proxy label agreement on shared games")
print("-"*66)
common_cols = ['game_id','first_to_five','is_ambiguous']
pxa = proxy[proxy['is_ambiguous']==0][['game_id','first_to_five']].rename(columns={'first_to_five':'proxy_ft5'})
pca = precise[precise['is_ambiguous']==0][['game_id','first_to_five']].rename(columns={'first_to_five':'prec_ft5'})
both = pxa.merge(pca, on='game_id', how='inner')
both = both[both['proxy_ft5'].isin(['blue','red']) & both['prec_ft5'].isin(['blue','red'])]
print(f"  Shared non-ambiguous games: {len(both)}")
if len(both)>0:
    agree = (both['proxy_ft5']==both['prec_ft5']).mean()
    print(f"  Labels agree: {agree*100:.1f}%")
    if agree < 0.55:
        print("  *** Near-random agreement -- labels are scrambled relative to")
        print("      each other (points to H1 swap or H4 flip).")
    elif agree < 0.80:
        print("  *** Substantial disagreement -- enough to flip a weak model")
        print("      below chance on the test slice.")
    else:
        print(f"  Fairly high agreement -- disagreement alone ({(1-agree)*100:.0f}%) may not")
        print("      fully explain below-chance; combine with systematic pattern below.")

    # is the disagreement SYSTEMATIC? (that's what breaks train vs test)
    both2 = both.merge(pp[['game_id','year','league']], on='game_id', how='left')
    both2['disagree'] = (both2['proxy_ft5']!=both2['prec_ft5']).astype(int)
    print("\n  Disagreement rate by year (systematic = breaks train/test):")
    for yr,sub in both2.groupby('year'):
        if len(sub)>=10:
            print(f"    {int(yr) if pd.notna(yr) else '?'}: {sub['disagree'].mean()*100:5.1f}%  ({len(sub)} games)")
    print("\n  Disagreement rate by league (top 8 by size):")
    lg = both2.groupby('league').agg(dis=('disagree','mean'),n=('disagree','size'))
    for lgname,rw in lg.sort_values('n',ascending=False).head(8).iterrows():
        print(f"    {str(lgname):<8} {rw['dis']*100:5.1f}%  ({int(rw['n'])} games)")

# -----------------------------------------------------------------
# Direct test: does FLIPPING the precise label recover agreement?
# -----------------------------------------------------------------
print("\n" + "-"*66)
print("Sanity: if we FLIP precise labels, does agreement invert?")
print("-"*66)
if len(both)>0:
    flip = {'blue':'red','red':'blue'}
    both['prec_flipped'] = both['prec_ft5'].map(flip)
    agree_flip = (both['proxy_ft5']==both['prec_flipped']).mean()
    print(f"  Agreement with precise FLIPPED: {agree_flip*100:.1f}%")
    if agree_flip > 0.80 and agree < 0.55:
        print("  *** SMOKING GUN: flipping precise labels restores agreement.")
        print("      The precise labels are polarity-inverted. Fix the flip and")
        print("      precise labels should work (and FT10 becomes viable).")

print("\n" + "="*66)
print("  Read the flagged (***) lines above -- they point at the cause.")
print("="*66)
