"""
WHICH FT5 SOURCE IS RIGHT? — proxy vs precise, adjudicated
==========================================================
We know proxy and precise disagree 27.6% of the time, systematically by
league (LPL 41%!) and year (2024+ worse). Neither being "more precise" tells
us which is CORRECT. This adjudicates using evidence independent of both.

Key context from fetch_kills.py: the lolesports feed returns 204/no_data for
many games, LPL especially. So the precise scrape may have thin/reconstructed
data exactly where it disagrees. This checks that directly.

Independent tiebreakers we have per game (from the precise CSV's own extra
columns + proxy):
  - blue_kills10 / red_kills10   : kills by 10 min (who's ahead early)
  - blue_fb / red_fb             : first blood (weak FT5 signal, ~71% concordant)
  - blue_time / red_time         : precise 5th-kill timestamps (precise's own)
  - proxy killsat windows        : the proxy's raw evidence

For each disagreement we ask: which source does the INDEPENDENT evidence back?
If precise wins the adjudication in leagues with good feed data but LOSES in
LPL (bad feed data), then precise is selectively unreliable -> FT10 viable
only where the feed is complete.

Requires: kill_timelines.csv (proxy), kill_timelines_v2.csv (precise),
proplay_matches.csv. Run from dataset builder folder.
"""

import pandas as pd
import numpy as np

TEAM_ALIASES = {'Team BDS':'Team Shifters','BDS':'Team Shifters'}
def norm(n): return TEAM_ALIASES.get(str(n).strip(), str(n).strip())

def load(path):
    df = pd.read_csv(path)
    df['first_to_five'] = df['first_to_five'].astype(str).str.strip().str.lower()
    for c in ['blue_time','red_time','blue_kills10','red_kills10','blue_fb','red_fb',
              'blue_golddiff10','red_golddiff10']:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors='coerce')
    if 'is_ambiguous' in df.columns:
        df['is_ambiguous']=pd.to_numeric(df['is_ambiguous'],errors='coerce').fillna(1).astype(int)
    else: df['is_ambiguous']=0
    return df

print("="*68)
print("  WHICH FT5 SOURCE IS RIGHT?")
print("="*68)

proxy   = load('kill_timelines.csv')
precise = load('kill_timelines_v2.csv')
pp = pd.read_csv('proplay_matches.csv', usecols=['game_id','date','league'])
pp['date']=pd.to_datetime(pp['date'],errors='coerce'); pp['year']=pp['date'].dt.year

# -----------------------------------------------------------------
# 1. PRECISE data quality per league: how "real" is the timing data?
#    A game whose 5th-kill times look degenerate (both exactly equal,
#    or suspiciously round) suggests reconstructed / thin feed data.
# -----------------------------------------------------------------
print("\n" + "-"*68)
print("1. Precise-scrape data quality per league")
print("-"*68)
prec = precise.merge(pp[['game_id','league','year']], on='game_id', how='left')
prec['both_times'] = (prec['blue_time']>0)&(prec['red_time']>0)
prec['tie_time']   = prec['both_times'] & (prec['blue_time']==prec['red_time'])
print(f"  {'league':<8}{'games':>7}{'has both times':>16}{'exact-tie times':>18}")
for lg,sub in prec.groupby('league'):
    if len(sub)>=50:
        bt=sub['both_times'].mean()*100
        tt=(sub['tie_time'].sum()/max(sub['both_times'].sum(),1))*100
        print(f"  {str(lg):<8}{len(sub):>7}{bt:>15.1f}%{tt:>17.1f}%")
print("  (low 'has both times' or high 'exact-tie' = degraded feed data)")

# -----------------------------------------------------------------
# 2. Adjudicate disagreements with INDEPENDENT evidence.
#    Use kills-by-10min: the team clearly ahead in kills at 10 is the
#    far more likely FT5 winner. For each disagreement, see which
#    source the kills10 evidence supports.
# -----------------------------------------------------------------
print("\n" + "-"*68)
print("2. Adjudicating disagreements with kills-at-10min evidence")
print("-"*68)
px = proxy[proxy['is_ambiguous']==0][['game_id','first_to_five']].rename(columns={'first_to_five':'proxy'})
pc = precise[precise['is_ambiguous']==0][['game_id','first_to_five','blue_kills10','red_kills10','blue_fb','red_fb']].rename(columns={'first_to_five':'prec'})
m = px.merge(pc, on='game_id', how='inner')
m = m[m['proxy'].isin(['blue','red']) & m['prec'].isin(['blue','red'])]
m = m.merge(pp[['game_id','league','year']], on='game_id', how='left')

# independent signal: who led kills at 10?
m['k10_leader'] = np.where(m['blue_kills10']>m['red_kills10'],'blue',
                   np.where(m['red_kills10']>m['blue_kills10'],'red','tie'))
# firstblood as secondary
m['fb_leader']  = np.where(m['blue_fb']==1,'blue',np.where(m['red_fb']==1,'red','none'))

dis = m[m['proxy']!=m['prec']].copy()
print(f"  Total shared non-ambiguous: {len(m)}")
print(f"  Disagreements: {len(dis)} ({len(dis)/len(m)*100:.1f}%)")

# On disagreements where kills10 has a clear leader, which source matches it?
adj = dis[dis['k10_leader'].isin(['blue','red'])].copy()
prox_right = (adj['proxy']==adj['k10_leader']).mean()
prec_right = (adj['prec']==adj['k10_leader']).mean()
print(f"\n  On disagreements with a clear kills@10 leader ({len(adj)} games):")
print(f"    kills@10 backs the PROXY label:   {prox_right*100:.1f}%")
print(f"    kills@10 backs the PRECISE label: {prec_right*100:.1f}%")
print("    (higher = that source is more often right on contested games)")

# -----------------------------------------------------------------
# 3. Same adjudication, split by league — the key question:
#    is precise right where feed is good, wrong where feed is bad (LPL)?
# -----------------------------------------------------------------
print("\n" + "-"*68)
print("3. Who's right on disagreements, BY LEAGUE (kills@10 adjudicator)")
print("-"*68)
print(f"  {'league':<8}{'disagr':>8}{'proxy right':>13}{'precise right':>15}  verdict")
for lg,sub in adj.groupby('league'):
    if len(sub)>=20:
        pr=(sub['proxy']==sub['k10_leader']).mean()
        pc_=(sub['prec']==sub['k10_leader']).mean()
        verdict = 'PROXY better' if pr>pc_+0.05 else ('PRECISE better' if pc_>pr+0.05 else 'tossup')
        print(f"  {str(lg):<8}{len(sub):>8}{pr*100:>12.1f}%{pc_*100:>14.1f}%  {verdict}")

# -----------------------------------------------------------------
# 4. Verdict synthesis
# -----------------------------------------------------------------
print("\n" + "-"*68)
print("4. SYNTHESIS")
print("-"*68)
print("""  Read together:
   - If precise has LOW 'has both times' / HIGH exact-ties in LPL, its LPL
     data is degraded -- and if kills@10 also says PROXY is better in LPL,
     then the precise scrape is the wrong one THERE specifically.
   - If precise wins the kills@10 adjudication in LCK/LEC/LCS but loses in
     LPL, the fix is: use precise labels only for good-feed leagues.
     That would make FT10 viable for those leagues.
   - If kills@10 backs the PROXY across the board, the precise scrape has a
     systematic FT5-extraction problem everywhere -> FT10 not viable from
     this scrape without fixing fetch_kills.py.
   - If kills@10 backs PRECISE across the board, then precise labels are
     actually CORRECT and 'precise underperforms' just means true FT5 is
     harder to predict than the firstblood-contaminated proxy -- a
     different (and important) conclusion.""")
print("="*68)
