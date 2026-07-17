"""
META REPORT — what's meta right now, and what's spiking
======================================================
Pure information tool. No model, no backtest. Answers:

  1. META PICKS PER ROLE  — what's actually being played right now, with
     play rate (share of that role's picks) and win rate over a recent window.

  2. EMERGING PICKS       — champions whose PLAY RATE is spiking, i.e. taking
     a bigger share of a role than they were before. This catches things like
     Viktor/Syndra ADC while they're still only a handful of games, because
     play rate is measurable long before win rate is meaningful.

IMPORTANT — READ THE WIN RATES CAREFULLY:
  Emerging picks have small samples BY DEFINITION. A 4-game champ at 75% is
  not "a 75% winrate champ" -- it's 3 wins out of 4. The PLAY RATE is the
  real signal (teams are choosing to pick it); the win rate on <10 games is
  mostly noise. This report shows raw win rate AND game count so you can see
  the difference, and flags anything under MIN_GAMES_FOR_WR as unreliable.

Config at the top: window sizes, league filter, minimum games.

Run from the dataset builder folder.
Usage:
    python meta_report.py                 # all T1 leagues, last 30 days
    python meta_report.py --days 45       # custom window
    python meta_report.py --league LCK    # single league
    python meta_report.py --role adc      # single role
"""

import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

POSITIONS=['top','jng','mid','adc','sup']
TEAM_ALIASES={'Team BDS':'Team Shifters','BDS':'Team Shifters'}

MIN_GAMES_FOR_WR = 10   # below this, win rate is flagged as unreliable
TOP_N_META       = 8    # meta picks to show per role
TOP_N_EMERGING   = 6    # emerging picks to show per role

ap=argparse.ArgumentParser()
ap.add_argument('--days',   type=int, default=30, help='recent window (days) for "meta now"')
ap.add_argument('--base',   type=int, default=90, help='baseline window (days) to compare against for emergence')
ap.add_argument('--league', type=str, default=None, help='filter to one league (e.g. LCK)')
ap.add_argument('--role',   type=str, default=None, help='filter to one role (top/jng/mid/adc/sup)')
ap.add_argument('--file',   type=str, default='proplay_matches.csv')
args=ap.parse_args()

df=pd.read_csv(args.file)
df['date']=pd.to_datetime(df['date'],errors='coerce')
df=df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
for c in ['blue_picks','red_picks']:
    df[c]=df[c].apply(lambda x:[s.strip() for s in str(x).split(',')])
if args.league:
    df=df[df['league'].str.upper()==args.league.upper()].reset_index(drop=True)
    if len(df)==0:
        raise SystemExit(f"No games found for league '{args.league}'. "
                         f"Available: {sorted(pd.read_csv(args.file)['league'].unique())}")

latest = df['date'].max()
cut_recent = latest - pd.Timedelta(days=args.days)
cut_base   = latest - pd.Timedelta(days=args.base)

print("="*72)
print(f"  META REPORT")
print("="*72)
print(f"  Data through:    {latest.date()}")
print(f"  Recent window:   last {args.days} days  ({cut_recent.date()} -> {latest.date()})")
print(f"  Baseline window: {args.base}d ago -> {args.days}d ago  (for spike detection)")
if args.league: print(f"  League filter:   {args.league.upper()}")
print(f"  Games in file:   {len(df)}")

# ---- explode into (date, role, champ, won) rows ----
recs=[]
for _,row in df.iterrows():
    res=row['blue_win']
    for i,c in enumerate(row['blue_picks'][:5]):
        recs.append({'date':row['date'],'role':POSITIONS[i],'champ':c,'won':res,'league':row['league']})
    for i,c in enumerate(row['red_picks'][:5]):
        recs.append({'date':row['date'],'role':POSITIONS[i],'champ':c,'won':1-res,'league':row['league']})
picks=pd.DataFrame(recs)

recent = picks[picks['date']>=cut_recent]
base   = picks[(picks['date']>=cut_base)&(picks['date']<cut_recent)]

print(f"  Picks in recent window:   {len(recent)}")
print(f"  Picks in baseline window: {len(base)}")
if len(recent)==0:
    raise SystemExit("\nNo games in the recent window. Try a larger --days value.")

roles = [args.role.lower()] if args.role else POSITIONS

# =====================================================================
# 1. META PICKS PER ROLE
# =====================================================================
print("\n" + "="*72)
print(f"  CURRENT META — most-picked per role (last {args.days} days)")
print("="*72)

for role in roles:
    r_recent = recent[recent['role']==role]
    total = len(r_recent)
    if total==0:
        print(f"\n  {role.upper()}: no picks in window")
        continue
    g = r_recent.groupby('champ').agg(games=('won','size'), wins=('won','sum'))
    g['play_rate'] = g['games']/total*100
    g['win_rate']  = g['wins']/g['games']*100
    g = g.sort_values('games',ascending=False).head(TOP_N_META)

    print(f"\n  {role.upper()}  ({total} picks in window)")
    print(f"    {'champion':<16}{'picks':>7}{'play%':>8}{'win%':>8}   note")
    print(f"    {'-'*54}")
    for champ,r in g.iterrows():
        note = '' if r['games']>=MIN_GAMES_FOR_WR else f"win% unreliable (n={int(r['games'])})"
        print(f"    {champ:<16}{int(r['games']):>7}{r['play_rate']:>7.1f}%{r['win_rate']:>7.1f}%   {note}")

# =====================================================================
# 2. EMERGING PICKS — play rate spiking
# =====================================================================
print("\n" + "="*72)
print(f"  EMERGING — play rate rising vs previous {args.base-args.days} days")
print("="*72)
print("  play% now vs play% before. This is the signal that catches new meta")
print("  picks EARLY -- play rate moves before win rate is measurable.")

any_emerging=False
for role in roles:
    r_recent = recent[recent['role']==role]
    r_base   = base[base['role']==role]
    tot_recent = len(r_recent); tot_base = len(r_base)
    if tot_recent==0: continue

    rc = r_recent.groupby('champ').agg(games=('won','size'), wins=('won','sum'))
    rc['pr_now'] = rc['games']/tot_recent*100
    rc['wr'] = rc['wins']/rc['games']*100

    if tot_base>0:
        bc = r_base.groupby('champ').agg(games_base=('won','size'))
        bc['pr_before'] = bc['games_base']/tot_base*100
    else:
        bc = pd.DataFrame(columns=['games_base','pr_before'])

    m = rc.join(bc, how='left')
    m['games_base'] = m['games_base'].fillna(0)
    m['pr_before']  = m['pr_before'].fillna(0.0)
    m['delta']      = m['pr_now'] - m['pr_before']
    m['is_new']     = m['games_base']==0

    em = m[m['delta']>0].sort_values('delta',ascending=False).head(TOP_N_EMERGING)
    em = em[em['games']>=2]   # at least 2 games to be worth mentioning
    if len(em)==0: continue
    any_emerging=True

    print(f"\n  {role.upper()}")
    print(f"    {'champion':<16}{'picks':>7}{'play% now':>11}{'was':>8}{'change':>9}{'win%':>8}  flag")
    print(f"    {'-'*68}")
    for champ,r in em.iterrows():
        flag=[]
        if r['is_new']: flag.append('NEW')
        if r['games']<MIN_GAMES_FOR_WR: flag.append(f"win% noise n={int(r['games'])}")
        print(f"    {champ:<16}{int(r['games']):>7}{r['pr_now']:>10.1f}%{r['pr_before']:>7.1f}%"
              f"{r['delta']:>+8.1f}%{r['wr']:>7.1f}%  {', '.join(flag)}")

if not any_emerging:
    print("\n  (nothing meaningfully rising in this window)")

# =====================================================================
# 3. OFF-META / NOVEL PICKS — champs appearing in unusual roles
# =====================================================================
print("\n" + "="*72)
print(f"  OFF-ROLE PICKS — champs in roles they rarely occupy historically")
print("="*72)
print("  Catches things like Viktor/Syndra ADC: a champion showing up in a")
print("  role where it has little/no history. Flagged on PLAY, not win rate.")

# historical role distribution per champ (all data before the recent window)
hist = picks[picks['date']<cut_recent]
hist_role = hist.groupby(['champ','role']).size().rename('n').reset_index()
hist_tot  = hist.groupby('champ').size().rename('total')
hist_role = hist_role.join(hist_tot, on='champ')
hist_role['share'] = hist_role['n']/hist_role['total']
hist_lookup = {(r['champ'],r['role']): r['share'] for _,r in hist_role.iterrows()}
hist_seen   = set(hist['champ'])

offrole=[]
for role in roles:
    r_recent = recent[recent['role']==role]
    tot = len(r_recent)
    if tot==0: continue
    g = r_recent.groupby('champ').agg(games=('won','size'), wins=('won','sum'))
    for champ,r in g.iterrows():
        share = hist_lookup.get((champ,role), 0.0)
        if champ not in hist_seen:
            continue   # brand new champ entirely, not an off-role case
        if share < 0.05 and r['games']>=2:   # <5% of this champ's history in this role
            offrole.append({'role':role,'champ':champ,'games':int(r['games']),
                            'wr':r['wins']/r['games']*100,
                            'hist_share':share*100})

if offrole:
    od=pd.DataFrame(offrole).sort_values('games',ascending=False)
    print(f"\n    {'role':<6}{'champion':<16}{'picks':>7}{'win%':>8}{'hist role share':>18}")
    print(f"    {'-'*56}")
    for _,r in od.iterrows():
        print(f"    {r['role']:<6}{r['champ']:<16}{r['games']:>7}{r['wr']:>7.1f}%"
              f"{r['hist_share']:>17.1f}%")
    print(f"\n    (hist role share = how much of this champ's ALL-TIME picks were")
    print(f"     in this role. Near 0% = genuinely novel placement.)")
else:
    print("\n    (no notable off-role picks in this window)")

print("\n" + "="*72)
print("  REMINDER: win rates on <10 games are noise. Play rate is the signal.")
print("="*72 + "\n")
