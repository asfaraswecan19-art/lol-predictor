"""
T2 PLAYER DATA DIAGNOSTIC
=========================
Question: is player data actually contributing anything to T2 predictions,
or is it being crushed to ~0.5 by shrinkage + the 10% PC_WEIGHT?

The player signal reaching the model is:
    PC_WEIGHT * shrunk_pc_rate + RC_WEIGHT * role_champ_rate
with PC_WEIGHT=0.10, RC_WEIGHT=0.90, and
    shrunk_pc_rate = (wins + K_PC*prior) / (games + K_PC), K_PC=12

If most (player, champ) pairs in T2 have few games, shrunk_pc_rate sits
near the prior (0.5) and contributes ~nothing distinctive -- meaning the
"player" part of the model is decorative.

This compares T1 vs T2 density and quantifies how far the shrunk rates
actually move away from the prior. No model training -- just data.

Run from the dataset builder folder. Paste output back.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

PC_WEIGHT = 0.10
RC_WEIGHT = 0.90
K_PC      = 12
K_ROLE    = 5
POSITIONS = ['top','jng','mid','adc','sup']

def shrunk_rate(wins, games, prior, k):
    return (wins + k*prior) / (games + k)

def analyze(path, label):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"\n{'='*62}\n{label}: FILE NOT FOUND ({path}) -- skipping\n{'='*62}")
        return None

    for c in ['blue_picks','red_picks','blue_players','red_players']:
        df[c] = df[c].apply(lambda x: [s.strip() for s in str(x).split(',')])

    pc_wins = defaultdict(int); pc_games = defaultdict(int)
    for _, row in df.iterrows():
        res = row['blue_win']
        for pl, c in zip(row['blue_players'], row['blue_picks']):
            pc_games[(pl,c)] += 1; pc_wins[(pl,c)] += res
        for pl, c in zip(row['red_players'], row['red_picks']):
            pc_games[(pl,c)] += 1; pc_wins[(pl,c)] += (1-res)

    games_per_pair = np.array(list(pc_games.values()))
    n_pairs = len(games_per_pair)

    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")
    print(f"  Total games:                {len(df)}")
    print(f"  Unique (player,champ) pairs: {n_pairs}")
    print(f"  Unique players:              {len(set(p for p,_ in pc_games))}")
    print(f"\n  Games per (player,champ) pair:")
    print(f"    mean:   {games_per_pair.mean():.2f}")
    print(f"    median: {np.median(games_per_pair):.1f}")
    print(f"    p75:    {np.percentile(games_per_pair,75):.1f}")
    print(f"    p90:    {np.percentile(games_per_pair,90):.1f}")
    print(f"    max:    {games_per_pair.max()}")

    print(f"\n  Distribution (how many pairs have N games):")
    for thresh in [1,2,3,5,8,12,20]:
        cnt = (games_per_pair >= thresh).sum()
        pct = cnt/n_pairs*100
        print(f"    >= {thresh:>2} games: {cnt:>6} pairs ({pct:5.1f}%)")

    # How much does shrinkage actually move the rate away from the prior?
    # (using flat 0.5 prior, which is what T2 currently uses)
    deviations = []
    for key, g in pc_games.items():
        w = pc_wins[key]
        sr = shrunk_rate(w, g, 0.5, K_PC)
        deviations.append(abs(sr - 0.5))
    deviations = np.array(deviations)

    print(f"\n  Shrunk PC rate deviation from prior (0.5):")
    print(f"    mean |rate - 0.5|:   {deviations.mean():.4f}")
    print(f"    median |rate - 0.5|: {np.median(deviations):.4f}")
    print(f"    p90 |rate - 0.5|:    {np.percentile(deviations,90):.4f}")
    print(f"    max |rate - 0.5|:    {deviations.max():.4f}")
    print(f"    (0.0 = shrinkage crushed it to prior, no player signal)")

    # After the 10% PC_WEIGHT, how much does player info move the blended feature?
    effective = deviations * PC_WEIGHT
    print(f"\n  AFTER PC_WEIGHT={PC_WEIGHT} -- actual contribution to blended feature:")
    print(f"    mean effective shift:   {effective.mean():.5f}")
    print(f"    median effective shift: {np.median(effective):.5f}")
    print(f"    p90 effective shift:    {np.percentile(effective,90):.5f}")
    print(f"    max effective shift:    {effective.max():.5f}")
    print(f"    (this is how far player identity moves blue_pc_avg, in win-rate units)")

    return {
        'label': label,
        'games': len(df),
        'pairs': n_pairs,
        'median_games': float(np.median(games_per_pair)),
        'mean_dev': float(deviations.mean()),
        'mean_effective': float(effective.mean()),
    }

r1 = analyze('proplay_matches.csv',    'TIER 1  (LCK/LPL/LEC/LCS/CBLOL/...)')
r2 = analyze('proplay_matches_t2.csv', 'TIER 2  (LCKC/LFL/EM/PRM)')

if r1 and r2:
    print(f"\n{'='*62}")
    print(f"  COMPARISON / VERDICT")
    print(f"{'='*62}")
    print(f"  {'metric':<34} {'T1':>12} {'T2':>12}")
    print(f"  {'-'*58}")
    print(f"  {'games':<34} {r1['games']:>12} {r2['games']:>12}")
    print(f"  {'(player,champ) pairs':<34} {r1['pairs']:>12} {r2['pairs']:>12}")
    print(f"  {'median games per pair':<34} {r1['median_games']:>12.1f} {r2['median_games']:>12.1f}")
    print(f"  {'mean |shrunk rate - 0.5|':<34} {r1['mean_dev']:>12.4f} {r2['mean_dev']:>12.4f}")
    print(f"  {'mean effective shift (x0.10)':<34} {r1['mean_effective']:>12.5f} {r2['mean_effective']:>12.5f}")
    ratio = r2['mean_effective']/r1['mean_effective'] if r1['mean_effective']>0 else 0
    print(f"\n  T2 player signal is {ratio*100:.0f}% as strong as T1's")
    if r2['mean_effective'] < 0.005:
        print(f"  --> T2 player contribution is NEGLIGIBLE (<0.005 win-rate units).")
        print(f"      PC_WEIGHT sweep should show ~no difference between 0.0 and 0.10,")
        print(f"      confirming player data is decorative for T2 as currently tuned.")
    else:
        print(f"  --> T2 player contribution is measurable. A PC_WEIGHT sweep is")
        print(f"      worth running to see if a higher weight helps.")
    print(f"{'='*62}\n")
