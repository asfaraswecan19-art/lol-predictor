"""
diagnose_signal.py — Information-ceiling test for the lolesports precise data.

Trains a minimal FT5 model using ONLY features derived from the precise kill
timeline data we scraped. No champion picks, no team rates from history, no
h2h. Just: kills_at_10min, kills_at_15min, and team-level running averages
of those.

If this model can hit ~60%+ accuracy on 2026 holdout, the precise data has
real predictive signal and a full FT5 rewrite is worth doing.

If it tops out at ~52-55%, the precise data is comparable to the proxy and
doesn't justify rewriting the model.

USAGE:
    python diagnose_signal.py
"""
import warnings; warnings.filterwarnings('ignore')
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

TARGET_LEAGUES = {'LCK','LPL','LEC','LCS','LCP','CBLOL','MSI','WLDs','Worlds',
                  'FST','LTA N','LTA S','LTA','EMEA Masters'}
FORM_WINDOW = 8


def to_binary_first_to_five(v):
    if isinstance(v, str):
        v = v.strip().lower()
        if v == 'blue': return 1
        if v == 'red':  return 0
    return None


def main():
    print("Loading data...")
    v2 = pd.read_csv('kill_timelines_v2.csv')
    pp = pd.read_csv('proplay_matches.csv', usecols=['game_id','date','league','game_duration_min'])
    pp['date'] = pd.to_datetime(pp['date'], errors='coerce')

    # Normalize types
    v2['first_to_five_binary'] = v2['first_to_five'].apply(to_binary_first_to_five)
    v2['is_ambiguous']         = pd.to_numeric(v2['is_ambiguous'], errors='coerce').fillna(1).astype(int)
    v2['blue_time']            = pd.to_numeric(v2['blue_time'], errors='coerce')
    v2['red_time']             = pd.to_numeric(v2['red_time'],  errors='coerce')
    v2['blue_kills10']         = pd.to_numeric(v2['blue_kills10'], errors='coerce')
    v2['red_kills10']          = pd.to_numeric(v2['red_kills10'],  errors='coerce')

    # Filter to non-ambiguous games with a clear winner
    v2 = v2[(v2['first_to_five_binary'].notna()) & (v2['is_ambiguous'] == 0)].copy()
    v2['first_to_five_binary'] = v2['first_to_five_binary'].astype(int)

    # CRITICAL: only keep games with PRECISE data (not proxy fallback)
    # Proxy fallback rows have blue_time = red_time = 30 and kills10 = 0.
    # We want only games where the times are real floats.
    v2['is_precise'] = ~(((v2['blue_time'] == 30.0) & (v2['red_time'] == 30.0))
                         | (v2['blue_time'].isna()) | (v2['red_time'].isna()))
    v2 = v2[v2['is_precise']].copy()
    print(f"  Precise-only games: {len(v2)}")

    # Drop conflicting cols then merge proplay
    for c in ('date','league','year'):
        if c in v2.columns: v2 = v2.drop(columns=[c])
    v2 = v2.merge(pp, on='game_id', how='inner')
    v2['year'] = v2['date'].dt.year
    v2 = v2[v2['league'].isin(TARGET_LEAGUES)].copy()
    v2 = v2.sort_values('date').reset_index(drop=True)
    print(f"  Tier-1 precise-only games: {len(v2)}")
    print(f"  By year: {dict(v2['year'].value_counts().sort_index())}")

    # Leak-free running state: per-team running averages of precise stats
    team_b_t5      = defaultdict(list)   # this team's prior blue_times (only when they were blue)
    team_r_t5      = defaultdict(list)   # this team's prior red_times (only when they were red)
    team_k10       = defaultdict(list)   # this team's prior kills_at_10
    team_k_against = defaultdict(list)   # opponents' kills_at_10 against this team
    team_t5        = defaultdict(list)   # this team's overall time-to-5 (whether blue or red)
    team_recent_ft5 = defaultdict(list)

    rows = []
    for _, r in v2.iterrows():
        blue, red = r['blue_team'], r['red_team']
        result = r['first_to_five_binary']
        bt5, rt5 = r['blue_time'], r['red_time']
        bk10, rk10 = r['blue_kills10'], r['red_kills10']

        # Features from prior data only
        def avg(lst, default):
            return sum(lst)/len(lst) if lst else default

        b_avg_t5    = avg(team_t5[blue],        22.0)
        r_avg_t5    = avg(team_t5[red],         22.0)
        b_avg_k10   = avg(team_k10[blue],       1.5)
        r_avg_k10   = avg(team_k10[red],        1.5)
        b_avg_kAg   = avg(team_k_against[blue], 1.5)  # how many kills blue typically gives up by 10min
        r_avg_kAg   = avg(team_k_against[red],  1.5)
        b_form      = avg(team_recent_ft5[blue][-FORM_WINDOW:], 0.5)
        r_form      = avg(team_recent_ft5[red][-FORM_WINDOW:],  0.5)

        rows.append({
            'game_id': r['game_id'], 'year': r['year'], 'result': result,
            'b_avg_t5': b_avg_t5, 'r_avg_t5': r_avg_t5, 't5_diff': b_avg_t5 - r_avg_t5,
            'b_avg_k10': b_avg_k10, 'r_avg_k10': r_avg_k10, 'k10_diff': b_avg_k10 - r_avg_k10,
            'b_avg_kAg': b_avg_kAg, 'r_avg_kAg': r_avg_kAg, 'kAg_diff': b_avg_kAg - r_avg_kAg,
            'b_form': b_form, 'r_form': r_form, 'form_diff': b_form - r_form,
            # also "net early dominance" — k10 - kAg
            'b_net_k10': b_avg_k10 - b_avg_kAg,
            'r_net_k10': r_avg_k10 - r_avg_kAg,
        })

        # Update state with the game we just processed
        if pd.notna(bt5) and bt5 < 30:
            team_t5[blue].append(bt5)
        if pd.notna(rt5) and rt5 < 30:
            team_t5[red].append(rt5)
        if pd.notna(bk10):
            team_k10[blue].append(bk10)
            team_k_against[red].append(bk10)  # bk10 is what red gave up
        if pd.notna(rk10):
            team_k10[red].append(rk10)
            team_k_against[blue].append(rk10)
        team_recent_ft5[blue].append(1 if result == 1 else 0)
        team_recent_ft5[red].append(0 if result == 1 else 1)

    feat = pd.DataFrame(rows)
    train = feat[feat['year'] <= 2025]
    test  = feat[feat['year'] == 2026]
    print(f"\nTrain: {len(train)}  Test: {len(test)}")

    feature_cols = ['b_avg_t5','r_avg_t5','t5_diff',
                    'b_avg_k10','r_avg_k10','k10_diff',
                    'b_avg_kAg','r_avg_kAg','kAg_diff',
                    'b_form','r_form','form_diff',
                    'b_net_k10','r_net_k10']
    X_tr, y_tr = train[feature_cols].values, train['result'].values
    X_te, y_te = test[feature_cols].values,  test['result'].values

    print("\nTraining minimal FT5 (gradient-boosted, precise-only features)...")
    model = GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.04, random_state=42)
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:,1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_te, preds)
    auc = roc_auc_score(y_te, probs)
    ll = log_loss(y_te, np.clip(probs, 1e-5, 1-1e-5))
    baseline = max(y_te.mean(), 1 - y_te.mean())

    print()
    print("="*60)
    print("  PRECISE-DATA SIGNAL DIAGNOSTIC")
    print("="*60)
    print(f"  Always-favorite baseline:  {baseline*100:.2f}%")
    print(f"  Precise-only model accuracy: {acc*100:.2f}%")
    print(f"  Precise-only model AUC:      {auc:.4f}")
    print(f"  Precise-only model log loss: {ll:.4f}")
    print(f"  Edge over baseline:          {(acc - baseline)*100:+.2f}%")
    print()

    # Feature importances
    importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
    print("Feature importances (top 10):")
    for name, imp in importances[:10]:
        bar = '#' * int(imp * 100)
        print(f"  {name:<14} {imp:.4f}  {bar}")

    print()
    print("INTERPRETATION:")
    if acc - baseline >= 0.04 and auc >= 0.60:
        print("  Precise data has STRONG signal. A full FT5 rewrite is worth it.")
        print("  Expect ~62-65% accuracy in the rewritten model.")
    elif acc - baseline >= 0.02 or auc >= 0.56:
        print("  Precise data has MODEST signal. Rewrite would help but not dramatically.")
        print("  Expect ~59-61% in a rewritten model.")
    else:
        print("  Precise data shows LITTLE signal beyond baseline.")
        print("  Rewriting the model probably won't move FT5 accuracy meaningfully.")
        print("  The scrape isn't wasted — but maybe better spent on a different model")
        print("  (e.g. total kills O/U, game duration) where these features fit naturally.")


if __name__ == '__main__':
    main()
