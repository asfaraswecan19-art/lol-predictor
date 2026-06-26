"""
compare_ft5.py — Train two FT5 models, one on kill_timelines.csv (proxy) and
one on kill_timelines_v2.csv (precise), evaluate both on the same 2026 holdout.

Identical features, hyperparameters, and train/test splits — the ONLY
difference is the kill-timing source. Honest measurement of whether precise
timings actually help.

USAGE:
    python compare_ft5.py

OUTPUT:
    ft5_comparison.txt — full report
    (prints summary to console)
"""
import warnings; warnings.filterwarnings('ignore')
import csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, roc_auc_score, brier_score_loss,
                             log_loss)

# Config — mirror train_and_save.py exactly
FORM_WINDOW   = 8
RECENT_WINDOW = 20
H2H_CAP       = 0.60
TARGET_LEAGUES = {'LCK','LPL','LEC','LCS','LCP','CBLOL','MSI','WLDs','Worlds',
                  'FST','LTA N','LTA S','LTA','EMEA Masters'}


def cap(r): return max(1-H2H_CAP, min(H2H_CAP, r))


def weighted_form(hist, window):
    h = hist[-window:] if hist else []
    if not h: return 0.5
    w = list(range(1, len(h)+1))
    return sum(v*x for v,x in zip(h,w)) / sum(w)


def build_ft5_features_and_train(ft5_csv_path, proplay_path, label):
    """Build leak-free FT5 features and train CalibratedClassifierCV.
    Returns (model, mlb, train_X, train_y, test_X, test_y, n_train, n_test).
    """
    print(f"\n{'='*60}")
    print(f"  Training FT5 from: {ft5_csv_path}  [{label}]")
    print(f"{'='*60}")

    # Load FT5 timing data
    ft5 = pd.read_csv(ft5_csv_path)
    ft5['blue_picks'] = ft5['blue_picks'].astype(str).apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
    ft5['red_picks']  = ft5['red_picks'].astype(str).apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])

    # first_to_five can be 'blue'/'red' strings (old proxy CSV) OR 1/2 ints (new v2 CSV)
    # Normalize to a binary: 1 = blue first to 5, 0 = red first to 5
    def to_binary(v):
        if isinstance(v, str):
            v = v.strip().lower()
            if v == 'blue': return 1
            if v == 'red':  return 0
            return None
        try:
            iv = int(float(v))
            if iv == 1: return 1
            if iv == 2: return 0
        except (TypeError, ValueError):
            pass
        return None
    ft5['first_to_five_binary'] = ft5['first_to_five'].apply(to_binary)
    ft5['is_ambiguous'] = pd.to_numeric(ft5['is_ambiguous'], errors='coerce').fillna(1).astype(int)
    # Blue/red times for kill_speed feature
    ft5['blue_time'] = pd.to_numeric(ft5['blue_time'], errors='coerce')
    ft5['red_time']  = pd.to_numeric(ft5['red_time'],  errors='coerce')
    # Keep games where we have a clear winner and they aren't ambiguous
    ft5 = ft5[(ft5['first_to_five_binary'].notna()) & (ft5['is_ambiguous'] == 0)].copy()
    ft5['first_to_five_binary'] = ft5['first_to_five_binary'].astype(int)
    print(f"  Rows after FT5 filter: {len(ft5)}")

    # Merge with proplay to get date (and confirm tier-1)
    # Use suffixes to avoid column collisions if v2 already has 'year'/'league'
    pp = pd.read_csv(proplay_path, usecols=['game_id','date','league'])
    pp['date'] = pd.to_datetime(pp['date'], errors='coerce')
    # Drop columns from ft5 that conflict with pp before merging
    for col in ('date', 'league', 'year'):
        if col in ft5.columns:
            ft5 = ft5.drop(columns=[col])
    ft5 = ft5.merge(pp, on='game_id', how='inner')
    ft5['year'] = ft5['date'].dt.year
    ft5 = ft5[ft5['league'].isin(TARGET_LEAGUES)].copy()
    ft5 = ft5.sort_values('date').reset_index(drop=True)
    print(f"  Usable FT5 games: {len(ft5)}")

    # Leak-free feature building
    ft5_champ_wins  = defaultdict(int); ft5_champ_games = defaultdict(int)
    ft5_team_wins   = defaultdict(int); ft5_team_games  = defaultdict(int)
    ft5_avg_time    = defaultdict(float); ft5_time_counts = defaultdict(int)
    ft5_h2h         = defaultdict(lambda: defaultdict(int))
    ft5_team_recent = defaultdict(list)

    rows = []
    for _, r in ft5.iterrows():
        blue, red = r['blue_team'], r['red_team']
        bp, rp = r['blue_picks'], r['red_picks']
        result = r['first_to_five_binary']

        # Champion aggression (FT5 win rate) — prior games only
        def champ_rate(c):
            g = ft5_champ_games[c]
            return ft5_champ_wins[c]/g if g > 0 else 0.5
        b_agg = sum(champ_rate(c) for c in bp)/len(bp) if bp else 0.5
        r_agg = sum(champ_rate(c) for c in rp)/len(rp) if rp else 0.5
        # Team early rate
        b_tg = ft5_team_games[blue]; r_tg = ft5_team_games[red]
        b_e = ft5_team_wins[blue]/b_tg if b_tg > 0 else 0.5
        r_e = ft5_team_wins[red]/r_tg  if r_tg > 0 else 0.5
        # Kill speed
        b_s = ft5_avg_time[blue]/ft5_time_counts[blue] if ft5_time_counts[blue] > 0 else None
        r_s = ft5_avg_time[red]/ft5_time_counts[red]   if ft5_time_counts[red]   > 0 else None
        # H2H
        mk = tuple(sorted([blue, red]))
        hr = ft5_h2h[mk]; ht = sum(hr.values())
        h2h_rate = cap(hr[blue]/ht) if ht > 0 else 0.5
        # Form
        b_form = weighted_form(ft5_team_recent[blue], FORM_WINDOW)
        r_form = weighted_form(ft5_team_recent[red],  FORM_WINDOW)

        rows.append({
            'game_id': r['game_id'],
            'blue_aggression': b_agg, 'red_aggression': r_agg, 'aggression_diff': b_agg-r_agg,
            'blue_early_rate': b_e, 'red_early_rate': r_e, 'early_rate_diff': b_e-r_e,
            'blue_kill_speed': b_s, 'red_kill_speed': r_s,
            'h2h_early_rate': h2h_rate,
            'blue_early_form': b_form, 'red_early_form': r_form, 'early_form_diff': b_form-r_form,
            'year': r['year'], 'result': result,
            'blue_picks': bp, 'red_picks': rp,
        })

        # Update state
        for c in bp:
            ft5_champ_games[c] += 1; ft5_champ_wins[c] += result
        for c in rp:
            ft5_champ_games[c] += 1; ft5_champ_wins[c] += (1-result)
        ft5_team_games[blue] += 1; ft5_team_games[red] += 1
        ft5_team_wins[blue] += result; ft5_team_wins[red] += (1-result)
        # CRITICAL: only accumulate kill_speed from games where the team
        # actually reached 5 kills. Values >= 30.0 are censored placeholders
        # (the team didn't get to 5 in the game), not real kill-speed data.
        # Including them would systematically bias stomped teams to look slow.
        bt_val = r['blue_time']
        rt_val = r['red_time']
        if not pd.isna(bt_val) and 0 < bt_val < 30.0:
            ft5_avg_time[blue] += bt_val; ft5_time_counts[blue] += 1
        if not pd.isna(rt_val) and 0 < rt_val < 30.0:
            ft5_avg_time[red] += rt_val;  ft5_time_counts[red]  += 1
        ft5_h2h[mk][blue] += result; ft5_h2h[mk][red] += (1-result)
        ft5_team_recent[blue].append(1 if result == 1 else 0)
        ft5_team_recent[red].append(0 if result == 1 else 1)

    # Compute league mean kill speed (used as default for unknown teams)
    KSD = (sum(ft5_avg_time.values()) / sum(ft5_time_counts.values())
           if ft5_time_counts else 22.0)
    print(f"  Kill-speed default: {KSD:.2f}")

    feat = pd.DataFrame(rows)
    feat['blue_kill_speed'] = feat['blue_kill_speed'].fillna(KSD)
    feat['red_kill_speed']  = feat['red_kill_speed'].fillna(KSD)
    feat['speed_diff']      = feat['red_kill_speed'] - feat['blue_kill_speed']

    # MLB on all champions seen in train (2023-2025)
    train_mask = feat['year'] <= 2025
    test_mask  = feat['year'] == 2026
    print(f"  Train games (<=2025): {train_mask.sum()}")
    print(f"  Test games (2026):    {test_mask.sum()}")

    mlb = MultiLabelBinarizer()
    mlb.fit((feat[train_mask]['blue_picks'] + feat[train_mask]['red_picks']).tolist())
    blue_enc = pd.DataFrame(mlb.transform(feat['blue_picks']),
                            columns=['blue_'+c for c in mlb.classes_]).reset_index(drop=True)
    red_enc  = pd.DataFrame(mlb.transform(feat['red_picks']),
                            columns=['red_'+c  for c in mlb.classes_]).reset_index(drop=True)

    cols = ['blue_aggression','red_aggression','aggression_diff',
            'blue_early_rate','red_early_rate','early_rate_diff',
            'blue_kill_speed','red_kill_speed','speed_diff',
            'h2h_early_rate','blue_early_form','red_early_form','early_form_diff']
    X = pd.concat([blue_enc, red_enc, feat[cols].reset_index(drop=True)], axis=1)
    y = feat['result'].reset_index(drop=True)

    X_train, y_train = X[train_mask.values], y[train_mask.values]
    X_test,  y_test  = X[test_mask.values],  y[test_mask.values]
    test_game_ids = feat[test_mask]['game_id'].reset_index(drop=True)

    print(f"  Training FT5 (CalibratedClassifierCV / GBM)...")
    base = GradientBoostingClassifier(n_estimators=125, max_depth=1, learning_rate=0.03, random_state=42)
    model = CalibratedClassifierCV(base, method='isotonic', cv=5)
    model.fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test, test_game_ids, KSD


def evaluate(model, X_test, y_test, label):
    """Return dict of metrics."""
    probs = model.predict_proba(X_test)[:,1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    ll = log_loss(y_test, np.clip(probs, 1e-5, 1-1e-5))
    # Calibration per band
    bands = []
    for lo, hi in [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,1.0)]:
        conf = np.maximum(probs, 1-probs)
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() < 5: continue
        actual = (preds[mask] == y_test[mask]).mean() * 100
        expected = conf[mask].mean() * 100
        bands.append((lo, hi, mask.sum(), actual, expected, actual-expected))
    return {
        'label': label,
        'n_test': len(y_test),
        'accuracy': acc,
        'auc': auc,
        'brier': brier,
        'log_loss': ll,
        'bands': bands,
    }


def main():
    proplay = 'proplay_matches.csv'
    proxy_csv   = 'kill_timelines.csv'
    precise_csv = 'kill_timelines_v2.csv'

    if not Path(proxy_csv).exists():
        print(f"ERROR: {proxy_csv} missing"); return
    if not Path(precise_csv).exists():
        print(f"ERROR: {precise_csv} missing — run merge_kills.py first"); return

    # Train both
    m_proxy,   X_tr_p, y_tr_p, X_te_p, y_te_p, gid_p, ksd_p = build_ft5_features_and_train(proxy_csv,   proplay, 'PROXY')
    m_precise, X_tr_v, y_tr_v, X_te_v, y_te_v, gid_v, ksd_v = build_ft5_features_and_train(precise_csv, proplay, 'PRECISE')

    # Honest comparison: evaluate both models on the SAME set of test games
    # (the intersection — games that passed BOTH the proxy filter AND the v2 filter).
    shared = set(gid_p) & set(gid_v)
    print(f"\nTest set intersection: {len(shared)} games (proxy had {len(gid_p)}, precise had {len(gid_v)})")

    # Build index lookups
    p_idx = {gid: i for i, gid in enumerate(gid_p)}
    v_idx = {gid: i for i, gid in enumerate(gid_v)}

    # Restrict each test set to the shared game IDs (preserving each model's feature ordering)
    p_rows = [p_idx[g] for g in shared if g in p_idx]
    v_rows = [v_idx[g] for g in shared if g in v_idx]
    X_te_p_shared = X_te_p.iloc[p_rows].reset_index(drop=True)
    y_te_p_shared = y_te_p.iloc[p_rows].reset_index(drop=True)
    X_te_v_shared = X_te_v.iloc[v_rows].reset_index(drop=True)
    y_te_v_shared = y_te_v.iloc[v_rows].reset_index(drop=True)

    # Sanity: labels for the same game should agree (within is_ambiguous filter)
    # If not, that's a side-swap bug somewhere.
    gid_p_arr = gid_p.iloc[p_rows].reset_index(drop=True)
    gid_v_arr = gid_v.iloc[v_rows].reset_index(drop=True)
    # Align them by game_id
    p_labels = dict(zip(gid_p_arr, y_te_p_shared))
    v_labels = dict(zip(gid_v_arr, y_te_v_shared))
    label_disagree = sum(1 for g in shared if p_labels.get(g) != v_labels.get(g))
    print(f"Label disagreements on shared games: {label_disagree}/{len(shared)}")

    res_proxy   = evaluate(m_proxy,   X_te_p_shared, y_te_p_shared, 'PROXY')
    res_precise = evaluate(m_precise, X_te_v_shared, y_te_v_shared, 'PRECISE')

    # Build report
    out = []
    out.append("="*60)
    out.append("  FT5 COMPARISON: PROXY vs PRECISE")
    out.append("="*60)
    out.append(f"{'Metric':<20} {'Proxy':>12} {'Precise':>12} {'Delta':>10}")
    out.append("-"*60)
    out.append(f"{'Test games':<20} {res_proxy['n_test']:>12} {res_precise['n_test']:>12}")
    out.append(f"{'Accuracy':<20} {res_proxy['accuracy']*100:>11.2f}% {res_precise['accuracy']*100:>11.2f}% {(res_precise['accuracy']-res_proxy['accuracy'])*100:+>9.2f}%")
    out.append(f"{'AUC':<20} {res_proxy['auc']:>12.4f} {res_precise['auc']:>12.4f} {res_precise['auc']-res_proxy['auc']:+>10.4f}")
    out.append(f"{'Brier (lower=bet)':<20} {res_proxy['brier']:>12.4f} {res_precise['brier']:>12.4f} {res_precise['brier']-res_proxy['brier']:+>10.4f}")
    out.append(f"{'Log loss (lower=bet)':<20} {res_proxy['log_loss']:>12.4f} {res_precise['log_loss']:>12.4f} {res_precise['log_loss']-res_proxy['log_loss']:+>10.4f}")

    out.append("")
    out.append("CALIBRATION BY BAND (PROXY)")
    out.append(f"  {'Band':>10} {'Games':>6} {'Acc%':>7} {'Exp%':>7} {'Diff':>7}")
    for lo, hi, n, a, e, d in res_proxy['bands']:
        out.append(f"  {lo*100:.0f}%-{hi*100:.0f}%  {n:>6} {a:>6.1f}% {e:>6.1f}% {d:+>6.1f}")
    out.append("")
    out.append("CALIBRATION BY BAND (PRECISE)")
    out.append(f"  {'Band':>10} {'Games':>6} {'Acc%':>7} {'Exp%':>7} {'Diff':>7}")
    for lo, hi, n, a, e, d in res_precise['bands']:
        out.append(f"  {lo*100:.0f}%-{hi*100:.0f}%  {n:>6} {a:>6.1f}% {e:>6.1f}% {d:+>6.1f}")

    out.append("")
    out.append("INTERPRETATION:")
    delta_acc = (res_precise['accuracy'] - res_proxy['accuracy']) * 100
    delta_auc = res_precise['auc'] - res_proxy['auc']
    if delta_acc > 1.0 and delta_auc > 0.01:
        out.append(f"  PRECISE wins meaningfully: +{delta_acc:.2f}% accuracy, +{delta_auc:.4f} AUC.")
        out.append(f"  Recommend: deploy precise timings.")
    elif delta_acc > 0.3 or delta_auc > 0.005:
        out.append(f"  PRECISE wins modestly: +{delta_acc:.2f}% accuracy, +{delta_auc:.4f} AUC.")
        out.append(f"  Improvement is real but small. Deploy at your discretion.")
    elif abs(delta_acc) < 0.3 and abs(delta_auc) < 0.005:
        out.append(f"  ESSENTIALLY TIED. {delta_acc:+.2f}% accuracy, {delta_auc:+.4f} AUC.")
        out.append(f"  The kill-speed feature has small weight in the FT5 model — switching")
        out.append(f"  data sources barely moves predictions. Deploy or not, your choice.")
    else:
        out.append(f"  PROXY is comparable or better: {delta_acc:+.2f}% accuracy, {delta_auc:+.4f} AUC.")
        out.append(f"  Recommend: stay on proxy data. The merge work isn't wasted (data is")
        out.append(f"  still in the file), but no need to deploy this version.")

    text = "\n".join(out)
    Path('ft5_comparison.txt').write_text(text, encoding='utf-8')
    print("\n" + text)
    print(f"\nWrote: ft5_comparison.txt")


if __name__ == '__main__':
    main()
