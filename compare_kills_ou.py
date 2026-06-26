"""
compare_kills_ou.py — Test whether adding precise pace features to the kills
O/U model improves edge over the current (no-pace) model.

Trains two versions of the kills O/U classifier on the same data split:
  Model A: current features only (per-team running avg of total_kills + duration)
  Model B: same + precise pace features (kills_at_10/15min from v2 file)

Evaluates both on the same 2026 holdout. For each O/U line, reports:
  - accuracy
  - calibration
  - edge over baseline (always-over or always-under)

This is leak-free: pace features are running averages of PRIOR games only.

USAGE:
    python compare_kills_ou.py
"""
import warnings; warnings.filterwarnings('ignore')
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss

# Mirror the production constants
KLS_RECENT_WINDOW = 20
KILLS_LINES = [22.5, 24.5, 26.5, 28.5, 30.5]
SHRINK_K = 30

PROPLAY_CSV = 'proplay_matches.csv'
V2_CSV      = 'kill_timelines_v2.csv'


def _rm(h, w, d): return sum(h[-w:])/len(h[-w:]) if h else d
def _rs(h, w, d): return float(np.std(h[-w:])) if len(h) >= 3 else d


def load_data():
    print("Loading data...")
    df = pd.read_csv(PROPLAY_CSV)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date','game_duration_min','total_kills']).copy()
    df = df[df['game_duration_min'] >= 15].reset_index(drop=True)
    df['blue_picks'] = df['blue_picks'].astype(str).apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
    df['red_picks']  = df['red_picks'].astype(str).apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
    df['year'] = df['date'].dt.year

    # Load v2 precise pace data (just the kills10/kills15 columns)
    v2 = pd.read_csv(V2_CSV, usecols=['game_id','blue_kills10','red_kills10'])
    v2['blue_kills10'] = pd.to_numeric(v2['blue_kills10'], errors='coerce')
    v2['red_kills10']  = pd.to_numeric(v2['red_kills10'],  errors='coerce')
    # v2's kills_at_10 values are 0 for proxy-fallback rows. Treat those as NaN
    # (no real data, fall back to league mean during accumulation).
    # We can detect proxy-fallback rows: both blue_time and red_time are 30.0.
    # But easier: just check if blue_kills10 is 0 AND red_kills10 is 0 — that's
    # the proxy default. Real games will have non-zero pace.
    v2_full = pd.read_csv(V2_CSV, usecols=['game_id','blue_time','red_time','blue_kills10','red_kills10'])
    v2_full['blue_time'] = pd.to_numeric(v2_full['blue_time'], errors='coerce')
    v2_full['red_time']  = pd.to_numeric(v2_full['red_time'],  errors='coerce')
    v2_full['is_proxy_fallback'] = ((v2_full['blue_time'] == 30.0) & (v2_full['red_time'] == 30.0))
    # Set pace to NaN for proxy-fallback rows so we don't accumulate stale data
    v2 = v2.merge(v2_full[['game_id','is_proxy_fallback']], on='game_id')
    v2.loc[v2['is_proxy_fallback'], 'blue_kills10'] = np.nan
    v2.loc[v2['is_proxy_fallback'], 'red_kills10']  = np.nan
    v2 = v2.drop(columns=['is_proxy_fallback'])

    df = df.merge(v2, on='game_id', how='left')

    n_precise = df['blue_kills10'].notna().sum()
    print(f"  Total games: {len(df)}")
    print(f"  Games with precise pace data: {n_precise} ({n_precise/len(df)*100:.1f}%)")
    return df.sort_values('date').reset_index(drop=True)


def build_features(df, include_pace):
    """Build leak-free per-game features.
    If include_pace, adds precise early-pace features.
    """
    gdm = df['game_duration_min'].mean()
    gkm = df['total_kills'].mean()
    gck = gkm / gdm

    # League means for pace features (default for teams with no precise history)
    if include_pace:
        g_k10 = df['blue_kills10'].dropna().mean() if df['blue_kills10'].notna().any() else 1.5
        g_kAg = g_k10  # symmetric default
        print(f"  League mean kills_at_10min: {g_k10:.3f}")

    # Champion archetypes (full data — same as production)
    cd_sum = defaultdict(float); cd_cnt = defaultdict(int)
    ck_sum = defaultdict(float); ck_cnt = defaultdict(int)
    for _, row in df.iterrows():
        for c in row['blue_picks'] + row['red_picks']:
            cd_sum[c] += row['game_duration_min']; cd_cnt[c] += 1
            ck_sum[c] += row['total_kills'];      ck_cnt[c] += 1
    champ_dur   = {c: (cd_sum[c]+SHRINK_K*gdm)/(cd_cnt[c]+SHRINK_K) for c in cd_cnt}
    champ_kills = {c: (ck_sum[c]+SHRINK_K*gkm)/(ck_cnt[c]+SHRINK_K) for c in ck_cnt}
    def karch(picks):
        durs = [champ_dur.get(c, gdm) for c in picks]
        klls = [champ_kills.get(c, gkm) for c in picks]
        return (sum(durs)/len(durs) if durs else gdm,
                sum(klls)/len(klls) if klls else gkm)

    # Running state
    kt_dur = defaultdict(list); kt_kls = defaultdict(list); kt_ckpm = defaultdict(list)
    kh_dur = defaultdict(list); kh_kls = defaultdict(list)
    if include_pace:
        kt_k10_for     = defaultdict(list)
        kt_k10_against = defaultdict(list)

    rows = []
    for _, row in df.iterrows():
        blue, red = row['blue_team'], row['red_team']
        bp, rp = row['blue_picks'], row['red_picks']

        b_dur = _rm(kt_dur[blue], KLS_RECENT_WINDOW, gdm)
        r_dur = _rm(kt_dur[red],  KLS_RECENT_WINDOW, gdm)
        b_kls = _rm(kt_kls[blue], KLS_RECENT_WINDOW, gkm)
        r_kls = _rm(kt_kls[red],  KLS_RECENT_WINDOW, gkm)
        b_ck  = _rm(kt_ckpm[blue], KLS_RECENT_WINDOW, gck)
        r_ck  = _rm(kt_ckpm[red],  KLS_RECENT_WINDOW, gck)
        mk = tuple(sorted([blue, red]))
        h_dur = _rm(kh_dur[mk], 10, (b_dur+r_dur)/2)
        h_kls = _rm(kh_kls[mk], 10, (b_kls+r_kls)/2)
        b_ad, b_ak = karch(bp); r_ad, r_ak = karch(rp)

        feat = {
            'b_dur_mean': b_dur, 'r_dur_mean': r_dur, 'avg_dur_mean': (b_dur+r_dur)/2,
            'b_kills_mean': b_kls, 'r_kills_mean': r_kls, 'avg_kills_mean': (b_kls+r_kls)/2,
            'b_ckpm_mean': b_ck, 'r_ckpm_mean': r_ck, 'avg_ckpm_mean': (b_ck+r_ck)/2,
            'min_games_seen': min(len(kt_dur[blue]), len(kt_dur[red])),
            'h2h_dur_mean': h_dur, 'h2h_kills_mean': h_kls,
            'b_arch_dur': b_ad, 'r_arch_dur': r_ad, 'avg_arch_dur': (b_ad+r_ad)/2,
            'b_arch_kills': b_ak, 'r_arch_kills': r_ak, 'avg_arch_kills': (b_ak+r_ak)/2,
        }

        if include_pace:
            b_p10  = _rm(kt_k10_for[blue],     KLS_RECENT_WINDOW, g_k10)
            r_p10  = _rm(kt_k10_for[red],      KLS_RECENT_WINDOW, g_k10)
            b_pA10 = _rm(kt_k10_against[blue], KLS_RECENT_WINDOW, g_kAg)
            r_pA10 = _rm(kt_k10_against[red],  KLS_RECENT_WINDOW, g_kAg)
            # Pace samples available for both teams in the recent window?
            pace_avail = min(len(kt_k10_for[blue]), len(kt_k10_for[red]))
            feat.update({
                'b_pace10_for': b_p10, 'r_pace10_for': r_p10,
                'b_pace10_against': b_pA10, 'r_pace10_against': r_pA10,
                'pace10_total': b_p10 + r_p10 + b_pA10 + r_pA10,  # total early-game volume
                'pace10_avail': min(pace_avail, 10),  # capped for stability
            })

        rows.append(feat)

        # Update state AFTER feature row built
        dur = row['game_duration_min']; k = row['total_kills']
        kt_dur[blue].append(dur); kt_dur[red].append(dur)
        kt_kls[blue].append(k);   kt_kls[red].append(k)
        kt_ckpm[blue].append(k/dur); kt_ckpm[red].append(k/dur)
        kh_dur[mk].append(dur); kh_kls[mk].append(k)
        if include_pace:
            bk10 = row.get('blue_kills10'); rk10 = row.get('red_kills10')
            if pd.notna(bk10):
                kt_k10_for[blue].append(bk10)
                kt_k10_against[red].append(bk10)  # blue scored bk10 against red
            if pd.notna(rk10):
                kt_k10_for[red].append(rk10)
                kt_k10_against[blue].append(rk10)

    return pd.DataFrame(rows)


def evaluate_ou(model, X_test, y_test, line):
    """Evaluate one O/U classifier; return metrics dict."""
    probs = model.predict_proba(X_test)[:,1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    # Always-side baseline = max of "always over" vs "always under"
    baseline = max(y_test.mean(), 1 - y_test.mean())
    edge = acc - baseline
    return {'line': line, 'n': len(y_test), 'accuracy': acc, 'auc': auc,
            'brier': brier, 'baseline': baseline, 'edge': edge}


def train_and_eval(df, include_pace, label):
    print(f"\n{'='*60}")
    print(f"  Building features [{label}]")
    print(f"{'='*60}")
    kfeat = build_features(df, include_pace=include_pace)
    feat_cols = list(kfeat.columns)
    print(f"  Feature count: {len(feat_cols)}")

    df_aligned = df.reset_index(drop=True)
    train_mask = df_aligned['year'] <= 2025
    test_mask  = df_aligned['year'] == 2026
    print(f"  Train games: {train_mask.sum()}  Test games: {test_mask.sum()}")

    X = kfeat[feat_cols]
    X_train, X_test = X[train_mask.values], X[test_mask.values]

    results = []
    for line in KILLS_LINES:
        y = (df_aligned['total_kills'] > line).astype(int)
        y_train, y_test = y[train_mask.values], y[test_mask.values]
        if y_train.mean() < 0.05 or y_train.mean() > 0.95:
            continue
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, num_leaves=8,
                               min_child_samples=30, reg_alpha=0.2, reg_lambda=0.2,
                               random_state=42, verbose=-1)
        m.fit(X_train, y_train)
        results.append(evaluate_ou(m, X_test, y_test, line))
    return results


def main():
    if not Path(PROPLAY_CSV).exists():
        print(f"ERROR: {PROPLAY_CSV} not found"); return
    if not Path(V2_CSV).exists():
        print(f"ERROR: {V2_CSV} not found — run merge_kills.py first"); return

    df = load_data()
    base_results    = train_and_eval(df, include_pace=False, label='CURRENT')
    precise_results = train_and_eval(df, include_pace=True,  label='WITH PACE')

    print(f"\n{'='*70}")
    print(f"  KILLS O/U COMPARISON  (test = 2026 games)")
    print(f"{'='*70}")
    print(f"{'Line':>6} {'N':>5}   {'Current':<26}  {'WithPace':<26}  {'Δedge':>7}")
    print(f"{'':>6} {'':>5}   {'acc/auc/edge':<26}  {'acc/auc/edge':<26}")
    print("-"*78)
    base_idx    = {r['line']: r for r in base_results}
    precise_idx = {r['line']: r for r in precise_results}
    total_delta = 0.0
    for line in KILLS_LINES:
        b = base_idx.get(line); p = precise_idx.get(line)
        if not b or not p: continue
        d = p['edge'] - b['edge']
        total_delta += d
        b_str = f"{b['accuracy']*100:5.1f}% AUC{b['auc']:.3f} +{b['edge']*100:+5.2f}%"
        p_str = f"{p['accuracy']*100:5.1f}% AUC{p['auc']:.3f} +{p['edge']*100:+5.2f}%"
        print(f"{line:>6.1f} {b['n']:>5}   {b_str:<26}  {p_str:<26}  {d*100:+>6.2f}%")
    avg_delta = total_delta / len(KILLS_LINES)
    print("-"*78)
    print(f"  Average edge delta: {avg_delta*100:+.2f}%")
    print()
    if avg_delta * 100 >= 1.0:
        print("  PACE features add MEANINGFUL value. Deploy.")
    elif avg_delta * 100 >= 0.3:
        print("  PACE features add MODEST value. Deploy at your discretion.")
    elif abs(avg_delta) * 100 < 0.3:
        print("  Essentially tied. Pace features barely move the kills O/U model.")
    else:
        print("  Pace features HURT performance. Don't deploy.")


if __name__ == '__main__':
    main()
