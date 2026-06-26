"""
compare_ft5_full.py — FT5 lane+snowball test against the REAL v8.1 baseline.

The earlier compare_lane_snowball.py used a stripped-down FT5 baseline (5 features)
and got 48% accuracy, which is wrong — the real v8.1 model has 25 features and
hits ~58%. This script replicates the real baseline exactly, then adds lane
matchup + snowball features and measures the lift.

USAGE:
    python compare_ft5_full.py
"""
import warnings; warnings.filterwarnings('ignore')
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss

# Production constants
TARGET_LEAGUES = {'LCK','LPL','LEC','LCS','LCP','CBLOL','MSI','WLDs','Worlds',
                  'FST','LTA N','LTA S','LTA','EMEA Masters'}
FORM_WINDOW = 8
H2H_CAP = 0.60
ROLES = ['top','jungle','mid','adc','support']
LANE_SHRINK = 5
SNOWBALL_MIN_GAMES = 20


def cap(r): return max(1-H2H_CAP, min(H2H_CAP, r))
def weighted_form(hist, window):
    h = hist[-window:] if hist else []
    if not h: return 0.5
    w = list(range(1, len(h)+1))
    return sum(v*x for v,x in zip(h,w)) / sum(w)


def load_data():
    print("Loading data...")
    df = pd.read_csv('proplay_matches.csv')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).copy()
    df['blue_picks'] = df['blue_picks'].astype(str).apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
    df['red_picks']  = df['red_picks'].astype(str).apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
    df = df[df['blue_picks'].apply(len) == 5]
    df = df[df['red_picks'].apply(len) == 5]
    df['year'] = df['date'].dt.year
    df = df[df['league'].isin(TARGET_LEAGUES)].copy()

    # Merge with v2 FT5 data — same as production train_and_save.py
    v2 = pd.read_csv('kill_timelines_v2.csv')
    v2['first_to_five'] = v2['first_to_five'].astype(str).str.strip().str.lower()
    v2['ft5_binary'] = v2['first_to_five'].map({'blue':1, 'red':0})
    v2['is_ambiguous'] = pd.to_numeric(v2['is_ambiguous'], errors='coerce').fillna(1).astype(int)
    v2['blue_time'] = pd.to_numeric(v2['blue_time'], errors='coerce')
    v2['red_time']  = pd.to_numeric(v2['red_time'],  errors='coerce')
    # Filter to clean FT5 games (matches production)
    v2 = v2[(v2['ft5_binary'].notna()) & (v2['is_ambiguous'] == 0)].copy()
    v2['ft5_binary'] = v2['ft5_binary'].astype(int)
    df = df.merge(v2[['game_id','ft5_binary','blue_time','red_time']],
                  on='game_id', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  FT5-eligible tier-1 games: {len(df)}")
    print(f"  By year: {dict(df['year'].value_counts().sort_index())}")
    return df


def build_ft5_features(df, include_new):
    """Build the REAL v8.1 FT5 feature set (leak-free). If include_new,
    also adds lane matchup + snowball features.
    """
    # Running state — FT5-specific (matches production)
    ft5_champ_wins  = defaultdict(int); ft5_champ_games = defaultdict(int)
    ft5_team_wins   = defaultdict(int); ft5_team_games  = defaultdict(int)
    ft5_avg_time    = defaultdict(float); ft5_time_counts = defaultdict(int)
    ft5_h2h         = defaultdict(lambda: defaultdict(int))
    ft5_team_recent = defaultdict(list)

    # Lane matchup + snowball state (only used if include_new)
    lane_games = [defaultdict(int) for _ in range(5)]
    lane_wins  = [defaultdict(int) for _ in range(5)]
    cp_games = [defaultdict(int) for _ in range(5)]
    cp_wins  = [defaultdict(int) for _ in range(5)]
    # For FT5 model, snowball needs WIN labels (champ wins more when ahead).
    # We use df['blue_win'] for that, separate from the target ft5_binary.
    snow_games_ft5  = defaultdict(int); snow_wins_ft5  = defaultdict(int)
    snow_games_nft5 = defaultdict(int); snow_wins_nft5 = defaultdict(int)

    rows = []
    for _, r in df.iterrows():
        blue, red = r['blue_team'], r['red_team']
        bp, rp = r['blue_picks'], r['red_picks']
        result = int(r['ft5_binary'])  # 1 = blue first to 5, 0 = red
        win = int(r['blue_win']) if pd.notna(r.get('blue_win')) else None

        # === REAL v8.1 FT5 features ===
        def champ_ft5_rate(c):
            g = ft5_champ_games[c]
            return ft5_champ_wins[c]/g if g > 0 else 0.5
        b_agg = sum(champ_ft5_rate(c) for c in bp) / 5
        r_agg = sum(champ_ft5_rate(c) for c in rp) / 5
        b_tg = ft5_team_games[blue]; r_tg = ft5_team_games[red]
        b_e = ft5_team_wins[blue]/b_tg if b_tg > 0 else 0.5
        r_e = ft5_team_wins[red]/r_tg  if r_tg > 0 else 0.5
        b_s = ft5_avg_time[blue]/ft5_time_counts[blue] if ft5_time_counts[blue] > 0 else None
        r_s = ft5_avg_time[red]/ft5_time_counts[red]   if ft5_time_counts[red]   > 0 else None
        mk = tuple(sorted([blue, red]))
        hr = ft5_h2h[mk]; ht = sum(hr.values())
        h2h_rate = cap(hr[blue]/ht) if ht > 0 else 0.5
        b_form = weighted_form(ft5_team_recent[blue], FORM_WINDOW)
        r_form = weighted_form(ft5_team_recent[red],  FORM_WINDOW)

        feat = {
            'game_id': r['game_id'], 'year': r['year'], 'date': r['date'],
            'result': result,
            'b_aggression': b_agg, 'r_aggression': r_agg, 'aggression_diff': b_agg - r_agg,
            'b_early_rate': b_e, 'r_early_rate': r_e, 'early_rate_diff': b_e - r_e,
            'b_kill_speed': b_s, 'r_kill_speed': r_s,
            'h2h_early_rate': h2h_rate,
            'b_early_form': b_form, 'r_early_form': r_form, 'early_form_diff': b_form - r_form,
            'blue_picks': bp, 'red_picks': rp,
        }

        if include_new:
            # === Lane matchup features ===
            lane_advs = []
            for i in range(5):
                bc = bp[i]; rc = rp[i]
                key = (bc, rc)
                n  = lane_games[i][key]
                w  = lane_wins[i][key]
                cp_n = cp_games[i][bc]; cp_w = cp_wins[i][bc]
                prior = cp_w / cp_n if cp_n > 0 else 0.5
                shrunk = (w + LANE_SHRINK * prior) / (n + LANE_SHRINK) if (n + LANE_SHRINK) > 0 else 0.5
                lane_advs.append(shrunk - 0.5)
            feat['lane_adv_total'] = sum(lane_advs)
            for i, role in enumerate(ROLES):
                feat[f'lane_adv_{role}'] = lane_advs[i]

            # === Snowball features ===
            def snow_score(c):
                gf = snow_games_ft5[c]; wf = snow_wins_ft5[c]
                gn = snow_games_nft5[c]; wn = snow_wins_nft5[c]
                if gf + gn < SNOWBALL_MIN_GAMES:
                    return 0.0
                rf = wf/gf if gf > 0 else 0.5
                rn = wn/gn if gn > 0 else 0.5
                return rf - rn
            b_snow = sum(snow_score(c) for c in bp) / 5
            r_snow = sum(snow_score(c) for c in rp) / 5
            feat['b_snowball'] = b_snow
            feat['r_snowball'] = r_snow
            feat['snowball_diff'] = b_snow - r_snow

        rows.append(feat)

        # === Update state AFTER building feature row ===
        # FT5 champ/team/h2h tracking — labels are FT5 outcome (result)
        for c in bp:
            ft5_champ_games[c] += 1; ft5_champ_wins[c] += result
        for c in rp:
            ft5_champ_games[c] += 1; ft5_champ_wins[c] += (1 - result)
        ft5_team_games[blue] += 1; ft5_team_games[red] += 1
        ft5_team_wins[blue] += result; ft5_team_wins[red] += (1 - result)
        # Kill_speed accumulation — Session 1 fix: only count games where team
        # actually got to 5 kills (time < 30, the censored placeholder)
        bt = r['blue_time']; rt = r['red_time']
        if pd.notna(bt) and 0 < bt < 30.0:
            ft5_avg_time[blue] += bt; ft5_time_counts[blue] += 1
        if pd.notna(rt) and 0 < rt < 30.0:
            ft5_avg_time[red] += rt;  ft5_time_counts[red]  += 1
        ft5_h2h[mk][blue] += result; ft5_h2h[mk][red] += (1 - result)
        ft5_team_recent[blue].append(result)
        ft5_team_recent[red].append(1 - result)

        if include_new:
            # Lane matchup state updates use FT5 outcome (consistent w/ FT5 features)
            for i in range(5):
                bc = bp[i]; rc = rp[i]
                lane_games[i][(bc, rc)] += 1
                lane_wins[i][(bc, rc)]  += result
                lane_games[i][(rc, bc)] += 1
                lane_wins[i][(rc, bc)]  += (1 - result)
                cp_games[i][bc] += 1; cp_wins[i][bc] += result
                cp_games[i][rc] += 1; cp_wins[i][rc] += (1 - result)
            # Snowball uses WIN labels combined with FT5 outcome
            if win is not None:
                # ft5_binary == 1 means blue got FT5
                for c in bp:
                    if result == 1:
                        snow_games_ft5[c] += 1; snow_wins_ft5[c] += win
                    else:
                        snow_games_nft5[c] += 1; snow_wins_nft5[c] += win
                for c in rp:
                    if result == 0:
                        snow_games_ft5[c] += 1; snow_wins_ft5[c] += (1 - win)
                    else:
                        snow_games_nft5[c] += 1; snow_wins_nft5[c] += (1 - win)

    feat_df = pd.DataFrame(rows)

    # Apply kill_speed default (league mean of teams that have data)
    KSD = (sum(ft5_avg_time.values()) / sum(ft5_time_counts.values())
           if ft5_time_counts else 22.0)
    feat_df['b_kill_speed'] = feat_df['b_kill_speed'].fillna(KSD)
    feat_df['r_kill_speed'] = feat_df['r_kill_speed'].fillna(KSD)
    feat_df['speed_diff'] = feat_df['r_kill_speed'] - feat_df['b_kill_speed']
    return feat_df


def train_and_eval(feat_df, label):
    train_mask = (feat_df['year'] <= 2025).values
    test_mask  = (feat_df['year'] == 2026).values

    mlb = MultiLabelBinarizer()
    mlb.fit((feat_df[train_mask]['blue_picks'] + feat_df[train_mask]['red_picks']).tolist())
    blue_enc = pd.DataFrame(mlb.transform(feat_df['blue_picks']),
                            columns=['b_'+c for c in mlb.classes_]).reset_index(drop=True)
    red_enc  = pd.DataFrame(mlb.transform(feat_df['red_picks']),
                            columns=['r_'+c for c in mlb.classes_]).reset_index(drop=True)

    drop_cols = {'game_id','year','date','result','blue_picks','red_picks'}
    num_cols = [c for c in feat_df.columns if c not in drop_cols]
    X = pd.concat([blue_enc, red_enc, feat_df[num_cols].reset_index(drop=True)], axis=1)
    y = feat_df['result'].astype(int).reset_index(drop=True)

    X_train, y_train = X[train_mask], y[train_mask].reset_index(drop=True)
    X_test,  y_test  = X[test_mask],  y[test_mask].reset_index(drop=True)
    gid_test = feat_df['game_id'][test_mask].reset_index(drop=True)

    # Match production FT5 hyperparameters from train_and_save.py:
    # n_estimators=125, max_depth=1, lr=0.03
    base = GradientBoostingClassifier(n_estimators=125, max_depth=1,
                                       learning_rate=0.03, random_state=42)
    model = CalibratedClassifierCV(base, method='isotonic', cv=5)
    print(f"    Training {label} on {len(X_train)} games, {X_train.shape[1]} features...")
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:,1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    baseline = max(y_test.mean(), 1 - y_test.mean())
    return {
        'label': label, 'n_train': len(X_train), 'n_test': len(X_test),
        'n_features': X_train.shape[1],
        'acc': acc, 'auc': auc, 'brier': brier, 'baseline': baseline,
        'edge': acc - baseline, 'gid_test': gid_test, 'probs': probs,
        'preds': preds, 'y_test': y_test,
    }


def main():
    df = load_data()

    print("\n[1/2] Building CURRENT features (real v8.1 baseline)...")
    cur_feat = build_ft5_features(df, include_new=False)
    print(f"  Feature rows: {len(cur_feat)}")

    print("\n[2/2] Building NEW features (+ lane + snowball)...")
    new_feat = build_ft5_features(df, include_new=True)
    print(f"  Feature rows: {len(new_feat)}")

    print("\n" + "="*70)
    print("  TRAINING & EVALUATING")
    print("="*70)
    cur = train_and_eval(cur_feat, 'CURRENT')
    new = train_and_eval(new_feat, 'NEW')

    print("\n" + "="*70)
    print("  FT5 RESULTS  (test = 2026 games)")
    print("="*70)
    def line(r):
        return (f"  {r['label']:<10}  n={r['n_test']:>4}  feats={r['n_features']:>3}  "
                f"acc={r['acc']*100:.2f}%  AUC={r['auc']:.4f}  "
                f"edge={r['edge']*100:+.2f}%  brier={r['brier']:.4f}")
    print(line(cur)); print(line(new))
    d_acc = (new['acc'] - cur['acc']) * 100
    d_auc = new['auc'] - cur['auc']
    print(f"\n  Δacc: {d_acc:+.2f}%   Δauc: {d_auc:+.4f}")

    # Sanity check baseline against known v8.1 number
    if cur['acc'] < 0.55:
        print(f"\n  ⚠️  CURRENT accuracy = {cur['acc']*100:.2f}% — production was ~57.9%.")
        print(f"      Baseline may not be properly replicating v8.1. Treat deltas with caution.")
    elif cur['acc'] >= 0.55 and cur['acc'] <= 0.61:
        print(f"\n  ✓ CURRENT accuracy {cur['acc']*100:.2f}% is in the expected ~57-58% range.")
        print(f"     Baseline matches production well enough — deltas are trustworthy.")

    print("\n  INTERPRETATION:")
    if d_acc >= 1.0 and d_auc >= 0.01:
        print(f"    STRONG win for lane+snowball on FT5. Deploy.")
    elif d_acc >= 0.3 and d_auc >= 0.005:
        print(f"    MODEST win — deploy at discretion.")
    elif abs(d_acc) < 0.3 and abs(d_auc) < 0.005:
        print(f"    ESSENTIALLY TIED — no clear benefit to FT5 from these features.")
    elif d_acc < 0 or d_auc < -0.005:
        print(f"    NEGATIVE — features hurt FT5. Do NOT add them to FT5 model.")
    else:
        print(f"    MIXED signal — improvement on one metric but not both.")


if __name__ == '__main__':
    main()
