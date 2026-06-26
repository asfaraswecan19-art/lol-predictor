"""
compare_lane_snowball.py — Test whether adding lane-matchup and snowball-draft
features improves the win and FT5 models.

For each model (win, FT5), trains two versions side-by-side on the same data:
  CURRENT: existing features (your live v8.1 model approximation)
  NEW:     same + lane matchups + snowball score

Both versions are leak-free: all features use prior games only.

Lane matchup feature design:
  For each (position, champ_pair) historical win rate, Bayesian-shrunk toward
  the champion's position prior. Per game: 5 lane advantages + total +
  per-lane breakdown.

Snowball feature design:
  Per champion, "snowball score" = how much that champion's win rate increases
  when its team gets the FT5 advantage. Team snowball = sum across 5 picks.

USAGE:
    python compare_lane_snowball.py
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

# Mirror production constants
TARGET_LEAGUES = {'LCK','LPL','LEC','LCS','LCP','CBLOL','MSI','WLDs','Worlds',
                  'FST','LTA N','LTA S','LTA','EMEA Masters'}
FORM_WINDOW = 8
H2H_CAP = 0.60
ROLES = ['top','jungle','mid','adc','support']

# Lane matchup Bayesian shrinkage strength
LANE_SHRINK = 5

# Snowball: how many games a champion needs before we trust its snowball score
SNOWBALL_MIN_GAMES = 20


def cap(r): return max(1-H2H_CAP, min(H2H_CAP, r))
def weighted_form(hist, window):
    h = hist[-window:] if hist else []
    if not h: return 0.5
    w = list(range(1, len(h)+1))
    return sum(v*x for v,x in zip(h,w)) / sum(w)


# ====================================================================
# DATA LOADING
# ====================================================================
def load_data():
    print("Loading data...")
    df = pd.read_csv('proplay_matches.csv')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date','blue_win']).copy()
    df['blue_picks'] = df['blue_picks'].astype(str).apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
    df['red_picks']  = df['red_picks'].astype(str).apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
    df = df[df['blue_picks'].apply(len) == 5]
    df = df[df['red_picks'].apply(len) == 5]
    df['year'] = df['date'].dt.year
    df = df[df['league'].isin(TARGET_LEAGUES)].copy()
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  Total tier-1 games (5v5 valid picks): {len(df)}")

    # FT5 labels from kill_timelines_v2
    v2 = pd.read_csv('kill_timelines_v2.csv',
                     usecols=['game_id','first_to_five','is_ambiguous'])
    def to_binary(v):
        if isinstance(v, str):
            v = v.strip().lower()
            if v == 'blue': return 1
            if v == 'red':  return 0
        return None
    v2['ft5_binary'] = v2['first_to_five'].apply(to_binary)
    v2['is_ambiguous'] = pd.to_numeric(v2['is_ambiguous'], errors='coerce').fillna(1).astype(int)
    v2 = v2[(v2['ft5_binary'].notna()) & (v2['is_ambiguous'] == 0)].copy()
    v2['ft5_binary'] = v2['ft5_binary'].astype(int)

    df = df.merge(v2[['game_id','ft5_binary']], on='game_id', how='left')
    print(f"  Games with non-ambiguous FT5 label: {df['ft5_binary'].notna().sum()}")
    return df


# ====================================================================
# FEATURE BUILDER — handles both targets (win + ft5) and both modes
# ====================================================================
def build_features(df, target_col, include_new):
    """Build features chronologically with running state.
    target_col: 'blue_win' (1 = blue won) or 'ft5_binary' (1 = blue FT5)
    include_new: if True, also computes lane matchup + snowball features

    Returns: (DataFrame of features, Series of labels, Series of years, Series of game_ids)
    Rows with NaN target are dropped from features+labels.
    """
    # Filter to rows with non-null target
    sub = df.dropna(subset=[target_col]).copy()
    sub[target_col] = sub[target_col].astype(int)
    sub = sub.reset_index(drop=True)

    # Standard win-model running state
    champ_wins  = defaultdict(int);  champ_games  = defaultdict(int)
    team_wins   = defaultdict(int);  team_games   = defaultdict(int)
    h2h         = defaultdict(lambda: defaultdict(int))
    team_recent = defaultdict(list)

    # NEW: lane matchup state
    # role -> (champ_in_role, champ_in_role) -> wins for first slot
    # We track ordered (myChamp, oppChamp) -> {games, wins}
    lane_games = [defaultdict(int) for _ in range(5)]
    lane_wins  = [defaultdict(int) for _ in range(5)]
    # Prior: champ-in-position win rate
    cp_games = [defaultdict(int) for _ in range(5)]
    cp_wins  = [defaultdict(int) for _ in range(5)]

    # NEW: snowball state — per champion, win rate split by "team got FT5" vs not
    # We need FT5 outcome to compute this. If df has ft5_binary, use it.
    # Snowball score = (champ_winrate_when_ft5) - (champ_winrate_when_not_ft5)
    snow_games_ft5  = defaultdict(int); snow_wins_ft5  = defaultdict(int)
    snow_games_nft5 = defaultdict(int); snow_wins_nft5 = defaultdict(int)

    rows = []
    n_ft5_known = 0
    for _, r in sub.iterrows():
        blue, red = r['blue_team'], r['red_team']
        bp, rp = r['blue_picks'], r['red_picks']
        target = r[target_col]

        # === Standard features ===
        def champ_rate(c):
            g = champ_games[c]
            return champ_wins[c]/g if g > 0 else 0.5
        b_cr = sum(champ_rate(c) for c in bp)/5
        r_cr = sum(champ_rate(c) for c in rp)/5
        b_tg = team_games[blue]; r_tg = team_games[red]
        b_tw = team_wins[blue]/b_tg if b_tg > 0 else 0.5
        r_tw = team_wins[red]/r_tg  if r_tg > 0 else 0.5
        mk = tuple(sorted([blue, red]))
        hr = h2h[mk]; ht = sum(hr.values())
        h2h_rate = cap(hr[blue]/ht) if ht > 0 else 0.5
        b_form = weighted_form(team_recent[blue], FORM_WINDOW)
        r_form = weighted_form(team_recent[red],  FORM_WINDOW)

        feat = {
            'game_id': r['game_id'], 'year': r['year'], 'date': r['date'],
            'b_champ_rate': b_cr, 'r_champ_rate': r_cr, 'champ_rate_diff': b_cr - r_cr,
            'b_team_rate':  b_tw, 'r_team_rate':  r_tw, 'team_rate_diff':  b_tw - r_tw,
            'h2h_rate': h2h_rate,
            'b_form': b_form, 'r_form': r_form, 'form_diff': b_form - r_form,
            'blue_picks': bp, 'red_picks': rp,
        }

        if include_new:
            # === Lane matchup features ===
            lane_advs = []
            for i in range(5):
                bc = bp[i]; rc = rp[i]
                # Specific matchup: blue's champ_i vs red's champ_i in role i
                # We track (myChamp, oppChamp). For blue, this is (bc, rc).
                key = (bc, rc)
                n  = lane_games[i][key]
                w  = lane_wins[i][key]
                # Prior: blue champ's general win rate in this role
                cp_n = cp_games[i][bc]; cp_w = cp_wins[i][bc]
                prior = cp_w / cp_n if cp_n > 0 else 0.5
                # Bayesian-shrunk specific matchup rate
                shrunk = (w + LANE_SHRINK * prior) / (n + LANE_SHRINK) if (n + LANE_SHRINK) > 0 else 0.5
                # Advantage: shrunken win rate - 0.5
                lane_advs.append(shrunk - 0.5)
            total_lane_adv = sum(lane_advs)
            feat['lane_adv_total'] = total_lane_adv
            for i, role in enumerate(ROLES):
                feat[f'lane_adv_{role}'] = lane_advs[i]

            # === Snowball features ===
            def snow_score(c):
                # Win-rate-when-ft5 minus win-rate-when-not-ft5
                gf = snow_games_ft5[c]; wf = snow_wins_ft5[c]
                gn = snow_games_nft5[c]; wn = snow_wins_nft5[c]
                if gf + gn < SNOWBALL_MIN_GAMES:
                    return 0.0   # not enough data; treat as neutral
                rf = wf/gf if gf > 0 else 0.5
                rn = wn/gn if gn > 0 else 0.5
                return rf - rn
            b_snow = sum(snow_score(c) for c in bp) / 5
            r_snow = sum(snow_score(c) for c in rp) / 5
            feat['b_snowball'] = b_snow
            feat['r_snowball'] = r_snow
            feat['snowball_diff'] = b_snow - r_snow

        rows.append(feat)

        # === Update state AFTER feature row built ===
        # Standard
        for c in bp:
            champ_games[c] += 1; champ_wins[c] += target
        for c in rp:
            champ_games[c] += 1; champ_wins[c] += (1 - target)
        team_games[blue] += 1; team_games[red] += 1
        team_wins[blue] += target; team_wins[red] += (1 - target)
        h2h[mk][blue] += target; h2h[mk][red] += (1 - target)
        team_recent[blue].append(target)
        team_recent[red].append(1 - target)

        # Lane matchup
        if include_new:
            for i in range(5):
                bc = bp[i]; rc = rp[i]
                # Blue perspective
                lane_games[i][(bc, rc)] += 1
                lane_wins[i][(bc, rc)]  += target
                # Red perspective (mirror)
                lane_games[i][(rc, bc)] += 1
                lane_wins[i][(rc, bc)]  += (1 - target)
                # Champ-position priors
                cp_games[i][bc] += 1; cp_wins[i][bc] += target
                cp_games[i][rc] += 1; cp_wins[i][rc] += (1 - target)

            # Snowball — needs FT5 label. If we don't have it, skip update.
            ft5 = r.get('ft5_binary')
            if pd.notna(ft5):
                n_ft5_known += 1
                ft5 = int(ft5)
                # ft5 == 1 means blue got FT5. For each blue champ, was this a
                # "team got FT5" game? Yes if ft5 == 1. Did the champ's team win? target.
                for c in bp:
                    if ft5 == 1:
                        snow_games_ft5[c] += 1; snow_wins_ft5[c] += target
                    else:
                        snow_games_nft5[c] += 1; snow_wins_nft5[c] += target
                for c in rp:
                    if ft5 == 0:
                        snow_games_ft5[c] += 1; snow_wins_ft5[c] += (1 - target)
                    else:
                        snow_games_nft5[c] += 1; snow_wins_nft5[c] += (1 - target)

    feat_df = pd.DataFrame(rows)
    y = sub[target_col].astype(int)
    game_ids = sub['game_id']
    years = sub['year']
    return feat_df, y, years, game_ids, n_ft5_known


# ====================================================================
# TRAIN + EVALUATE ONE CONFIG
# ====================================================================
def train_and_eval(feat_df, y, years, game_ids, target_label):
    """Train win-model-style classifier, evaluate on 2026 holdout.
    Uses champion presence (MLB) + per-game features.
    """
    train_mask = (years <= 2025).values
    test_mask  = (years == 2026).values
    # Use only train games' champion vocabulary
    mlb = MultiLabelBinarizer()
    mlb.fit((feat_df[train_mask]['blue_picks'] + feat_df[train_mask]['red_picks']).tolist())
    blue_enc = pd.DataFrame(mlb.transform(feat_df['blue_picks']),
                            columns=['b_'+c for c in mlb.classes_]).reset_index(drop=True)
    red_enc  = pd.DataFrame(mlb.transform(feat_df['red_picks']),
                            columns=['r_'+c for c in mlb.classes_]).reset_index(drop=True)

    # Numeric feature columns (everything except game_id, year, date, blue_picks, red_picks)
    drop_cols = {'game_id','year','date','blue_picks','red_picks'}
    num_cols = [c for c in feat_df.columns if c not in drop_cols]
    X = pd.concat([blue_enc, red_enc, feat_df[num_cols].reset_index(drop=True)], axis=1)

    X_train, y_train = X[train_mask], y[train_mask].reset_index(drop=True)
    X_test,  y_test  = X[test_mask],  y[test_mask].reset_index(drop=True)
    gid_test = game_ids[test_mask].reset_index(drop=True)

    base = GradientBoostingClassifier(n_estimators=125, max_depth=2,
                                       learning_rate=0.1, random_state=42)
    model = CalibratedClassifierCV(base, method='isotonic', cv=5)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:,1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    baseline = max(y_test.mean(), 1 - y_test.mean())
    return {
        'label': target_label,
        'n_train': train_mask.sum(),
        'n_test': test_mask.sum(),
        'acc': acc, 'auc': auc, 'brier': brier,
        'baseline': baseline, 'edge': acc - baseline,
        'gid_test': gid_test, 'probs': probs, 'preds': preds, 'y_test': y_test,
    }


# ====================================================================
# MAIN
# ====================================================================
def main():
    if not Path('proplay_matches.csv').exists():
        print("ERROR: proplay_matches.csv not found"); return
    if not Path('kill_timelines_v2.csv').exists():
        print("ERROR: kill_timelines_v2.csv not found — run merge_kills.py first"); return

    df = load_data()

    print("\n" + "="*70)
    print("  BUILDING FEATURES")
    print("="*70)
    print("\n[1/4] Win model — CURRENT features")
    win_cur_feat, win_y, win_years, win_gids, _ = build_features(df, 'blue_win', include_new=False)
    print(f"  Built {len(win_cur_feat)} rows")
    print("[2/4] Win model — NEW features (+ lane + snowball)")
    win_new_feat, _, _, _, n_ft5 = build_features(df, 'blue_win', include_new=True)
    print(f"  Built {len(win_new_feat)} rows  (FT5 known for {n_ft5} snowball updates)")
    print("[3/4] FT5 model — CURRENT features")
    ft5_cur_feat, ft5_y, ft5_years, ft5_gids, _ = build_features(df, 'ft5_binary', include_new=False)
    print(f"  Built {len(ft5_cur_feat)} rows")
    print("[4/4] FT5 model — NEW features (+ lane + snowball)")
    ft5_new_feat, _, _, _, n_ft5_2 = build_features(df, 'ft5_binary', include_new=True)
    print(f"  Built {len(ft5_new_feat)} rows  (FT5 known for {n_ft5_2} snowball updates)")

    print("\n" + "="*70)
    print("  TRAINING + EVALUATING")
    print("="*70)
    print("\nTraining WIN model (CURRENT)...")
    win_cur = train_and_eval(win_cur_feat, win_y, win_years, win_gids, 'WIN/CURRENT')
    print("Training WIN model (NEW)...")
    win_new = train_and_eval(win_new_feat, win_y, win_years, win_gids, 'WIN/NEW')
    print("Training FT5 model (CURRENT)...")
    ft5_cur = train_and_eval(ft5_cur_feat, ft5_y, ft5_years, ft5_gids, 'FT5/CURRENT')
    print("Training FT5 model (NEW)...")
    ft5_new = train_and_eval(ft5_new_feat, ft5_y, ft5_years, ft5_gids, 'FT5/NEW')

    print("\n" + "="*70)
    print("  RESULTS  (test = 2026 games)")
    print("="*70)

    def line(r):
        return (f"  {r['label']:<14} n={r['n_test']:>4}  "
                f"acc={r['acc']*100:.2f}%  AUC={r['auc']:.4f}  "
                f"edge={r['edge']*100:+.2f}%  brier={r['brier']:.4f}")

    print("\nWIN MODEL:")
    print(line(win_cur)); print(line(win_new))
    d_acc = (win_new['acc'] - win_cur['acc']) * 100
    d_auc = win_new['auc'] - win_cur['auc']
    print(f"  Δacc: {d_acc:+.2f}%   Δauc: {d_auc:+.4f}")

    print("\nFT5 MODEL:")
    print(line(ft5_cur)); print(line(ft5_new))
    d_acc_ft5 = (ft5_new['acc'] - ft5_cur['acc']) * 100
    d_auc_ft5 = ft5_new['auc'] - ft5_cur['auc']
    print(f"  Δacc: {d_acc_ft5:+.2f}%   Δauc: {d_auc_ft5:+.4f}")

    print("\n" + "="*70)
    print("  INTERPRETATION")
    print("="*70)
    for name, d_a, d_u in [('Win', d_acc, d_auc), ('FT5', d_acc_ft5, d_auc_ft5)]:
        verdict = ('STRONG win — deploy' if d_a >= 1.0 and d_u >= 0.01 else
                   'modest win — deploy at discretion' if d_a >= 0.3 and d_u >= 0.005 else
                   'essentially tied' if abs(d_a) < 0.3 else
                   'NEGATIVE — do not deploy')
        print(f"  {name} model: {verdict}  ({d_a:+.2f}% / {d_u:+.4f} AUC)")


if __name__ == '__main__':
    main()
