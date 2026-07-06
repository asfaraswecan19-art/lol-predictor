# =================================================================
# LIVE-STATE WIN MODEL  (build + train + backtest, one file)
# =================================================================
# Predicts blue win probability from IN-GAME state at a checkpoint.
# Unified model: minute is a feature, so one model serves 10/15/20/25.
#
# Leak-free by design. Oracle's Elixir has NO timestamped objective
# data (towers/dragons/barons are END-OF-GAME totals) so we use ONLY
# checkpoint columns that are true snapshots at that minute:
#     golddiff, xpdiff, csdiff, killdiff  (+ minute)
# All are things you can read off the scoreboard live.
#
# Split mirrors V8:  grid-search on 2023-24, validate on 2025,
# final backtest on 2026.  Isotonic calibration.  T1 leagues.
# Saves model_payload_live.pkl
# =================================================================
import os
import pickle
import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

FILES = [
    '2023_LoL_esports_match_data_from_OraclesElixir.csv',
    '2024_LoL_esports_match_data_from_OraclesElixir.csv',
    '2025_LoL_esports_match_data_from_OraclesElixir.csv',
    '2026_LoL_esports_match_data_from_OraclesElixir.csv',
]
TARGET_LEAGUES_T1 = [
    'LCK', 'LPL', 'LEC', 'LCS', 'CBLOL',
    'MSI', 'WLDs', 'LTA N', 'LTA S', 'LTA', 'FST',
]
CHECKPOINTS = [10, 15, 20, 25]

# Same neighbourhood as the V8 grid (est=200, depth=2, lr=0.05 was best there)
PARAM_GRID = {
    'n_estimators':  [150, 200, 300],
    'max_depth':     [2, 3],
    'learning_rate': [0.03, 0.05, 0.1],
}

FEATURES = ['golddiff', 'xpdiff', 'csdiff', 'killdiff', 'minute']


def safe(v, default=0.0):
    try:
        return float(v) if pd.notna(v) else default
    except Exception:
        return default


# =================================================================
# STEP 1 — BUILD CHECKPOINT SNAPSHOT ROWS
# =================================================================
# One row per (game, checkpoint), from BLUE's perspective.
# label = blue_win.  Only 'complete' checkpoint data is used.
def build_snapshots():
    rows = []
    for file in FILES:
        if not os.path.exists(file):
            print(f"  Skipping {file} — not found")
            continue
        print(f"  Reading {file}...")
        raw = pd.read_csv(file, low_memory=False)
        raw['year'] = pd.to_datetime(raw['date'], errors='coerce').dt.year
        raw = raw[raw['league'].isin(TARGET_LEAGUES_T1)].copy()
        if raw.empty:
            continue
        team = raw[raw['position'] == 'team'].copy()

        for game_id, g in team.groupby('gameid'):
            if len(g) != 2:
                continue
            blue_rows = g[g['side'] == 'Blue']
            red_rows = g[g['side'] == 'Red']
            if len(blue_rows) != 1 or len(red_rows) != 1:
                continue
            blue = blue_rows.iloc[0]
            red = red_rows.iloc[0]

            # need complete checkpoint data (partial rows lack the at-N cols)
            if str(blue.get('datacompleteness', '')) != 'complete':
                continue

            try:
                blue_win = int(blue['result'])
            except Exception:
                continue

            year = int(blue['year']) if pd.notna(blue['year']) else 0
            league = blue['league']

            for t in CHECKPOINTS:
                gd = blue.get(f'golddiffat{t}', np.nan)
                xd = blue.get(f'xpdiffat{t}', np.nan)
                cd = blue.get(f'csdiffat{t}', np.nan)
                bk = blue.get(f'killsat{t}', np.nan)
                rk = blue.get(f'opp_killsat{t}', np.nan)
                # skip checkpoints past game end / missing (all NaN)
                if pd.isna(gd) or pd.isna(xd) or pd.isna(cd) or pd.isna(bk) or pd.isna(rk):
                    continue
                rows.append({
                    'game_id':  game_id,
                    'league':   league,
                    'year':     year,
                    'minute':   float(t),
                    'golddiff': safe(gd),
                    'xpdiff':   safe(xd),
                    'csdiff':   safe(cd),
                    'killdiff': safe(bk) - safe(rk),
                    'blue_win': blue_win,
                })
    df = pd.DataFrame(rows)
    return df


# =================================================================
# STEP 2 — GRID SEARCH (train 2023-24, validate 2025)
# =================================================================
def grid_search(df):
    train = df[df['year'].isin([2023, 2024])]
    val = df[df['year'] == 2025]
    print(f"\n  Grid-search train rows: {len(train)}  | validate rows: {len(val)}")
    if len(train) == 0 or len(val) == 0:
        print("  WARNING: missing 2023-24 or 2025 data — cannot grid search.")
        return None

    Xtr, ytr = train[FEATURES].values, train['blue_win'].values
    Xva, yva = val[FEATURES].values, val['blue_win'].values

    best = None
    for est, depth, lr in product(
        PARAM_GRID['n_estimators'], PARAM_GRID['max_depth'], PARAM_GRID['learning_rate']
    ):
        clf = GradientBoostingClassifier(
            n_estimators=est, max_depth=depth, learning_rate=lr, random_state=42
        )
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(yva, p)
        acc = ((p >= 0.5).astype(int) == yva).mean()
        if best is None or auc > best['auc']:
            best = {'est': est, 'depth': depth, 'lr': lr, 'auc': auc, 'acc': acc}
        print(f"    est={est:<3} depth={depth} lr={lr:<4} -> val AUC {auc:.4f}  acc {acc*100:.2f}%")

    print(f"\n  BEST: est={best['est']} depth={best['depth']} lr={best['lr']} "
          f"(val AUC {best['auc']:.4f}, acc {best['acc']*100:.2f}%)")
    return best


# =================================================================
# STEP 3 — BACKTEST ON 2026, PER CHECKPOINT
# =================================================================
def backtest(model, df):
    test = df[df['year'] == 2026]
    if len(test) == 0:
        print("\n  No 2026 rows to backtest.")
        return
    print(f"\n{'='*55}\n  BACKTEST 2026  (per checkpoint)\n{'='*55}")
    print(f"  {'min':<6}{'n':<7}{'acc':<10}{'auc':<10}{'base(fav)':<10}")
    for t in CHECKPOINTS + ['ALL']:
        sub = test if t == 'ALL' else test[test['minute'] == t]
        if len(sub) == 0:
            continue
        X, y = sub[FEATURES].values, sub['blue_win'].values
        p = model.predict_proba(X)[:, 1]
        acc = ((p >= 0.5).astype(int) == y).mean()
        try:
            auc = roc_auc_score(y, p)
        except ValueError:
            auc = float('nan')
        # baseline: always pick current gold leader (blue if golddiff>0)
        fav = (sub['golddiff'].values > 0).astype(int)
        base = (fav == y).mean()
        label = 'ALL' if t == 'ALL' else str(int(t))
        print(f"  {label:<6}{len(sub):<7}{acc*100:<10.2f}{auc:<10.4f}{base*100:<10.2f}")


# =================================================================
# STEP 3b — DISAGREEMENT / "DOUBLE-DOWN" READOUT
# =================================================================
# When the model overrides the gold leader (gives the TRAILING team
# >50%), how often is it right? And does precision rise with the
# model's confidence? If yes -> the high-conviction end is tradeable.
# If precision is flat ~55% regardless of confidence -> it's noise.
def disagreement_readout(model, df):
    test = df[df['year'] == 2026]
    if len(test) == 0:
        print("\n  No 2026 rows for disagreement readout.")
        return
    print(f"\n{'='*55}\n  DISAGREEMENT READOUT 2026  (model overrides gold leader)\n{'='*55}")

    # Build per-row trailing-team probability + outcome, gold leader excluded
    test = test.copy()
    p_blue = model.predict_proba(test[FEATURES].values)[:, 1]
    test['p_blue'] = p_blue
    test = test[test['golddiff'] != 0]  # need a defined leader
    trailing_is_blue = test['golddiff'].values < 0
    p_trailing = np.where(trailing_is_blue, test['p_blue'].values,
                          1 - test['p_blue'].values)
    trailing_won = np.where(trailing_is_blue, test['blue_win'].values,
                            1 - test['blue_win'].values)

    # Overall: cases where model flags the trailing team (p_trailing > 0.5)
    flag = p_trailing > 0.5
    n_flag = flag.sum()
    if n_flag == 0:
        print("  Model never overrode the gold leader on 2026 data.")
        return
    prec = trailing_won[flag].mean()
    n_cb = int(trailing_won.sum())  # total actual comebacks in sample
    recall = trailing_won[flag].sum() / n_cb if n_cb else float('nan')
    print(f"  Times model flagged trailing team: {n_flag}")
    print(f"  Overall precision (flagged & won):  {prec*100:.1f}%")
    print(f"  Recall (of all comebacks caught):   {recall*100:.1f}%  ({int(trailing_won[flag].sum())}/{n_cb})")

    # Bucketed by model confidence in the trailing team.
    # THE KEY TABLE: does precision climb with confidence?
    print(f"\n  Precision bucketed by model confidence in trailing team:")
    print(f"  {'conf band':<14}{'n':<7}{'won':<7}{'precision':<12}{'avg_p':<8}")
    bands = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 1.01)]
    for lo, hi in bands:
        mask = (p_trailing >= lo) & (p_trailing < hi)
        n = mask.sum()
        if n == 0:
            print(f"  {f'{lo:.2f}-{hi:.2f}':<14}{0:<7}{'-':<7}{'-':<12}{'-':<8}")
            continue
        won = trailing_won[mask].sum()
        print(f"  {f'{lo:.2f}-{hi:.2f}':<14}{n:<7}{int(won):<7}"
              f"{won/n*100:<12.1f}{p_trailing[mask].mean():<8.3f}")
    print("\n  Read: if precision rises across bands and roughly tracks avg_p,")
    print("  the high-conviction flags are tradeable. If flat ~55%, it's noise.")

    # Same, split by checkpoint minute (are late overrides more reliable?)
    print(f"\n  Precision by checkpoint (flagged trailing-team calls):")
    print(f"  {'min':<6}{'flagged':<10}{'won':<7}{'precision':<10}")
    for t in CHECKPOINTS:
        sub = (test['minute'].values == t) & flag
        n = sub.sum()
        if n == 0:
            print(f"  {int(t):<6}{0:<10}{'-':<7}{'-':<10}")
            continue
        won = trailing_won[sub].sum()
        print(f"  {int(t):<6}{n:<10}{int(won):<7}{won/n*100:<10.1f}")


# =================================================================
# MAIN
# =================================================================
if __name__ == '__main__':
    print("Building live-state snapshot dataset...")
    df = build_snapshots()
    print(f"\n  Total snapshot rows: {len(df)}")
    if len(df):
        print("  Year breakdown:")
        print(df['year'].value_counts().sort_index().to_string())
        print("  Rows per checkpoint:")
        print(df['minute'].value_counts().sort_index().to_string())

    best = grid_search(df)

    # Decide final params: grid result if available, else the V8 default
    if best is not None:
        params = {'n_estimators': best['est'], 'max_depth': best['depth'],
                  'learning_rate': best['lr']}
    else:
        print("\n  Falling back to V8 default params (est=200, depth=2, lr=0.05).")
        params = {'n_estimators': 200, 'max_depth': 2, 'learning_rate': 0.05}

    # Final fit: train on everything up to 2025 (2023-25), calibrate, backtest 2026
    fit_df = df[df['year'].isin([2023, 2024, 2025])]
    if len(fit_df) == 0:
        print("\n  WARNING: no 2023-25 data present. Fitting on all non-2026 rows.")
        fit_df = df[df['year'] != 2026]
    if len(fit_df) == 0:
        print("  ERROR: nothing to train on. Need 2023-2025 CSVs locally. Exiting.")
        raise SystemExit(1)

    X, y = fit_df[FEATURES].values, fit_df['blue_win'].values
    base_clf = GradientBoostingClassifier(random_state=42, **params)
    model = CalibratedClassifierCV(base_clf, method='isotonic', cv=3)
    model.fit(X, y)
    print(f"\n  Final model trained on {len(fit_df)} rows "
          f"(params: {params}, isotonic calibration).")

    backtest(model, df)
    disagreement_readout(model, df)

    # Gold-lead -> winrate reference table (built from fit data, for the app banner)
    gold_table = {}
    ref = fit_df[fit_df['minute'] == 15]
    if len(ref):
        bins = [(-9e9, -3000, 'down 3k+'), (-3000, -1000, 'down 1-3k'),
                (-1000, 1000, 'even'), (1000, 3000, 'up 1-3k'),
                (3000, 9e9, 'up 3k+')]
        for lo, hi, name in bins:
            m = ref[(ref['golddiff'] > lo) & (ref['golddiff'] <= hi)]
            if len(m):
                gold_table[name] = round(m['blue_win'].mean(), 3)

    payload = {
        'live_model':   model,
        'features':     FEATURES,
        'checkpoints':  CHECKPOINTS,
        'params':       params,
        'gold_wr_at15': gold_table,
        'leagues':      TARGET_LEAGUES_T1,
    }
    with open('model_payload_live.pkl', 'wb') as f:
        pickle.dump(payload, f)
    print(f"\n✅ Saved model_payload_live.pkl")
    print(f"   gold_wr_at15 reference: {gold_table}")
