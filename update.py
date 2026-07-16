import os
import subprocess
import sys
from datetime import datetime
import hashlib

PYTHON = r"C:\Users\Alex\AppData\Local\Python\pythoncore-3.14-64\python.exe"
FOLDER = os.path.dirname(os.path.abspath(__file__))
os.chdir(FOLDER)

FILE_2026    = "2026_LoL_esports_match_data_from_OraclesElixir.csv"
HASH_FILE_T1 = ".last_2026_hash"
HASH_FILE_T2 = ".last_t2_hash"

# Tier 2 leagues — check these files for changes
T2_FILES = [
    "2023_LoL_esports_match_data_from_OraclesElixir.csv",
    "2024_LoL_esports_match_data_from_OraclesElixir.csv",
    "2025_LoL_esports_match_data_from_OraclesElixir.csv",
    "2026_LoL_esports_match_data_from_OraclesElixir.csv",
]

def get_file_hash(filepath):
    if not os.path.exists(filepath):
        return None
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def get_combined_hash(files):
    """Hash multiple files combined"""
    h = hashlib.md5()
    for f in files:
        if os.path.exists(f):
            with open(f, 'rb') as fh:
                for chunk in iter(lambda: fh.read(8192), b''):
                    h.update(chunk)
    return h.hexdigest()

def run(cmd, description):
    print(f"\n{'='*55}")
    print(f"  {description}...")
    print(f"{'='*55}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        input("\nPress Enter to close...")
        sys.exit(1)
    print(f"✅ Done: {description}")

def run_soft(cmd, description):
    """Like run(), but does NOT abort the update on failure.

    Used for the backtesters. They only ATTACH reporting stats to the
    payloads (backtest_win_acc etc. for app.py's header) -- they don't
    build or change the models themselves. If a backtester fails, the
    trained model is still perfectly valid and should still ship; we'd
    just lose the fresh stats and app.py falls back to its last known
    numbers. Aborting the push over that would be worse than continuing.
    Returns True on success.
    """
    print(f"\n{'='*55}")
    print(f"  {description}...")
    print(f"{'='*55}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"⚠️  {description} FAILED — continuing anyway.")
        print(f"    The model is still valid and will be pushed.")
        print(f"    app.py will show its last known stats instead of fresh ones.")
        return False
    print(f"✅ Done: {description}")
    return True

print(f"\n🚀 LOL PREDICTOR UPDATE")
print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# =================================================================
# CHECK TIER 1 DATA
# =================================================================
print(f"\n{'='*55}")
print(f"  Checking tier 1 data (2026 CSV)...")
print(f"{'='*55}")

current_hash_t1 = get_file_hash(FILE_2026)
last_hash_t1    = None

if os.path.exists(HASH_FILE_T1):
    with open(HASH_FILE_T1, 'r') as f:
        last_hash_t1 = f.read().strip()

if current_hash_t1 is None:
    print(f"⚠️  {FILE_2026} not found — skipping update")
    input("\nPress Enter to close...")
    sys.exit(1)
elif current_hash_t1 == last_hash_t1:
    print(f"ℹ️  Tier 1 data unchanged since last update")
    t1_changed = False
else:
    print(f"✅ New tier 1 data detected!")
    t1_changed = True

# =================================================================
# CHECK TIER 2 DATA
# =================================================================
print(f"\n{'='*55}")
print(f"  Checking tier 2 data (all year CSVs)...")
print(f"{'='*55}")

current_hash_t2 = get_combined_hash(T2_FILES)
last_hash_t2    = None

if os.path.exists(HASH_FILE_T2):
    with open(HASH_FILE_T2, 'r') as f:
        last_hash_t2 = f.read().strip()

if current_hash_t2 == last_hash_t2:
    print(f"ℹ️  Tier 2 data unchanged since last update")
    t2_changed = False
else:
    print(f"✅ Tier 2 data changed — will rebuild tier 2 model")
    t2_changed = True

data_changed = t1_changed or t2_changed

if not data_changed:
    print(f"\nℹ️  No data changes detected — rebuilding anyway to keep model in sync...")

# Show latest tier 1 match
try:
    import pandas as pd
    df = pd.read_csv(FILE_2026)
    tier1 = ['LCK', 'LPL', 'LEC', 'LCS', 'MSI', 'WLDs', 'FST']
    if 'league' in df.columns and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        t1 = df[df['league'].isin(tier1)].dropna(subset=['date'])
        if len(t1) > 0:
            latest = t1.sort_values('date').iloc[-1]
            print(f"\n  📅 Latest tier 1 match:")
            print(f"     {latest.get('league','?')} | {str(latest['date'])[:10]} | "
                  f"{latest.get('teamname','?')} (patch {latest.get('patch','?')})")
except Exception as e:
    print(f"\n  ⚠️  Could not read latest match: {e}")

# Save new hashes
with open(HASH_FILE_T1, 'w') as f: f.write(current_hash_t1)
with open(HASH_FILE_T2, 'w') as f: f.write(current_hash_t2)

# =================================================================
# REBUILD CSVs
# =================================================================
run(f'"{PYTHON}" build_dataset.py', "Rebuilding datasets (T1 + T2)")

# =================================================================
# RETRAIN MODELS
# =================================================================
run(f'"{PYTHON}" train_and_save.py', "Retraining models (T1 + T2)")

# =================================================================
# BACKTEST — must run AFTER train_and_save.py and BEFORE the push.
# =================================================================
# These are the steps that write the real out-of-sample 2026 numbers into
# the payloads:
#   backtester.py    -> model_payload.pkl    (backtest_win_acc/auc,
#                                             backtest_ft5_acc/auc,
#                                             backtest_ft5_league_edges)
#   backtester_t2.py -> model_payload_t2.pkl (backtest_win_acc/auc,
#                                             backtest_win_league_edges)
# app.py reads those keys for its header/footer/league tips. Without this,
# the payloads ship with NO stats and app.py falls back to hardcoded
# "last hand-verified" numbers -- which is exactly the stale-number problem
# this pipeline was built to eliminate.
#
# Order is load-bearing: train writes the payload, backtest adds stats to it,
# THEN we push. Running these after the push would ship statless payloads.
#
# Non-fatal (run_soft): a backtest failure doesn't invalidate the model.
print(f"\n{'='*55}")
print(f"  BACKTESTING (this is the slow part — grid searches)")
print(f"{'='*55}")

bt_t1_ok = run_soft(f'"{PYTHON}" backtester.py', "Backtesting T1 + writing stats to payload")

if os.path.exists("backtester_t2.py"):
    bt_t2_ok = run_soft(f'"{PYTHON}" backtester_t2.py', "Backtesting T2 + writing stats to payload")
else:
    print(f"\n⚠️  backtester_t2.py not found — skipping T2 backtest.")
    print(f"    The T2 tab will show its last known stats.")
    bt_t2_ok = False

# =================================================================
# PUSH TO GITHUB
# =================================================================
print(f"\n{'='*55}")
print(f"  Pushing to GitHub...")
print(f"{'='*55}")

os.system("git add app.py")
os.system("git add model_payload.pkl")
os.system("git add model_payload_t2.pkl")
commit_msg    = f"auto-update {datetime.now().strftime('%Y-%m-%d %H:%M')}"
commit_result = os.system(f'git commit -m "{commit_msg}"')
os.system('attrib -r .git\\FETCH_HEAD 2>nul')
os.system("git add .")
os.system('git commit -m "pre-pull sync" 2>nul')
os.system("git pull origin main --no-rebase")
push_result   = os.system("git push origin main")
if push_result == 0:
    print("✅ Pushed to GitHub — Streamlit will redeploy in ~1 minute!")
else:
    print("❌ Push failed — check your git credentials")

# =================================================================
# SUMMARY
# =================================================================
print(f"\n{'='*55}")
print(f"  UPDATE COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*55}")
print(f"  📊 T1 data: {'NEW' if t1_changed else 'unchanged'}")
print(f"  📊 T2 data: {'NEW' if t2_changed else 'unchanged'}")
print(f"  🎯 T1 backtest: {'stats refreshed' if bt_t1_ok else '⚠️  FAILED (app shows last known stats)'}")
print(f"  🎯 T2 backtest: {'stats refreshed' if bt_t2_ok else '⚠️  FAILED (app shows last known stats)'}")
if not (bt_t1_ok and bt_t2_ok):
    print(f"  ⚠️  A backtest failed — the model still shipped, but the")
    print(f"      accuracy shown in the app may be out of date.")
print(f"  🌐 Website will update automatically shortly")
print(f"{'='*55}")

input("\nPress Enter to close...")
