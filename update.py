import os
import subprocess
import sys
from datetime import datetime
import hashlib

PYTHON = r"C:\Users\Alex\AppData\Local\Python\pythoncore-3.14-64\python.exe"
FOLDER = os.path.dirname(os.path.abspath(__file__))
os.chdir(FOLDER)

FILE_2026 = "2026_LoL_esports_match_data_from_OraclesElixir.csv"
HASH_FILE = ".last_2026_hash"

def get_file_hash(filepath):
    if not os.path.exists(filepath):
        return None
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
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

print(f"\n🚀 LOL PREDICTOR UPDATE")
print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# =================================================================
# CHECK IF 2026 DATA IS NEW
# =================================================================
print(f"\n{'='*55}")
print(f"  Checking 2026 data for changes...")
print(f"{'='*55}")

current_hash = get_file_hash(FILE_2026)
last_hash    = None

if os.path.exists(HASH_FILE):
    with open(HASH_FILE, 'r') as f:
        last_hash = f.read().strip()

if current_hash is None:
    print(f"⚠️  {FILE_2026} not found — skipping update")
    input("\nPress Enter to close...")
    sys.exit(1)
elif current_hash == last_hash:
    print(f"ℹ️  2026 data is unchanged since last update — no new games")
    print(f"   Continuing anyway to keep model in sync...")
    data_changed = False
else:
    if last_hash is None:
        print(f"✅ First time running — building fresh model")
    else:
        print(f"✅ New data detected in 2026 CSV — model will be updated!")
    data_changed = True

# Show latest tier 1 match in the CSV
try:
    import pandas as pd
    df = pd.read_csv(FILE_2026)
    tier1 = ['LCK', 'LPL', 'LEC', 'LCS', 'MSI', 'WLDs', 'FST']
    if 'league' in df.columns and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        t1 = df[df['league'].isin(tier1)].dropna(subset=['date'])
        if len(t1) > 0:
            latest = t1.sort_values('date').iloc[-1]
            print(f"\n  📅 Latest tier 1 match in CSV:")
            print(f"     {latest.get('league','?')} | {str(latest['date'])[:10]} | "
                  f"{latest.get('teamname','?')} (patch {latest.get('patch','?')})")
        else:
            print(f"\n  📅 No tier 1 matches found in CSV")
except Exception as e:
    print(f"\n  ⚠️  Could not read latest match: {e}")

# Save new hash
with open(HASH_FILE, 'w') as f:
    f.write(current_hash)

# =================================================================
# REBUILD CSVs
# =================================================================
run(f'"{PYTHON}" build_dataset.py', "Rebuilding datasets")

# =================================================================
# RETRAIN MODELS
# =================================================================
run(f'"{PYTHON}" train_and_save.py', "Retraining models")

# =================================================================
# PUSH TO GITHUB
# =================================================================
print(f"\n{'='*55}")
print(f"  Pushing to GitHub...")
print(f"{'='*55}")

os.system("git add app.py")
os.system("git add model_payload.pkl")
commit_msg    = f"auto-update {datetime.now().strftime('%Y-%m-%d %H:%M')}"
commit_result = os.system(f'git commit -m "{commit_msg}"')
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
if data_changed:
    print(f"  📊 New 2026 data was found and included!")
else:
    print(f"  📊 No new 2026 data — model rebuilt from same data")
print(f"  🌐 Website will update automatically shortly")
print(f"{'='*55}")

input("\nPress Enter to close...")

