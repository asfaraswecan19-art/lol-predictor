"""
Quick inspection script — run this and paste the output back.
Goal: confirm which columns we have available to build the
champion-duration feature (duration profile + duration-win correlation).
"""

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = pd.read_csv("proplay_matches.csv")

print("=" * 60)
print("proplay_matches.csv")
print("=" * 60)
print(f"Shape: {df.shape}")
print("\nColumns:")
for c in df.columns:
    print(f"  {c}  ({df[c].dtype})")

print("\nFirst 3 rows:")
print(df.head(3))

# Try to spot likely relevant columns automatically
print("\n--- Likely relevant columns (heuristic match) ---")
keywords = ["champ", "length", "duration", "gamelength", "win", "result",
            "team", "side", "date", "league", "split", "year"]
for c in df.columns:
    if any(k in c.lower() for k in keywords):
        print(f"  {c}: sample values -> {df[c].dropna().unique()[:5]}")

# Also check kill_timelines.csv briefly, in case gamelength/champ live there instead
try:
    df2 = pd.read_csv("kill_timelines.csv")
    print("\n" + "=" * 60)
    print("kill_timelines.csv")
    print("=" * 60)
    print(f"Shape: {df2.shape}")
    print("\nColumns:")
    for c in df2.columns:
        print(f"  {c}  ({df2[c].dtype})")
    print("\nFirst 3 rows:")
    print(df2.head(3))
except FileNotFoundError:
    print("\n(kill_timelines.csv not found in this directory — skipping)")
