"""audit_v2.py — inspect kill_timelines_v2.csv to verify precise data landed."""
import pandas as pd

v2 = pd.read_csv("kill_timelines_v2.csv")
v2["blue_time"] = pd.to_numeric(v2["blue_time"], errors="coerce")
v2["red_time"]  = pd.to_numeric(v2["red_time"],  errors="coerce")

# A "precise" value is a float that isn't a clean multiple of 10.
# A "quantized" value is exactly 10, 20, or 30 (proxy fallback).
def is_float(x):
    return pd.notna(x) and x != int(x)
def is_quantized(x):
    return pd.notna(x) and x == int(x) and int(x) % 10 == 0

v2["bt_float"]  = v2["blue_time"].apply(is_float)
v2["bt_quant"]  = v2["blue_time"].apply(is_quantized)
v2["bt_blank"]  = v2["blue_time"].isna()

print("=" * 60)
print("kill_timelines_v2.csv AUDIT")
print("=" * 60)
print(f"Total rows:                {len(v2)}")
print(f"  Precise (float):         {v2['bt_float'].sum()}")
print(f"  Quantized (proxy):       {v2['bt_quant'].sum()}")
print(f"  Blank:                   {v2['bt_blank'].sum()}")

print("\n=== Sample PRECISE rows ===")
print(v2[v2["bt_float"]][["game_id","year","tournament","blue_team","red_team",
                          "first_to_five","blue_time","red_time"]].head(8).to_string())

print("\n=== Sample QUANTIZED rows ===")
print(v2[v2["bt_quant"]][["game_id","year","tournament","blue_team","red_team",
                          "first_to_five","blue_time","red_time"]].head(8).to_string())

print("\n=== Precise rate by league ===")
for lg, g in v2.groupby("tournament"):
    n_float = g["bt_float"].sum()
    n_quant = g["bt_quant"].sum()
    n_blank = g["bt_blank"].sum()
    total = len(g)
    if total < 20: continue
    pct = n_float / total * 100
    print(f"  {lg:<25} total={total:>5}  precise={n_float:>5} ({pct:>5.1f}%)  quant={n_quant:>5}  blank={n_blank:>4}")

print("\n=== Precise rate by year ===")
for yr, g in v2.groupby("year"):
    n_float = g["bt_float"].sum()
    total = len(g)
    pct = n_float / total * 100
    print(f"  {yr}: total={total}, precise={n_float} ({pct:.1f}%)")
