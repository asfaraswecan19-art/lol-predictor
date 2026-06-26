"""Quick audit: what date range does kill_data/ actually cover?"""
import json
from pathlib import Path
from collections import Counter

kd = Path("kill_data")
years = Counter()
months = Counter()
total = 0
for fp in kd.glob("*.json"):
    try:
        data = json.loads(fp.read_text(encoding='utf-8'))
        ts = data.get('start_time', '')
        if ts:
            years[ts[:4]] += 1
            months[ts[:7]] += 1
            total += 1
    except Exception:
        pass

print(f"Total kill files: {total}")
print("\nBy year:")
for y, n in sorted(years.items()):
    print(f"  {y}: {n}")

print("\nEarliest months:")
for m, n in sorted(months.items())[:6]:
    print(f"  {m}: {n}")
print("\nLatest months:")
for m, n in sorted(months.items())[-6:]:
    print(f"  {m}: {n}")
