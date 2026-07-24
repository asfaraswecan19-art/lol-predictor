import json, glob

fp = glob.glob('kill_data/*.json')[0]
d = json.load(open(fp))
print("=== all top-level keys in a JSON file ===")
for k in d:
    v = d[k]
    if k == 'kills':
        print(f"  {k}: list of {len(v)}, first={v[0] if v else None}")
    elif k == 'summary':
        print(f"  {k}: {v}")
    else:
        print(f"  {k}: {v!r}")

print("\n=== unique team-name styles in JSON (first 200 files) ===")
teams = set()
for fp in glob.glob('kill_data/*.json')[:200]:
    try:
        d = json.load(open(fp))
        teams.add(d.get('blue_team')); teams.add(d.get('red_team'))
    except: pass
print(sorted(str(t) for t in teams if t))