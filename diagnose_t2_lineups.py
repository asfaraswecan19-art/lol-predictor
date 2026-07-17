"""
T2 LINEUP AUTOFILL DIAGNOSTIC
=============================
Traces the whole chain to find exactly where autofill breaks:

  1. Is train_and_save.py the fixed version?
  2. Does proplay_matches_t2.csv exist and have player data?
  3. Did the T2 build block actually run?
  4. Does model_payload_t2.pkl have team_lineups populated?
  5. Do the team names in team_lineups match all_teams?
  6. Would fuzzy_match_team() actually resolve to a key in team_lineups?

Run from the dataset builder folder. Paste the whole output back.
"""

import os
import pickle
import difflib

print("="*64)
print("  T2 LINEUP AUTOFILL DIAGNOSTIC")
print("="*64)

# ---------------------------------------------------------------
# 1. Is train_and_save.py the fixed version?
# ---------------------------------------------------------------
print("\n[1] train_and_save.py version check")
if not os.path.exists('train_and_save.py'):
    print("    ERROR: train_and_save.py not found in this folder!")
else:
    src = open('train_and_save.py', encoding='utf-8', errors='replace').read()
    has_build   = 'T2 team lineups built' in src
    has_assign  = "'team_lineups':     team_lineups_t2" in src
    has_old     = "'team_lineups':     {}," in src
    print(f"    builds team_lineups_t2:      {'YES' if has_build else 'NO  <-- OLD VERSION'}")
    print(f"    assigns it to t2_payload:    {'YES' if has_assign else 'NO  <-- OLD VERSION'}")
    print(f"    old hardcoded empty present: {'YES <-- BAD' if has_old else 'no (good)'}")

# ---------------------------------------------------------------
# 2. Does the T2 source CSV exist with player data?
# ---------------------------------------------------------------
print("\n[2] proplay_matches_t2.csv check")
if not os.path.exists('proplay_matches_t2.csv'):
    print("    ERROR: proplay_matches_t2.csv NOT FOUND")
    print("    -> The T2 build block in train_and_save.py is wrapped in a")
    print("       file-exists check. If this file is missing, the whole T2")
    print("       section SILENTLY SKIPS and your old payload survives")
    print("       untouched -- including its empty team_lineups.")
else:
    import pandas as pd
    t2df = pd.read_csv('proplay_matches_t2.csv')
    print(f"    rows: {len(t2df)}")
    print(f"    columns present: blue_players={'blue_players' in t2df.columns}, "
          f"red_players={'red_players' in t2df.columns}")
    if 'blue_players' in t2df.columns:
        n_null = t2df['blue_players'].isna().sum()
        print(f"    blue_players nulls: {n_null} / {len(t2df)}")
        sample = t2df['blue_players'].dropna().head(3).tolist()
        print(f"    sample blue_players values:")
        for s in sample:
            print(f"      {s!r}")
        # check they actually look like 5 names
        if sample:
            parts = str(sample[0]).split(',')
            print(f"    first row splits into {len(parts)} names")
    if 'blue_team' in t2df.columns:
        print(f"    sample teams: {t2df['blue_team'].dropna().unique()[:5].tolist()}")

# ---------------------------------------------------------------
# 3/4. Payload check
# ---------------------------------------------------------------
print("\n[3] model_payload_t2.pkl check")
if not os.path.exists('model_payload_t2.pkl'):
    print("    ERROR: model_payload_t2.pkl NOT FOUND")
else:
    mtime = os.path.getmtime('model_payload_t2.pkl')
    import datetime
    print(f"    last modified: {datetime.datetime.fromtimestamp(mtime)}")
    print(f"    (^ if this is OLD, train_and_save.py didn't rewrite it)")
    p2 = pickle.load(open('model_payload_t2.pkl','rb'))
    tl = p2.get('team_lineups')
    if tl is None:
        print("    team_lineups: KEY MISSING")
    else:
        print(f"    team_lineups: {len(tl)} teams")
        for t, lu in list(tl.items())[:3]:
            print(f"      {t!r}: {lu}")

    at = p2.get('all_teams', [])
    print(f"    all_teams: {len(at)} teams")
    print(f"      sample: {at[:5]}")

    # ---------------------------------------------------------------
    # 5. Do team_lineups keys match all_teams?
    # ---------------------------------------------------------------
    if tl:
        print("\n[4] Key matching: team_lineups vs all_teams")
        tl_keys = set(tl.keys())
        at_set  = set(at)
        missing = at_set - tl_keys
        extra   = tl_keys - at_set
        print(f"    teams in all_teams but NOT in team_lineups: {len(missing)}")
        if missing:
            print(f"      {sorted(missing)[:8]}")
        print(f"    teams in team_lineups but NOT in all_teams: {len(extra)}")
        if extra:
            print(f"      {sorted(extra)[:8]}")
        if not missing and not extra:
            print("    -> PERFECT MATCH (autofill should work for every team)")

        # ---------------------------------------------------------------
        # 6. Simulate app.py's fuzzy_match_team + autofill condition
        # ---------------------------------------------------------------
        print("\n[5] Simulating app.py autofill for each T2 team")
        TEAM_ALIASES = p2.get('team_aliases', {})
        def normalize_team(n): return TEAM_ALIASES.get(str(n), str(n))
        def fuzzy_match_team(raw, all_teams):
            if not raw or not raw.strip(): return None, False
            raw_strip = raw.strip()
            normed = normalize_team(raw_strip)
            if normed != raw_strip and normed in all_teams: return normed, True
            for t in all_teams:
                if raw_strip.lower() == t.lower(): return t, True
            raw_clean = raw_strip.lower().replace(" ","")
            team_map  = {t.lower().replace(" ",""): t for t in all_teams}
            matches   = difflib.get_close_matches(raw_clean, team_map.keys(), n=1, cutoff=0.7)
            return (team_map[matches[0]], False) if matches else (None, False)

        ok = 0; fail = 0
        for team in at[:12]:
            match, exact = fuzzy_match_team(team, at)
            would_fill = bool(match and match in tl)
            status = "FILLS" if would_fill else "NO FILL"
            if would_fill: ok += 1
            else: fail += 1
            print(f"      typing {team!r:<28} -> match={match!r} exact={exact} {status}")
        print(f"\n    {ok} would autofill, {fail} would not")

print("\n" + "="*64)
print("  WHAT TO LOOK FOR")
print("="*64)
print("  [1] NO on either check  -> train_and_save.py still old, replace it")
print("  [2] file missing        -> T2 block silently skips; old payload kept")
print("  [3] 0 teams / old mtime -> train_and_save.py didn't rebuild the payload")
print("  [4] key mismatch        -> lineups exist but under different names")
print("  [5] NO FILL             -> fuzzy match resolves to a name not in lineups")
print("="*64)
