import pandas as pd
pp = pd.read_csv('proplay_matches.csv')
pp['date'] = pd.to_datetime(pp['date'], errors='coerce')

# Is the T1 vs KT game from 2026-05-28 actually in proplay?
may28 = pp[(pp['date'].dt.date == pd.Timestamp('2026-05-28').date())]
print(f"ALL proplay games on 2026-05-28: {len(may28)}")
print(may28[['game_id','league','blue_team','red_team']].to_string())

# specifically T1/KT that day
t1kt = may28[((may28['blue_team']=='T1')|(may28['red_team']=='T1'))]
print(f"\nT1 games on 2026-05-28 in proplay: {len(t1kt)}")