import pandas as pd
pp = pd.read_csv('proplay_matches.csv')
# show a series: same teams, same day, multiple games
print(pp[['game_id','date','blue_team','red_team']].head(12).to_string())
print("\n=== does game_id encode series/game order? ===")
print("sample game_ids:", pp['game_id'].head(8).tolist())