import pandas as pd

# Load current matches
current_matches_df = pd.read_csv('football_matches2.csv')

# Load historical matches from multiple CSV files
historical_dfs = []
for i in range(0, 5):  # Assuming you have 5 historical CSV files
    historical_dfs.append(pd.read_csv(f'historical_matches_{i}.csv'))

# Combine all historical DataFrames into one
historical_df = pd.concat(historical_dfs, ignore_index=True)

print("Available Matches:")
for index, row in current_matches_df.iterrows():
    print(f"{index}: {row['homeTeam']} vs {row['awayTeam']} on {row['date']}")

match_index = int(input("Select a match by entering the index number: "))
selected_match = current_matches_df.iloc[match_index]
home_team = selected_match['homeTeam']
away_team = selected_match['awayTeam']

# Filter historical data for the selected teams
historical_matches = historical_df[
    ((historical_df['HomeTeam'] == home_team) & (historical_df['AwayTeam'] == away_team)) |
    ((historical_df['HomeTeam'] == away_team) & (historical_df['AwayTeam'] == home_team))
]

# Calculate outcomes
wins = historical_matches[historical_matches['FTHG'] > historical_matches['FTAG']].shape[0]
draws = historical_matches[historical_matches['FTHG'] == historical_matches['FTAG']].shape[0]
losses = historical_matches[historical_matches['FTHG'] < historical_matches['FTAG']].shape[0]

# Display the results
print(f"Results for {home_team} against {away_team}:")
print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")