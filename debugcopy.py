import os
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define a mapping of historical team names to current team names
TEAM_NAME_MAPPING = {
    "Burnley": "Burnley FC",
    "Man City": "Manchester City FC",
    "Arsenal": "Arsenal FC",
    "Nott'm Forest": "Nottingham Forest FC",
    "Nottingham Forest": "Nottingham Forest FC",
    "Manchester United": "Manchester United FC",
    "Man United": "Manchester United FC",
    "Fulham": "Fulham FC",
    "Brighton": "Brighton & Hove Albion FC",
    "Spurs": "Tottenham Hotspur FC",
    "Tottenham": "Tottenham Hotspur FC",
    "West Ham": "West Ham United FC",
    "Newcastle": "Newcastle United FC",
    "Bournemouth": "AFC Bournemouth",
    "Crystal Palace": "Crystal Palace FC",
    "Leicester": "Leicester City FC",
    "Southampton": "Southampton FC",
    "Everton": "Everton FC",
    "Aston Villa": "Aston Villa FC",
    "Brentford": "Brentford FC",
    "Chelsea": "Chelsea FC",
    "Liverpool": "Liverpool FC",
    "Ipswich": "Ipswich Town FC",
    "Wolves": "Wolverhampton Wanderers FC"
}

def standardize_team_names(df):
    """Standardizes team names using predefined mappings."""
    df['homeTeam'] = df['homeTeam'].map(TEAM_NAME_MAPPING).fillna(df['homeTeam'])
    df['awayTeam'] = df['awayTeam'].map(TEAM_NAME_MAPPING).fillna(df['awayTeam'])
    return df

def preprocess_historical_data(file_paths):
    """Loads, cleans, and standardizes historical match data."""
    all_dfs = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue

        df = pd.read_csv(file_path)
        df_cleaned = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].copy()
        df_cleaned['date'] = pd.to_datetime(df_cleaned['Date'], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%dT00:00:00Z')
        
        df_cleaned = df_cleaned.rename(columns={
            'HomeTeam': 'homeTeam', 'AwayTeam': 'awayTeam',
            'FTHG': 'homeScore', 'FTAG': 'awayScore'
        })
        
        df_cleaned = standardize_team_names(df_cleaned)
        df_cleaned['status'] = 'FINISHED'
        df_cleaned['goalDifference'] = df_cleaned['homeScore'] - df_cleaned['awayScore']
        
        all_dfs.append(df_cleaned)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def load_historical_data(file_pattern, num_files):
    """Loads historical data from multiple files."""
    file_paths = [file_pattern.format(i) for i in range(num_files)]
    return preprocess_historical_data(file_paths)

def analyze_and_predict(current_df, historical_df):
    """Trains a logistic regression model and predicts outcomes for a user-selected match."""
    historical_df = standardize_team_names(historical_df)
    current_df = standardize_team_names(current_df)

    # Remove matches involving Ipswich Town FC
    current_df = current_df[~current_df['homeTeam'].isin(["Ipswich Town FC"]) & ~current_df['awayTeam'].isin(["Ipswich Town FC"])]

    # Ensure historical data contains teams from current matches
    historical_teams = set(historical_df['homeTeam']).union(set(historical_df['awayTeam']))
    current_teams = set(current_df['homeTeam']).union(set(current_df['awayTeam']))
    missing_teams = current_teams - historical_teams

    if missing_teams:
        print(f"Skipping matches due to missing teams in historical data: {missing_teams}")
        return

    # Display available matches
    print("\nAvailable Matches:")
    for idx, row in current_df.iterrows():
        print(f"{idx}: {row['homeTeam']} vs {row['awayTeam']} on {row['date']}")

    # Ask user to select a match
    while True:
        try:
            match_index = int(input("\nEnter the index of the match you want to predict: "))
            if match_index in current_df.index:
                break
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")

    # Get selected match
    selected_match = current_df.loc[match_index]
    home_team, away_team = selected_match['homeTeam'], selected_match['awayTeam']

    # Filter historical matches
    historical_matches = historical_df[
        ((historical_df['homeTeam'] == home_team) & (historical_df['awayTeam'] == away_team)) |
        ((historical_df['homeTeam'] == away_team) & (historical_df['awayTeam'] == home_team))
    ]

    if historical_matches.empty:
        print(f"No historical data available for {home_team} vs {away_team}.")
        return

    # Define match outcome (1 = Home Win, 0 = Draw/Away Win)
    historical_matches = historical_matches.copy()  # Fix SettingWithCopyWarning
    historical_matches.loc[:, 'Result'] = (historical_matches['homeScore'] > historical_matches['awayScore']).astype(int)

    # Features and target
    X = historical_matches[['homeScore', 'awayScore']]
    y = historical_matches['Result']

    # Train logistic regression model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"\nModel accuracy: {accuracy * 100:.2f}%")

    # Predict match result
    avg_home_goals = historical_matches['homeScore'].mean()
    avg_away_goals = historical_matches['awayScore'].mean()

    prediction_data = pd.DataFrame([[avg_home_goals, avg_away_goals]], columns=['homeScore', 'awayScore'])  # Fix feature names issue
    prediction = model.predict(prediction_data)

    result_map = {1: "Win", 0: "Draw/Away Win"}
    print(f"\nPredicted result for {home_team} vs {away_team}: {result_map[prediction[0]]}")

if __name__ == "__main__":
    API_URL = 'https://api.football-data.org/v4/competitions/PL/matches'
    API_TOKEN = '5412d5e47c5d4c34a313710ed20eecb7'
    CURRENT_MATCHES_FILE = 'football_matches2.csv'

    # Load historical data
    historical_data = load_historical_data('HISTORICAL/historical_matches_{}.csv', 5)

    # Load current matches
    if os.path.exists(CURRENT_MATCHES_FILE):
        current_matches = pd.read_csv(CURRENT_MATCHES_FILE)
        current_matches = standardize_team_names(current_matches)

        # Predict outcomes
        if not historical_data.empty and not current_matches.empty:
            analyze_and_predict(current_matches, historical_data)
