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
}

def standardize_team_names(df):

    df['homeTeam'] = df['homeTeam'].replace(TEAM_NAME_MAPPING)
    df['awayTeam'] = df['awayTeam'].replace(TEAM_NAME_MAPPING)
    return df

def fetch_real_time_data(api_url, api_token, output_file):
 
    headers = {'X-Auth-Token': api_token}  
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if 'matches' in data:
            match_data = [
                {
                    'homeTeam': match['homeTeam']['name'],
                    'awayTeam': match['awayTeam']['name'],
                    'homeScore': match['score']['fullTime']['home'],
                    'awayScore': match['score']['fullTime']['away'],
                    'status': match['status'],
                    'date': match['utcDate']
                }
                for match in data['matches']
            ]

            current_matches_df = pd.DataFrame(match_data)
            current_matches_df.to_csv(output_file, index=False)
            print(f"Real-time data saved to {output_file}")
            return current_matches_df
        else:
            print("'matches' key not found in API response.")
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except Exception as e:
        print(f"An error occurred: {e}")

def preprocess_historical_data(file_paths, output_file="historical_matches_cleaned2.csv"):
    all_dfs = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue

        df = pd.read_csv(file_path)

        # Select only relevant columns and rename them
        df_cleaned = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].copy()

        # Convert date format
        # df_cleaned['date'] = pd.to_datetime(df_cleaned['Date']).dt.strftime('%Y-%m-%dT00:00:00Z')
        df_cleaned['date'] = pd.to_datetime(df_cleaned['Date'], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%dT00:00:00Z')


        # Standardize team names
        df_cleaned = standardize_team_names(df_cleaned)

        # Add a status column and set it to "FINISHED"
        df_cleaned['status'] = 'FINISHED'

        # Rename columns to match the desired output
        df_cleaned = df_cleaned.rename(columns={
            'HomeTeam': 'homeTeam',
            'AwayTeam': 'awayTeam',
            'FTHG': 'homeScore',
            'FTAG': 'awayScore',
            'date': 'date'
        })

        # Reorder columns to match the desired format
        df_cleaned = df_cleaned[['homeTeam', 'awayTeam', 'homeScore', 'awayScore', 'status', 'date']]

        # Append to list
        all_dfs.append(df_cleaned)

    # Combine all historical data into one DataFrame
    if all_dfs:
        historical_df = pd.concat(all_dfs, ignore_index=True)
        historical_df.to_csv(output_file, index=False)
        print(f"Historical data saved as {output_file}")
        return historical_df
    else:
        print("No historical data was processed.")
        return pd.DataFrame()

def load_historical_data(file_pattern, num_files):
    file_paths = [file_pattern.format(i) for i in range(num_files)]
    return preprocess_historical_data(file_paths)

def analyze_and_predict(current_df, historical_df):
    
    # Standardize team names
    historical_df = standardize_team_names(historical_df)
    current_df = standardize_team_names(current_df)

    # Check for unmatched teams
    unmatched_home = set(current_df['homeTeam']) - set(historical_df['homeTeam'])
    unmatched_away = set(current_df['awayTeam']) - set(historical_df['awayTeam'])

    print("Unmatched Home Teams:", unmatched_home)
    print("Unmatched Away Teams:", unmatched_away)

    print("Available Matches:")
    for index, row in current_df.iterrows():
        print(f"{index}: {row['homeTeam']} vs {row['awayTeam']} on {row['date']}")

    match_index = int(input("Select a match by entering the index number: "))
    selected_match = current_df.iloc[match_index]
    home_team = selected_match['homeTeam']
    away_team = selected_match['awayTeam']

    # Filter historical data for the selected teams
    historical_matches = historical_df[
        ((historical_df['homeTeam'] == home_team) & (historical_df['awayTeam'] == away_team)) |
        ((historical_df['homeTeam'] == away_team) & (historical_df['awayTeam'] == home_team))
    ]

    if historical_matches.empty:
        print(f"No historical data available for {home_team} vs {away_team}.")
        return

    historical_matches['Result'] = historical_matches.apply(
        lambda row: 1 if row['homeScore'] > row['awayScore'] else (-1 if row['homeScore'] < row['awayScore'] else 0), axis=1
    )

    X = historical_matches[['homeScore', 'awayScore']]
    y = historical_matches['Result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    avg_home_goals = historical_matches['homeScore'].mean()
    avg_away_goals = historical_matches['awayScore'].mean()
    prediction = model.predict([[avg_home_goals, avg_away_goals]])

    result_map = {1: "Win", 0: "Draw", -1: "Loss"}
    print(f"Predicted result for {home_team} vs {away_team}: {result_map[prediction[0]]}")

if __name__ == "__main__":
    API_URL = 'https://api.football-data.org/v4/competitions/PL/matches'
    API_TOKEN = '5412d5e47c5d4c34a313710ed20eecb7'
    CURRENT_MATCHES_FILE = 'football_matches2.csv'

    current_matches = fetch_real_time_data(API_URL, API_TOKEN, CURRENT_MATCHES_FILE)
    historical_data = load_historical_data('HISTORICAL/historical_matches_{}.csv', 5)

    if current_matches is not None and not historical_data.empty:
        analyze_and_predict(current_matches, historical_data)


