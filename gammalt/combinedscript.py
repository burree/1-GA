import os
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === Step 1: Fetch Real-Time Data ===
def fetch_real_time_data(api_url, api_token, output_file):
    """
    Fetches real-time match data from the football API and saves it to a CSV file.

    Args:
        api_url (str): The API endpoint URL.
        api_token (str): The API authentication token.
        output_file (str): The path to save the fetched data as a CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the fetched match data.
    """
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

# === Step 2: Preprocess Historical Data ===
def preprocess_historical_data(file_paths, output_file="historical_matches_cleaned.csv"):
    """
    Processes historical football match data to match the structure of the current matches CSV.

    Args:
        file_paths (list): List of file paths for historical match CSVs.
        output_file (str): Output file to save the cleaned data.
    """
    all_dfs = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue

        df = pd.read_csv(file_path)

        # Select only relevant columns and rename them
        df_cleaned = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].copy()
        df_cleaned.rename(columns={
            'Date': 'date',
            'HomeTeam': 'homeTeam',
            'AwayTeam': 'awayTeam',
            'FTHG': 'homeScore',
            'FTAG': 'awayScore'
        }, inplace=True)

        # Convert date format (assuming original format is DD/MM/YYYY)
        df_cleaned['date'] = pd.to_datetime(df_cleaned['date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%dT00:00:00Z')

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

# === Step 3: Load Historical Data ===
def load_historical_data(file_pattern, num_files):
    """
    Loads and combines multiple historical match CSV files into a single DataFrame.

    Args:
        file_pattern (str): The pattern of the filenames, e.g., "HISTORICAL/historical_matches_{}.csv".
        num_files (int): The number of historical CSV files to load.

    Returns:
        pd.DataFrame: A combined DataFrame containing all historical match data.
    """
    file_paths = [file_pattern.format(i) for i in range(num_files)]
    return preprocess_historical_data(file_paths)

# === Step 4: Analyze and Predict ===
def analyze_and_predict(current_df, historical_df):
    """
    Analyzes the matches from the current dataset and uses historical data to generate insights.

    Args:
        current_df (pd.DataFrame): The DataFrame containing the current match data.
        historical_df (pd.DataFrame): The DataFrame containing the historical match data.
    """
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

    # Prepare data for training
    if historical_matches.empty:
        print(f"No historical data available for {home_team} vs {away_team}.")
        return

    # Create labels (win=1, draw=0, loss=-1) based on the home team's perspective
    historical_matches['Result'] = historical_matches.apply(
        lambda row: 1 if row['homeScore'] > row['awayScore'] else (-1 if row['homeScore'] < row['awayScore'] else 0), axis=1
    )

    # Features: Home goals, Away goals
    X = historical_matches[['homeScore', 'awayScore']]
    y = historical_matches['Result']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Predict the outcome for the selected match
    avg_home_goals = historical_matches['homeScore'].mean()
    avg_away_goals = historical_matches['awayScore'].mean()
    prediction = model.predict([[avg_home_goals, avg_away_goals]])

    # Display the prediction
    result_map = {1: "Win", 0: "Draw", -1: "Loss"}
    print(f"Predicted result for {home_team} vs {away_team}: {result_map[prediction[0]]}")

# === Main Function ===
if __name__ == "__main__":
    # API Details
    API_URL = 'https://api.football-data.org/v4/competitions/PL/matches'
    API_TOKEN = '5412d5e47c5d4c34a313710ed20eecb7'
    CURRENT_MATCHES_FILE = 'football_matches2.csv'

    # Historical Data Details
    HISTORICAL_FILE_PATTERN = 'HISTORICAL/historical_matches_{}.csv'
    NUM_HISTORICAL_FILES = 5

    # Fetch real-time data
    current_matches = fetch_real_time_data(API_URL, API_TOKEN, CURRENT_MATCHES_FILE)

    # Load historical data
    historical_data = load_historical_data(HISTORICAL_FILE_PATTERN, NUM_HISTORICAL_FILES)

    # Analyze and predict
    if current_matches is not None and not historical_data.empty:
        analyze_and_predict(current_matches, historical_data)
