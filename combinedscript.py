import os
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Gör allt, callar api för att uppdatera de nuvarande matcherna i [football_matches2.csv] och går igenom historical_matches[i] för att hitta resultat.

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
    headers = {'X-Auth-Token': '5412d5e47c5d4c34a313710ed20eecb7'} 
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

# === Step 2: Load Historical Data ===
def load_historical_data(file_pattern, num_files):
    """
    Loads and combines multiple historical match CSV files into a single DataFrame.

    Args:
        file_pattern (str): The pattern of the filenames, e.g., "HISTORICAL/historical_matches_{}.csv".
        num_files (int): The number of historical CSV files to load.

    Returns:
        pd.DataFrame: A combined DataFrame containing all historical match data.
    """
    historical_dfs = []
    for i in range(num_files):
        file_name = file_pattern.format(i)
        
        # Ensure the file exists
        if os.path.exists(file_name):
            print(f"Loading file: {file_name}")
            historical_dfs.append(pd.read_csv(file_name))
        else:
            print(f"Warning: File not found - {file_name}")
    
    # Combine all historical DataFrames into one
    if historical_dfs:
        historical_df = pd.concat(historical_dfs, ignore_index=True)
        print(f"Successfully loaded {len(historical_dfs)} historical files.")
        return historical_df
    else:
        print("No historical files loaded.")
        return pd.DataFrame()  # Return an empty DataFrame if no files were loaded

# === Step 3: Analyze and Predict ===
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
        ((historical_df['HomeTeam'] == home_team) & (historical_df['AwayTeam'] == away_team)) |
        ((historical_df['HomeTeam'] == away_team) & (historical_df['AwayTeam'] == home_team))
    ]

    # Prepare data for training
    if historical_matches.empty:
        print(f"No historical data available for {home_team} vs {away_team}.")
        return

    # Create labels (win=1, draw=0, loss=-1) based on the home team's perspective
    historical_matches['Result'] = historical_matches.apply(
        lambda row: 1 if row['FTHG'] > row['FTAG'] else (-1 if row['FTHG'] < row['FTAG'] else 0), axis=1
    )

    # Features: Home goals (FTHG), Away goals (FTAG)
    X = historical_matches[['FTHG', 'FTAG']]
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
    # We'll use average goals scored by the home and away teams for prediction
    avg_home_goals = historical_matches['FTHG'].mean()
    avg_away_goals = historical_matches['FTAG'].mean()
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
