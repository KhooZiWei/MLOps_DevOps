import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error


# Perform feature engineering
def feature_engineering(df):
    """
    Perform feature engineering for time series forecasting:
    - Ensure 'Date' column is of date type.
    - Add new date features (year, month).
    - Group by year and month to compute average sales.
    - Create lag features (previous month's sales).
    """
    # Ensure the 'Date' column is of datetime type
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Add new date features (year, month)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    # Remove the date column after transformation
    df = df.drop('Date', axis=1)

    # Group by year and month to compute average sales
    data_df = df.groupby(['Year', 'Month']).mean().reset_index()
    data_df.rename(columns={'Sales': 'Average_Sales'}, inplace=True)
    
    # Sort the DataFrame by year and month
    data_df = data_df.sort_values(by=['Year', 'Month']).reset_index(drop=True)
    
    # Create lag features (previous month's sales)
    data_df['Lag_1'] = data_df['Average_Sales'].shift(1)
    data_df['Lag_2'] = data_df['Average_Sales'].shift(2)
    
    # Drop missing values
    data_df = data_df.dropna()
    
    return data_df

# Split the data into training and testing sets
def split_data(data_df):
    # Split the dataframe into training and validation datasets
    
    X = data_df.drop(['Average_Sales'], axis=1)
    y = data_df['Average_Sales']
    
    # Combine features and target variable into a single dataframe
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    return X_train, X_valid, y_train, y_valid

# Train the RandomForest model
def train_model(X_train, y_train, best_params):
    # Train the RandomForest model with the given datasets and parameters

    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Evaluate the model
def get_model_metrics(model, X_valid, y_valid):
    # Evaluate the RandomForest model on validation data.
    # Make predictions
    predictions = model.predict(X_valid)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = mean_absolute_percentage_error(y_valid, predictions)

    return mape
