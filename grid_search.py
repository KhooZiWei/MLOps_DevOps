import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error


salesDF = pd.read_csv(r'C:\Users\ZKhoo\OneDrive - SLB\MLOps_DevOps\data\superstore_dataset.csv', encoding='latin1')

# Data Preprocessing
def remove_unnecessary_columns(df, columns_to_keep):

    # Select only the necessary columns from the DataFrame.
    df = df[columns_to_keep]
    return df

def parse_date(date_str):

    # Attempt to parse a date string with multiple formats.
    formats = ['%d/%m/%Y', '%d-%m-%Y']
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    return pd.NaT

def transform_date_column(df, column_name, new_column_name):

    # Transform the date column with multiple formats into a new column with datetime values.
    # Apply the parse_date function to each entry in the column
    df[new_column_name] = df[column_name].apply(parse_date)
    return df

def add_date_features(df, date_column):

    # Create new features (e.g., year, month) from the date column.
    df['Year'] = df[date_column].dt.year
    df['Month'] = df[date_column].dt.month
    return df

# Remove unnecessary columns
columns_to_keep = ['Order Date', 'Sales']
salesDF = remove_unnecessary_columns(salesDF, columns_to_keep)

# Transform the date column
salesDF = transform_date_column(salesDF, 'Order Date', 'Order Date')

# Add new features (e.g., year, month)
salesDF = add_date_features(salesDF, 'Order Date')

# Remove the "Order_Date" column
salesDF = salesDF.drop('Order Date', axis=1)


# Feature Engineering
def feature_engineering(df):
    """
    Perform feature engineering for time series forecasting:
    - Group by year and month to compute average sales.
    - Create lag features (previous month's sales).
    """
    
    # Group by year and month to compute average sales
    df_monthly = df.groupby(['Year', 'Month'])['Sales'].mean().reset_index()
    df_monthly.rename(columns={'Sales': 'Average_Sales'}, inplace=True)
    
    # Sort the DataFrame in ascending order by year and month
    df_monthly = df_monthly.sort_values(by=['Year', 'Month']).reset_index(drop=True)
    
    # Create lag features (previous month's sales)
    df_monthly['Lag_1'] = df_monthly['Average_Sales'].shift(1)
    df_monthly['Lag_2'] = df_monthly['Average_Sales'].shift(2)
    
    # Drop missing values created by lagging
    df_monthly = df_monthly.dropna()
    
    return df_monthly

df_monthly = feature_engineering(salesDF)


# Spliting Data
def prepare_data_for_modeling(df_monthly):
    
    # Define features and target
    X = df_monthly[['Lag_1', 'Lag_2']]
    y = df_monthly['Average_Sales']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = prepare_data_for_modeling(df_monthly)

# Perform Grid Search with Cross-Validation
def perform_grid_search(X_train, y_train):
    
    # Define the model
    rf = RandomForestRegressor(random_state=42)

    # Define the parameters grid
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [1.0, 'sqrt', 'log2']  
    }
    
    # Grid Search
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=5, 
        scoring='neg_mean_squared_error', 
        n_jobs=1, 
        verbose=2
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    return best_params, best_score

best_params, best_score = perform_grid_search(X_train, y_train)
print(best_params)