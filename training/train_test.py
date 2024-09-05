import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from train import feature_engineering, split_data, train_model, get_model_metrics

"""A set of simple unit tests for protecting against forecasting in train.py"""

def test_feature_engineering():
    test_data = {
        'Date': ['01/01/2021', '01/02/2021', '01/03/2021', '01/04/2021'],
        'Sales': [100, 200, 150, 300]
    }

    df = pd.DataFrame(data=test_data)

    engineered_df = feature_engineering(df)

    # Verify that the new columns 'Year', 'Month', 'Lag_1', and 'Lag_2' are added
    assert 'Year' in engineered_df.columns
    assert 'Month' in engineered_df.columns
    assert 'Lag_1' in engineered_df.columns
    assert 'Lag_2' in engineered_df.columns

    # Verify that no null values are present after dropna
    assert not engineered_df.isnull().values.any()

def test_split_data():
    test_data = {
        'Year': [2021, 2021, 2021, 2021],
        'Month': [1, 2, 3, 4],
        'Average_Sales': [100, 200, 150, 300],
        'Lag_1': [0, 100, 200, 150],  
        'Lag_2': [0, 0, 100, 200]    
    }

    df = pd.DataFrame(data=test_data).dropna()

    X_train, X_valid, y_train, y_valid = split_data(df)

    # Verify that the data was split correctly
    assert X_train.shape[0] + X_valid.shape[0] == df.shape[0]  # Total rows should match original
    assert X_train.shape[0] > 0  # Training set should not be empty
    assert X_valid.shape[0] > 0  # Validation set should not be empty
    assert 'Average_Sales' not in X_train.columns  # Target should be excluded from features
    assert 'Average_Sales' not in X_valid.columns  # Target should be excluded from features

def test_train_model():
    X_train = pd.DataFrame({
        'Year': [2021, 2021, 2021],
        'Month': [1, 2, 3],
        'Lag_1': [50, 100, 200],
        'Lag_2': [25, 50, 100]
    }).dropna()
    
    y_train = pd.Series([200, 150, 250])

    best_params = {
        "n_estimators": 100,
        "max_depth": 5
    }

    model = train_model(X_train, y_train, best_params)

    assert isinstance(model, RandomForestRegressor)  # Verify that the model is a RandomForestRegressor

    # Check if model's parameters match the ones provided
    model_params = model.get_params()
    for param_name, param_value in best_params.items():
        assert param_name in model_params
        assert model_params[param_name] == param_value

def test_get_model_metrics():
    class MockModel:
        @staticmethod
        def predict(data):
            return np.array([100, 150])

    X_valid = pd.DataFrame({
        'Year': [2021, 2021],
        'Month': [3, 4],
        'Lag_1': [200, 150],
        'Lag_2': [100, 200]
    })
    
    y_valid = pd.Series([150, 300])

    mape = get_model_metrics(MockModel(), X_valid, y_valid)

    # Check if the MAPE value is calculated correctly 
    assert np.isclose(mape, 0.4166, atol=0.05)

# Running the tests would be done through a testing framework like pytest or unittest. Run pytest in the terminal to execute the tests.
