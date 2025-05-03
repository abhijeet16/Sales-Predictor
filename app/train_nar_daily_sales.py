# Import libraries
import os
import json
import numpy as np
import pandas as pd
import logging

import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocess import preprocess_sales_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define function to create dataset with defined lags
def create_lagged_dataset(series, lag, horizon):
    X, y = [], []
    for i in range(len(series)-lag-horizon+1):
        X.append(series[i:i+lag])
        y.append(series[i+lag:i+lag+horizon])
    return np.array(X), np.array(y)


# Load configuration
try:
    config_path = os.path.join(os.path.dirname(__file__), "../config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
except Exception as e:
    logger.error("Failed to load config: %s", e)
    raise

# Load data
try:
    data_path = config["data_path"]
    df = pd.read_excel(data_path)
    logger.info("Data loaded successfully.")
except Exception as e:
    logger.error("Failed to load data: %s", e)
    raise

# Feature engineering
logger.info("Performing feature engineering.")
preprocess_df = preprocess_sales_data(df)


# Define series, LAG, and 
series = preprocess_df['ClippedSalesLog'].values
lag, horizon = 30, 14

# Create lagged dataset
X, y = create_lagged_dataset(series, lag, horizon)

# Train/test split
X_train, X_test = X[:-1], X[-1:]
y_train, y_test = y[:-1], y[-1]

# Define model and parameter grid
base_model = MLPRegressor(max_iter=3000, random_state=config["random_state"])
param_grid = {
    'estimator__hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'estimator__activation': ['relu', 'tanh'],
    'estimator__alpha': [0.0001, 0.001, 0.01],
    'estimator__learning_rate': ['constant', 'adaptive']
}

# Wrap in MultiOutputRegressor for multi-step forecasting
multi_output_model = MultiOutputRegressor(base_model)
grid_search = GridSearchCV(multi_output_model, param_grid,
                           scoring='neg_root_mean_squared_error',
                           cv=3, n_jobs=-1, verbose=1)

# Fit grid search
grid_search.fit(X_train, y_train)

# Evaluate best model
best_model_nar = grid_search.best_estimator_
y_pred = best_model_nar.predict(X_test).flatten()

# Calculate the MAE and RMSE on log transformed values
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Output
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best 14-day MAE: {mae:.2f}")
print(f"Best 14-day RMSE: {rmse:.2f}")

# Log the results
logger.info("Best Parameters: %s", grid_search.best_params_)
logger.info("MAE: %.2f | RMSE: %.2f", mae, rmse)

# Save best model
save_model_path = config["dailey_sales_model_path"]
joblib.dump(best_model_nar, save_model_path)
logger.info("Best category model saved to %s", save_model_path)