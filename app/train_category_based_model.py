import os
import json
import pandas as pd
import numpy as np
import joblib
import logging

from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error

import joblib

from preprocess import preprocess_prediciion_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
preprocessed_df = preprocess_prediciion_data(df)

# Define features and target
X = preprocessed_df[['Product Category', 'Day', 'Month', 'Weekday', 'isWeekend', 'isAfter25']]
y = preprocessed_df['LogTotalAmount']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"])

# Preprocessing
categorical_features = ['Product Category']
numerical_features = ['Day', 'Month', 'Weekday', 'isWeekend', 'isAfter25']

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('cat', categorical_transformer, categorical_features),
    ('num', numerical_transformer, numerical_features)
])

# Model pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=config["random_state"]))
])

grid_search = GridSearchCV(pipeline, config["category_model_params"], cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
logger.info("Training category-level model with GridSearchCV.")
grid_search.fit(X_train, y_train)

# Evaluate
logger.info("Evaluating model.")
y_pred_log = grid_search.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
logger.info("MAE: %.2f | RMSE: %.2f", mae, rmse)

# Save model
model_path = config["category_model_path"]
joblib.dump(grid_search.best_estimator_, model_path)
logger.info("Best category model saved to %s", model_path)
