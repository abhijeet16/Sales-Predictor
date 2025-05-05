import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

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


# Utility for feature generation
def generate_test_features(date_str):
    dt = pd.to_datetime(date_str)
    return {
        'Day': dt.day,
        'Month': dt.month,
        'Weekday': dt.weekday(),
        'isWeekend': int(dt.weekday() in [5, 6]),
        'isAfter25': int(dt.day > 25)
    }

# Test forcast Model
def test_best_model_nar():
    # Load the model
    assert os.path.exists(config["dailey_sales_model_path"]), f"Forest model file not found."
    model_nar = joblib.load(config["dailey_sales_model_path"])

    # Prepare sample input (last 30 days of sales data)
    lag = 30
    horizon = 14
    sample_sales_data = np.log1p(np.arange(1, lag + 1))
    sample_input = sample_sales_data.reshape(1, -1)

    # Predict the next 14 days
    forecast_nar_log = model_nar.predict(sample_input).flatten()
    forecast_nar = np.expm1(forecast_nar_log)  # Reverse log transformation

    # Validate the forecast
    assert len(forecast_nar) == horizon, f"Expected {horizon} predictions, got {len(forecast_nar)}"
    assert all(forecast_nar > 0), "All forecasted sales should be positive."

    # Print the forecasted values
    forecast_dates = pd.date_range(start="2025-05-01", periods=horizon)
    forecast_df = pd.DataFrame({
        "Date": forecast_dates.strftime("%Y-%m-%d"),
        "Forecasted_Sales": forecast_nar
    })
    print("Test passed. \nForecasted values:\n", forecast_df)


# Test transaction Model
def test_transaction_model():
    assert Path(config["full_model_path"]).exists(), "Transaction model file not found."
    model = joblib.load(config["full_model_path"])
    features = generate_test_features("2025-05-10")
    input_df = pd.DataFrame([{
        "Age": 30,
        "Gender": "Male",
        "Product Category": "Clothing",
        **features
    }])
    pred = model.predict(input_df)
    assert pred.shape == (1,), "Prediction should return a single value."
    assert isinstance(pred[0], float), "Prediction should be a float."
    print("Full model test passed. Predicted log value:", pred[0])

# Test Category Model
def test_category_model():
    assert Path(config["category_model_path"]).exists(), "Category model file not found."
    model = joblib.load(config["category_model_path"])
    features = generate_test_features("2025-05-10")
    input_df = pd.DataFrame([{
        "Product Category": "Clothing",
        **features
    }])
    pred = model.predict(input_df)
    assert pred.shape == (1,), "Prediction should return a single value."
    assert isinstance(pred[0], (float, np.floating)), "Prediction should be a float."
    print("Category model test passed. Predicted log value:", pred[0])

# Run All Tests
if __name__ == "__main__":
    test_best_model_nar()
    test_transaction_model()
    test_category_model()
