import requests
from datetime import datetime

# Base URL for local testing
BASE_URL = "http://localhost:8000"

# Utilty Functions
def validate_response(response, expected_keys):
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    data = response.json()
    for key in expected_keys:
        assert key in data, f"Missing key '{key}' in response."
    return data

# Test: Forecast Next 14 Days
def test_forecast_sales():
    print("\nTesting endpoint: /forecast-sales")

    # Upload the file and test the API
    sample_csv_path = "./data/sales_data_sample.xlsx"
    with open(sample_csv_path, "rb") as file:
        response = requests.post(
            f"{BASE_URL}/forecast-sales",
            files={"file": ("test_forecast_sales.xlsx", file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        )
    
    # Validate the response
    result = validate_response(response, expected_keys=["forecast"])
    assert isinstance(result["forecast"], dict), "Forecast should be a dictionry."
    assert len(result["forecast"]) == 14, "Forecast should contain 14 days of predictions."
    
    print("Test passed. \nForecasted values:", result["forecast"])


# Test: Predict Sales Transaction
def test_predict_sales_transaction():
    print("\nTesting endpoint: /predict-sales-by-transaction")
    payload = {
        "age": 28,
        "gender": "Male",
        "product_category": "Clothing",
        "date": datetime.today().strftime("%Y-%m-%d")
    }
    response = requests.post(f"{BASE_URL}/predict-sales-by-transaction", json=payload)
    result = validate_response(response, expected_keys=["predicted_sales"])
    assert isinstance(result['predicted_sales'], float), "Predction should be a float."
    print("Test passed. \nPredicted value:", result['predicted_sales'])


# Test: Predict Category Sales
def test_predict_sales_by_category():
    print("\nTesting endpoint: /predict-sales-by-category")
    payload = {
        "product_category": "Clothing",
        "date": datetime.today().strftime("%Y-%m-%d")
    }
    response = requests.post(f"{BASE_URL}/predict-sales-by-category", json=payload)
    result = validate_response(response, expected_keys=["predicted_sales"])
    assert isinstance(result['predicted_sales'], float), "Predction should be a float."
    print("Test passed. \nPredicted value:", result['predicted_sales'])

# Run Tests
if __name__ == "__main__":
    try:
        test_forecast_sales()
        test_predict_sales_transaction()
        test_predict_sales_by_category()
        print("\nAll API tests passed successsfully.")
    except AssertionError as ae:
        print("Test failed:", ae)
    except Exception as e:
        print("Unexpected error:", e)
