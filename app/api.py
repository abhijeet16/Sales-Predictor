import os
import json
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from io import BytesIO
import pandas as pd
import numpy as np
import joblib
from typing import Literal, Dict
import logging

from app.preprocess import preprocess_sales_data

# Logger Config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI App Init
app = FastAPI(
    title="Sales Forecasting API",
    description="Predict future sales based on transaction or product category",
    version="1.0.0"
)

# Load configuration
try:
    config_path = os.path.join(os.path.dirname(__file__), "../config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
except Exception as e:
    logger.error("Failed to load config file: %s", e)
    raise

# Load Pre-trained Models
try:
    full_model = joblib.load(config["full_model_path"])
    category_model = joblib.load(config["category_model_path"])
    daily_sales_model = joblib.load(config["dailey_sales_model_path"])  # Typo in "daily"
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error("Failed to load models: %s", e)
    raise RuntimeError("Model loading failed.")


# Input Schemas
class FullTransactionInput(BaseModel):
    """Schema for input data for full transaction sales prediction."""
    age: int = Field(..., gt=0, lt=120, description="Age of the customer (1-119).", example=30)
    gender: Literal['Male', 'Female'] = Field(..., description="Gender of the customer.", example="Male")
    product_category: Literal['Beauty', 'Clothing', 'Electronics'] = Field(..., description="Category of the product.", example="Clothing")
    date: str = Field(..., description="Date of the transaction in YYYY-MM-DD format.", example="2025-05-10")

class CategoryInput(BaseModel):
    """Schema for input data for category sales prediction."""
    product_category: Literal['Beauty', 'Clothing', 'Electronics'] = Field(..., description="Category of the product.", example="Clothing")
    date: str = Field(..., description="Date for the prediction in YYYY-MM-DD format.", example="2025-05-10")


# Helper Functions
def extract_date_features(date_str: str) -> Dict[str, int]:
    """
    Extract calendar-based features from a date string.

    Args
    ----------
    date_str: str
        Date in YYYY-MM-DD format.

    Returns
    ----------
    Dict[str, int]
        A dictionary containing extracted date features:
            - Day: Day of the month.
            - Month: Month of the year.
            - Weekday: Day of the week (0->Monday, 6->Sunday).
            - isWeekend: 1 if the day is a weekend, 0 otherwise.
            - isAfter25: 1 if the day is after the 25th of the month, 0 otherwise.

    Raises
    ----------
        HTTPException: If the date format is invlid.
    """
    try:
        dt = pd.to_datetime(date_str)
        return {
            "Day": dt.day,
            "Month": dt.month,
            "Weekday": dt.weekday(),
            "isWeekend": int(dt.weekday() in [5, 6]),
            "isAfter25": int(dt.day > 25)
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")


@app.post("/forecast-sales", summary="Forecast next 14 days of sales from uploaded file")
async def forecast_sales(file: UploadFile = File(...)) -> Dict[str, Dict[str, float]]:
    """
    Forecast daily sales for the next 14 days from an uploaded file. It supports both CSV(.csv) and Excel(.xlsx) formats.
    """
    try:
        # Validate file type
        if file.content_type not in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only .csv or .xlsx files are supported.")

        # Read the file into a DataFrame
        if file.content_type == "text/csv":
            # Detect delimiter dynamically
            content = await file.read()
            sample = content.decode("utf-8").splitlines()[0]
            delimiter = ";" if ";" in sample else ","
            df = pd.read_csv(BytesIO(content), delimiter=delimiter)
        else:
            df = pd.read_excel(BytesIO(await file.read()))

        # Ensure the required columns are present
        if "Date" not in df.columns or "Total Amount" not in df.columns:
            raise HTTPException(status_code=400, detail="File musst contain 'Date' and 'Total Amount' columns.")

        # Convert 'Date' column to datetime and sort by date
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        # Aggregate sales data by date (if needed)
        df = df.groupby("Date").sum().reset_index()

        # Ensure at least 30 days of data are present
        if len(df) < 30:
            raise HTTPException(status_code=400, detail="File musst contain at least 30 days of sales data.")

        # Preprocesss the data        
        preprocess_df = preprocess_sales_data(df)

        # Prepare the last input with a lag of 30
        lag = 30
        horizon = 14
        clipped_sales_log = preprocess_df['ClippedSalesLog'].values
        last_input = clipped_sales_log[-lag:].reshape(1, -1)

        # Forecast the next 14 days
        forecast_nar_log = daily_sales_model.predict(last_input).flatten()
        forecast_nar = np.expm1(forecast_nar_log)

        # Generate forecast dates
        forecast_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon)

        # Create a dictionary with dates as keys and forecsted sales as values
        forecastResult = {
            date.strftime("%Y-%m-%d"): round(sales, 2)
            for date, sales in zip(forecast_dates, forecast_nar)
        }

        logger.info("/forecast-sales: Forecasting successful")
        logger.info("Response: %s", forecastResult)
        return {"forecast": forecastResult}
    except Exception as e:
        logger.exception("Error during /forecast-sales")
        raise HTTPException(status_code=500, detail=str(e))
    

# Endpoint: Predict Sales by Transaction
@app.post("/predict-sales-by-transaction", summary="Predict sales for a transaction")
def predict_sales(input_data: FullTransactionInput) -> Dict[str, float]:
    """
    Predict the sales amount with given inputs Age, Gender, and Product Category for a specific date.
    """
    try:
        dateFeatures = extract_date_features(input_data.date)
        input_df = pd.DataFrame([{
            "Age": input_data.age,
            "Gender": input_data.gender,
            "Product Category": input_data.product_category,
            **dateFeatures
        }])

        log_prediction = full_model.predict(input_df)[0]
        prediction = np.expm1(log_prediction)
        logger.info("/predict-sales-by-transaction: Prediction successful")
        logger.info("Response: %s", round(prediction, 2))
        return {"predicted_sales": round(prediction, 2)}
    except Exception as e:
        logger.exception("Error during /predict-sales-by-transaction prediction")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint: Predict Category Forecast
@app.post("/predict-sales-by-category", summary="Predict sales by product category")
def predict_category(input_data: CategoryInput) -> Dict[str, float]:
    """
    Predict the total sales of a specific product category for a given date.
    """
    try:
        dateFeatures = extract_date_features(input_data.date)
        input_df = pd.DataFrame([{
            "Product Category": input_data.product_category,
            **dateFeatures
        }])

        log_prediction = category_model.predict(input_df)[0]
        prediction = np.expm1(log_prediction)
        logger.info("/predict-sales-by-category: Prediction successsful")
        logger.info("Response: %s", round(prediction, 2))
        return {"predicted_sales": round(prediction, 2)}
    except Exception as e:
        logger.exception("Error during /predict-category prediction")
        raise HTTPException(status_code=500, detail=str(e))
