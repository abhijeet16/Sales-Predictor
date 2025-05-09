# Sales Forecasting API

A robust API for forecasting daily sales using time-series analysis and machine learning. The API is built with FastAPI, containerized with Docker, and supports recursive predictions for future dates.

---

## Folder Structure

```
sales-predictor/
│
├── app/
│   ├── api.py                            # Main FastAPI application
│   ├── preprocess.py                     # Data preprocessing utilities
│   ├── train_full_transaction_model.py   # Script to train the full transaction model
│   ├── train_category_model.py           # Script to train the category sales model
│   ├── train_nar_daily_sales.py          # Script to train NAR model for forecasting
│   └── config.json                       # Configuration file for paths and parameters
│
├── data/
│   └── sales_data.xlsx                   # Raw sales data
│   └── sales_data_sample.xlsx            # Sample sales data
│
├── models/
│   ├── best_model_nar.pkl                # Trained model for forecasting 14 days ahead
│   ├── full_transaction_model.pkl        # Trained model for full transaction sales
│   └── category_sales_model.pkl          # Trained model for category sales
│
├── test/
│   ├── test_models.py                    # Unit tests for models
│   └── test_endpoints.py                 # Integration tests for API endpoints
│
├── .github/
│   └── workflows/
│       └── ci.yml                        # GitHub Actions workflow for CI
│
├── .gitignore                            # Git ignore file
├── config.json                           # Configuaration file containing all static values
├── Dockerfile                            # Dockerfile for containerizing the API
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation
```

---

## Prerequisites
- Docker Desktop or Rancher Desktop installed with proper wsl.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/abhijeet16/Sales-Predictor.git
cd Sales-Predictor
```

### 2. Build the Docker Image
```bash
docker build -t sales-api .
```

### 3. Run the Docker Container
```bash
docker run -p 8000:8000 sales-api
```

---

## API Usage
```bash
localhost:8000/docs
```
Open the above URL on any browser. To try any API, select the API and click on `Try it out`. Provide the expected payload and click on execute. The resonse and CURL commands will be shown in the response section.

### Endpoint: `/forecast-sales`
**Method:** POST  
Forecast daily sales for the next 14 days from an uploaded file. It supports both CSV(.csv) and Excel(.xlsx) formats.

#### Payload
Chose a file to upload. The file must contain data from atleast 30 days. If more, it will take the last 30 days to predict. The file also must have columns `Date` and `Total Amount`.

#### Example Response
```json
{
  "forecast": {
    "2024-01-02": 1335.94,
    "2024-01-03": 1202.72,
    "2024-01-04": 1847.56,
    "2024-01-05": 547.83,
    "2024-01-06": 5919.32,
    "2024-01-07": 112.05,
    "2024-01-08": 1138.48,
    "2024-01-09": 82.16,
    "2024-01-10": 341.18,
    "2024-01-11": 231.12,
    "2024-01-12": 1056.05,
    "2024-01-13": 194.49,
    "2024-01-14": 1636.65,
    "2024-01-15": 534.1
  }
}
```
#### CURL REQUEST
Open the bash terminal to perform this:
```
curl -X 'POST' \
  'http://localhost:8000/forecast-sales' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@./data/sales_data.xlsx;type=application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
```
---

### Endpoint: `/predict-sales-by-transaction`
**Method:** POST  
Predict the sales amount with given inputs Age, Gender, and Product Category for a specific date.

#### Example Payload
```json
{
  "age": 30,
  "gender": "Male",
  "product_category": "Clothing",
  "date": "2025-05-05"
}
```

#### Example Response
```json
{
  "predicted_sales": 177.33
}
```

#### CURL REQUEST
Open the bash terminal to perform this:
```
curl -X 'POST' \
  'http://localhost:8000/predict-sales-by-transaction' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 30,
  "gender": "Male",
  "product_category": "Clothing",
  "date": "2025-05-10"
}'
```


---

### Endpoint: `/predict-sales-by-category`
**Method:** POST  
Predict the total sales of a specific product category for a given date.

#### Example Payload
```json
{
  "product_category": "Beauty",
  "date": "2025-05-05"
}
```

#### Example Response
```json
{
  "predicted_sales": 287.02
}
```

#### CURL REQUEST
Open the bash terminal to perform this:
```
curl -X 'POST' \
  'http://localhost:8000/predict-sales-by-category' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "product_category": "Clothing",
  "date": "2025-05-10"
}'
```

---

## Modeling Approach
Two different types of modelling is experimented.
- Forecasting model:<br>
A forecasting model with a horizon of 14 days(2-weeks) is experimented with different algorithms. Refer to the [Dailey Sales Forecasting Report](notebooks/Daily-Sales-Modelling.pdf) for detailed explanation of modelling choices, feature engineering, and model training steps. The same can be reproduce at [Dailey Sales Forecasting Notebook](notebooks/Daily-Sales-Modelling.ipynb) <br>

- Date based prediction:<br>
Two prediction models with multivariate models are experimented to predicted sales on a particular day in future. Refer to the [Date based Prediction Report](notebooks/Date-based-Future-Prediction.pdf) for detailed explanation of modelling choices, feature engineering, and model training steps. The same can be reproduce at [Date based Prediction Notebook](notebooks/Date-based-Future-Prediction.ipynb) 
---


## Continuous Integration
This repository is powered by CI with GitHub Actions. The workflow is defined in `.github/workflows/ci.yml` and runs the following steps on every push or pull request to the `main` branch:.
The CI has only one job, i.e. test. In this job
- It builds the docker image
- Runs the container, which also starts the FastAPI
- All the test cases are run using `pytest`
- Any fail test casse or any failed stage in the job will fail the action
- At the end, it stops and cleans the docker container

---


## Testing
To perform tests locally,
- Create a python environment with `python -m venv sales-predictor`.
- Activate the environment `sales-predictor\Scripts\activate` on powershell.
- Install all dependencies `pip install -r requirement.txt`

### Run Unit Tests
To test the models and API endpoints, use the following command:
```bash
pytest test/test_models.py
```

### Run Integration Tests
The FastAPI application needs to be started before doing the API test (from home directory):
```bash
uvicorn app.api:app --reload
```

To test the models and API endpoints, use the following command:
```bash
pytest test/test_api.py
```
---


## Logging
Standard logging is done using `logging` library of python. The logs can be observed in the terminal where container is running in the logs section of running container in Docker Desktop.


---

## Enhancements
- Add authentication mechanisms for secure API access.
- Set up a cron-based pipeline for periodic model retraining from automatic downloading of latest data.
- Deploy the API on cloud platforms like AWS or GCP.
- Add detailed structuring logging and monitoring for production use.
- Rate limit to avoid misuse and resource exhaustion. This will stablize the APIs and also help in managing the costs
- The dataset was limited only for a year. Wider dataset may result in better model understandng.
- Configure Slack webhook, for immediate notification of CRON jons or test results on CI.

---

## Scale in production
- Deploy the APIs on cloud platforms like AWS, Azure or GCP
- Use a load balancer like NGINX to spread the traffic
- Auto-scaling can be configured with auto-spinning of pods
- Use a database service to store data like sales-data
- Performance monitoring using Grafana, Prometheus or similar service
- Auto-training models on cloud with dynamic updation database maybe (every 3 months)
- Versioning models for easy rollback

---

© 2025 Abhijeet Anand
