FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./
COPY config.json ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/

# Expose the port for the FastAPI application
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]