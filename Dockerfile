FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances en version allégée / CPU
RUN pip install --no-cache-dir --default-timeout=1000 \
    fastapi \
    uvicorn \
    pandas \
    scikit-learn \
    prometheus-fastapi-instrumentator \
    joblib \
    mlflow \
    "xgboost[cpu]"

COPY src /app/src

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]