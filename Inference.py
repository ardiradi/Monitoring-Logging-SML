"""
7.Inference.py
FastAPI-based Inference Server untuk Wine Quality Model (Kriteria 4)

Server ini menyediakan:
- Endpoint /predict untuk melakukan prediksi
- Endpoint /metrics untuk Prometheus scraping
- Endpoint /health untuk health check
- Integrasi dengan Prometheus metrics

Author: ardir
"""

import os
import time
import json
import numpy as np
import pandas as pd
import joblib
import psutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional

from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# PROMETHEUS METRICS
# ============================================================
REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    'inference_requests_total',
    'Total inference requests',
    ['method', 'endpoint', 'status'],
    registry=REGISTRY
)

REQUEST_LATENCY = Histogram(
    'inference_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY
)

PREDICTION_COUNTER = Counter(
    'inference_predictions_total',
    'Total predictions by class',
    ['predicted_class'],
    registry=REGISTRY
)

PREDICTION_CONFIDENCE = Gauge(
    'inference_prediction_confidence',
    'Latest prediction confidence',
    ['class_label'],
    registry=REGISTRY
)

MODEL_LOADED = Gauge(
    'inference_model_loaded',
    'Whether the model is loaded (1=yes, 0=no)',
    registry=REGISTRY
)

ACTIVE_REQUESTS = Gauge(
    'inference_active_requests',
    'Number of active requests',
    registry=REGISTRY
)

CPU_USAGE = Gauge(
    'inference_system_cpu_percent',
    'System CPU usage percent',
    registry=REGISTRY
)

MEMORY_USAGE = Gauge(
    'inference_system_memory_percent',
    'System memory usage percent',
    registry=REGISTRY
)

UPTIME = Gauge(
    'inference_uptime_seconds',
    'Server uptime in seconds',
    registry=REGISTRY
)

ERROR_COUNT = Counter(
    'inference_errors_total',
    'Total inference errors',
    ['error_type'],
    registry=REGISTRY
)

MODEL_VERSION_INFO = Info(
    'inference_model',
    'Model information',
    registry=REGISTRY
)


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================
class WineFeatures(BaseModel):
    """Input features untuk prediksi wine quality."""
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    wine_type: int  # 0=red, 1=white


class BatchWineFeatures(BaseModel):
    """Batch input untuk prediksi."""
    instances: List[WineFeatures]


class PredictionResponse(BaseModel):
    """Response prediksi."""
    prediction: str
    prediction_label: int
    confidence: float
    probabilities: dict


class BatchPredictionResponse(BaseModel):
    """Response prediksi batch."""
    predictions: List[PredictionResponse]
    count: int


# ============================================================
# GLOBAL VARIABLES
# ============================================================
model = None
scaler = None
encoders = None
feature_names = None
class_labels = ['low', 'medium', 'high']
start_time = time.time()

MODEL_DIR = os.environ.get('MODEL_DIR', 'wine_quality_preprocessing')
MODEL_PATH = os.environ.get('MODEL_PATH', 'model.pkl')


# ============================================================
# LIFESPAN (Load model on startup)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model dan resources saat startup."""
    global model, scaler, encoders, feature_names
    
    try:
        # Coba load model dari MLflow atau pickle
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"[INFO] Model loaded from {MODEL_PATH}")
        else:
            # Buat model sederhana untuk demonstrasi
            print("[WARNING] Model file tidak ditemukan. Membuat model dummy untuk demonstrasi.")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            # Fit dengan data dummy
            X_dummy = np.random.randn(100, 15)
            y_dummy = np.random.choice([0, 1, 2], 100)
            model.fit(X_dummy, y_dummy)
        
        # Load scaler
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"[INFO] Scaler loaded from {scaler_path}")
        
        # Load encoders
        encoders_path = os.path.join(MODEL_DIR, 'encoders.pkl')
        if os.path.exists(encoders_path):
            encoders = joblib.load(encoders_path)
            print(f"[INFO] Encoders loaded from {encoders_path}")
        
        MODEL_LOADED.set(1)
        MODEL_VERSION_INFO.info({
            'model_type': 'RandomForestClassifier',
            'version': '1.0',
            'dataset': 'wine_quality',
            'framework': 'scikit-learn',
        })
        
        print("[INFO] Inference server ready!")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        MODEL_LOADED.set(0)
    
    yield
    
    print("[INFO] Shutting down inference server...")


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Wine Quality ML Inference API",
    description="API untuk prediksi kualitas wine menggunakan model ML",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Wine Quality ML Inference API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "uptime_seconds": time.time() - start_time
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: WineFeatures):
    """Melakukan prediksi untuk satu instance."""
    ACTIVE_REQUESTS.inc()
    start = time.time()
    
    try:
        if model is None:
            ERROR_COUNT.labels(error_type='model_not_loaded').inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Prepare features
        feature_dict = features.model_dump()
        
        # Tambahkan engineered features
        feature_dict['total_acidity'] = (
            feature_dict['fixed_acidity'] + feature_dict['volatile_acidity']
        )
        feature_dict['free_sulfur_ratio'] = (
            feature_dict['free_sulfur_dioxide'] / 
            max(feature_dict['total_sulfur_dioxide'], 1)
        )
        feature_dict['alcohol_density_ratio'] = (
            feature_dict['alcohol'] / max(feature_dict['density'], 0.001)
        )
        
        # Convert to array
        input_df = pd.DataFrame([feature_dict])
        
        # Scale if scaler available
        if scaler is not None:
            scaled_cols = [c for c in input_df.columns if c in scaler.feature_names_in_]
            input_df[scaled_cols] = scaler.transform(input_df[scaled_cols])
        
        # Predict
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        pred_label = class_labels[int(prediction)]
        confidence = float(max(probabilities))
        
        # Update metrics
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='200').inc()
        PREDICTION_COUNTER.labels(predicted_class=pred_label).inc()
        
        for i, label in enumerate(class_labels):
            PREDICTION_CONFIDENCE.labels(class_label=label).set(float(probabilities[i]))
        
        latency = time.time() - start
        REQUEST_LATENCY.labels(endpoint='/predict').observe(latency)
        
        return PredictionResponse(
            prediction=pred_label,
            prediction_label=int(prediction),
            confidence=confidence,
            probabilities={label: float(prob) for label, prob in zip(class_labels, probabilities)}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.labels(error_type='prediction_error').inc()
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='500').inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.dec()
        # Update system metrics
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().percent)
        UPTIME.set(time.time() - start_time)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchWineFeatures):
    """Melakukan prediksi untuk batch instances."""
    predictions = []
    for instance in batch.instances:
        result = await predict(instance)
        predictions.append(result)
    
    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions)
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    import uvicorn
    
    print("=" * 60)
    print("  WINE QUALITY ML INFERENCE SERVER")
    print("=" * 60)
    print("  API:     http://localhost:8000")
    print("  Docs:    http://localhost:8000/docs")
    print("  Metrics: http://localhost:8000/metrics")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
