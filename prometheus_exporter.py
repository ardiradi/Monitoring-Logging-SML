"""
prometheus_exporter.py
Custom Prometheus Exporter untuk ML Model Monitoring (Kriteria 4)

Mengekspos 10+ metriks monitoring ke Prometheus:
1.  model_predictions_total        - Total prediksi yang dilakukan
2.  model_prediction_latency       - Latensi prediksi (histogram)
3.  model_prediction_confidence    - Skor confidence prediksi
4.  model_prediction_by_class      - Distribusi prediksi per kelas
5.  model_error_total              - Total error yang terjadi
6.  model_data_drift_score         - Skor data drift
7.  model_feature_value            - Statistik fitur input
8.  system_cpu_usage_percent       - Penggunaan CPU
9.  system_memory_usage_percent    - Penggunaan memori
10. system_memory_usage_bytes      - Penggunaan memori (bytes)
11. model_uptime_seconds           - Uptime model server
12. model_requests_in_progress     - Request yang sedang diproses

Author: ardir
"""

import time
import random
import threading
import psutil
from prometheus_client import (
    start_http_server,
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest
)
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# REGISTRY & METRICS DEFINITION
# ============================================================
REGISTRY = CollectorRegistry()

# 1. Total Predictions Counter
PREDICTIONS_TOTAL = Counter(
    'model_predictions_total',
    'Total number of predictions made by the model',
    ['model_name', 'version'],
    registry=REGISTRY
)

# 2. Prediction Latency Histogram
PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Time spent processing prediction requests',
    ['model_name'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY
)

# 3. Prediction Confidence Gauge
PREDICTION_CONFIDENCE = Gauge(
    'model_prediction_confidence',
    'Latest prediction confidence score',
    ['model_name', 'class_label'],
    registry=REGISTRY
)

# 4. Predictions by Class Counter
PREDICTIONS_BY_CLASS = Counter(
    'model_prediction_by_class_total',
    'Number of predictions per class',
    ['model_name', 'predicted_class'],
    registry=REGISTRY
)

# 5. Error Counter
ERROR_TOTAL = Counter(
    'model_error_total',
    'Total number of prediction errors',
    ['model_name', 'error_type'],
    registry=REGISTRY
)

# 6. Data Drift Score Gauge
DATA_DRIFT_SCORE = Gauge(
    'model_data_drift_score',
    'Data drift score (0=no drift, 1=full drift)',
    ['model_name', 'feature_name'],
    registry=REGISTRY
)

# 7. Feature Value Gauge (statistik fitur input)
FEATURE_VALUE = Gauge(
    'model_feature_value',
    'Current feature value statistics',
    ['model_name', 'feature_name', 'statistic'],
    registry=REGISTRY
)

# 8. System CPU Usage Gauge
CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'Current CPU usage percentage',
    registry=REGISTRY
)

# 9. System Memory Usage (percent)
MEMORY_USAGE_PERCENT = Gauge(
    'system_memory_usage_percent',
    'Current memory usage percentage',
    registry=REGISTRY
)

# 10. System Memory Usage (bytes)
MEMORY_USAGE_BYTES = Gauge(
    'system_memory_usage_bytes',
    'Current memory usage in bytes',
    registry=REGISTRY
)

# 11. Model Uptime
MODEL_UPTIME = Gauge(
    'model_uptime_seconds',
    'Model server uptime in seconds',
    ['model_name'],
    registry=REGISTRY
)

# 12. Requests In Progress
REQUESTS_IN_PROGRESS = Gauge(
    'model_requests_in_progress',
    'Number of requests currently being processed',
    ['model_name'],
    registry=REGISTRY
)

# Model Info
MODEL_INFO = Info(
    'model',
    'Information about the ML model',
    registry=REGISTRY
)


# ============================================================
# SIMULATED METRICS COLLECTOR
# ============================================================
MODEL_NAME = "wine_quality_rf"
MODEL_VERSION = "1.0"
CLASS_LABELS = ['low', 'medium', 'high']
FEATURE_NAMES = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid',
    'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
    'total_sulfur_dioxide', 'density', 'pH', 'sulphates',
    'alcohol', 'wine_type', 'total_acidity',
    'free_sulfur_ratio', 'alcohol_density_ratio'
]

start_time = time.time()


def simulate_prediction():
    """Simulasi sebuah prediksi untuk mengupdate metriks."""
    # Simulasi latency
    latency = random.uniform(0.001, 0.05)
    time.sleep(latency)
    
    # Update prediction counter
    PREDICTIONS_TOTAL.labels(model_name=MODEL_NAME, version=MODEL_VERSION).inc()
    
    # Update latency
    PREDICTION_LATENCY.labels(model_name=MODEL_NAME).observe(latency)
    
    # Simulasi prediksi kelas
    predicted_class = random.choices(
        CLASS_LABELS, 
        weights=[0.1, 0.7, 0.2],  # distribusi realistis
        k=1
    )[0]
    PREDICTIONS_BY_CLASS.labels(
        model_name=MODEL_NAME, predicted_class=predicted_class
    ).inc()
    
    # Simulasi confidence scores
    for label in CLASS_LABELS:
        confidence = random.uniform(0.1, 0.95)
        PREDICTION_CONFIDENCE.labels(
            model_name=MODEL_NAME, class_label=label
        ).set(confidence)
    
    # Simulasi error (jarang terjadi)
    if random.random() < 0.02:
        error_types = ['input_validation', 'timeout', 'model_error']
        ERROR_TOTAL.labels(
            model_name=MODEL_NAME,
            error_type=random.choice(error_types)
        ).inc()


def update_system_metrics():
    """Update metriks sistem (CPU, memory)."""
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    
    mem = psutil.virtual_memory()
    MEMORY_USAGE_PERCENT.set(mem.percent)
    MEMORY_USAGE_BYTES.set(mem.used)


def update_drift_metrics():
    """Simulasi metriks data drift."""
    for feature in FEATURE_NAMES[:5]:  # Top 5 fitur
        drift_score = random.uniform(0.0, 0.3)  # drift rendah
        DATA_DRIFT_SCORE.labels(
            model_name=MODEL_NAME, feature_name=feature
        ).set(drift_score)


def update_feature_stats():
    """Simulasi statistik fitur input."""
    feature_stats = {
        'fixed_acidity': {'mean': 7.2, 'std': 1.3},
        'volatile_acidity': {'mean': 0.34, 'std': 0.16},
        'alcohol': {'mean': 10.5, 'std': 1.2},
        'pH': {'mean': 3.2, 'std': 0.16},
        'density': {'mean': 0.995, 'std': 0.003},
    }
    for feature, stats in feature_stats.items():
        for stat_name, stat_value in stats.items():
            # Tambahkan sedikit variasi
            value = stat_value + random.uniform(-0.1, 0.1) * stat_value
            FEATURE_VALUE.labels(
                model_name=MODEL_NAME,
                feature_name=feature,
                statistic=stat_name
            ).set(value)


def update_uptime():
    """Update model uptime."""
    uptime = time.time() - start_time
    MODEL_UPTIME.labels(model_name=MODEL_NAME).set(uptime)


def metrics_updater():
    """Background thread yang mengupdate metriks secara periodik."""
    # Set model info
    MODEL_INFO.info({
        'model_name': MODEL_NAME,
        'model_version': MODEL_VERSION,
        'framework': 'scikit-learn',
        'algorithm': 'RandomForestClassifier',
        'dataset': 'wine_quality',
        'n_classes': '3',
    })
    
    while True:
        try:
            # Simulasi beberapa prediksi
            n_predictions = random.randint(1, 5)
            REQUESTS_IN_PROGRESS.labels(model_name=MODEL_NAME).set(n_predictions)
            
            for _ in range(n_predictions):
                simulate_prediction()
            
            REQUESTS_IN_PROGRESS.labels(model_name=MODEL_NAME).set(0)
            
            # Update sistem metriks
            update_system_metrics()
            update_drift_metrics()
            update_feature_stats()
            update_uptime()
            
        except Exception as e:
            print(f"[ERROR] Metrics update failed: {e}")
        
        time.sleep(5)  # Update setiap 5 detik


# ============================================================
# MAIN
# ============================================================
def main():
    """Menjalankan Prometheus exporter pada port 8001."""
    port = 8001
    
    print(f"{'='*60}")
    print(f"  PROMETHEUS EXPORTER - Wine Quality ML Model")
    print(f"{'='*60}")
    print(f"  Port: {port}")
    print(f"  Metrics endpoint: http://localhost:{port}/metrics")
    print(f"  Model: {MODEL_NAME} v{MODEL_VERSION}")
    print(f"{'='*60}")
    
    # Start metrics updater thread
    updater_thread = threading.Thread(target=metrics_updater, daemon=True)
    updater_thread.start()
    
    # Start HTTP server for Prometheus scraping
    from prometheus_client import start_http_server as _start
    _start(port, registry=REGISTRY)
    
    print(f"\n[INFO] Exporter berjalan di http://localhost:{port}")
    print(f"[INFO] Tekan Ctrl+C untuk berhenti.\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Exporter dihentikan.")


if __name__ == '__main__':
    main()
