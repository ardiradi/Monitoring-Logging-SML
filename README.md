# 📊 Heart Disease MLOps — Monitoring & Logging

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Prometheus-Metrics-E6522C?logo=prometheus&logoColor=white" alt="Prometheus">
  <img src="https://img.shields.io/badge/Grafana-Dashboard-F46800?logo=grafana&logoColor=white" alt="Grafana">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

## 📋 Overview

Tahap **Monitoring & Logging** dari end-to-end MLOps pipeline. Mengimplementasikan sistem monitoring real-time untuk model prediksi penyakit jantung menggunakan **FastAPI** (inference), **Prometheus** (metrics collection), dan **Grafana** (visualization & alerting).

## 🗂️ Struktur Direktori

```
├── 7.Inference.py                  # FastAPI inference server
├── 3.prometheus_exporter.py        # Custom Prometheus metrics exporter
├── 2.prometheus.yml                # Prometheus scraping configuration
├── Dockerfile                      # Docker image for inference API
├── Dockerfile.exporter             # Docker image for metrics exporter
├── docker-compose.yml              # Full stack orchestration
├── heart_disease_preprocessing/    # Model input data
├── 1.bukti_serving/                # 📸 Bukti serving API
├── 4.bukti monitoring Prometheus/  # 📸 Bukti Prometheus metrics
│   ├── 1.monitoring_predictions_total.jpg
│   ├── 2.monitoring_cpu_usage.jpg
│   ├── 3.monitoring_memory_usage.jpg
│   ├── 4.monitoring_prediction_latency.jpg
│   └── 5.monitoring_data_drift.jpg
├── 5.bukti monitoring Grafana/     # 📸 Bukti Grafana dashboard
│   ├── 1.monitoring_predictions_total.jpg
│   ├── 2.monitoring_prediction_latency.jpg
│   ├── 3.monitoring_cpu_usage.jpg
│   ├── ...
│   └── 12.monitoring_feature_values.jpg
└── 6.bukti alerting Grafana/       # 📸 Bukti alerting rules
    ├── 1.rules_overview.jpg
    ├── 2.rules_high_cpu_detail.jpg
    └── 3.rules_all_active.jpg
```

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│   Client/User   │────▶│   FastAPI     │────▶│   ML Model   │
│   POST /predict │     │  Inference    │     │ Random Forest│
└─────────────────┘     └──────┬───────┘     └──────────────┘
                               │ metrics
                               ▼
                        ┌──────────────┐     ┌──────────────┐
                        │  Prometheus  │────▶│   Grafana    │
                        │  :9090       │     │   :3000      │
                        └──────────────┘     └──────┬───────┘
                                                    │
                                              ┌─────▼──────┐
                                              │  Alerting   │
                                              │  Rules      │
                                              └────────────┘
```

## 📈 Custom Prometheus Metrics (12 Metrics)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `ml_predictions_total` | Counter | Total prediksi per kelas |
| 2 | `ml_prediction_latency_seconds` | Histogram | Latensi prediksi |
| 3 | `ml_cpu_usage_percent` | Gauge | Penggunaan CPU |
| 4 | `ml_memory_usage_percent` | Gauge | Penggunaan memory |
| 5 | `ml_memory_usage_bytes` | Gauge | Memory dalam bytes |
| 6 | `ml_predictions_by_class` | Gauge | Distribusi prediksi per kelas |
| 7 | `ml_prediction_confidence` | Histogram | Confidence score |
| 8 | `ml_data_drift_score` | Gauge | Skor data drift per fitur |
| 9 | `ml_error_count` | Counter | Jumlah error |
| 10 | `ml_model_uptime_seconds` | Gauge | Uptime model |
| 11 | `ml_requests_in_progress` | Gauge | Request yang sedang diproses |
| 12 | `ml_feature_values` | Gauge | Nilai fitur real-time |

## 🔔 Grafana Alerting Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| High CPU Usage | CPU > 80% for 5m | Warning |
| High Memory Usage | Memory > 85% for 5m | Warning |
| High Error Rate | Errors > 10/min | Critical |
| High Latency | p95 > 1s | Warning |

## 🚀 Cara Menjalankan

```bash
# Clone repository
git clone https://github.com/ardiradi/Monitoring-Logging-SML.git
cd Monitoring-Logging-SML

# Jalankan full stack dengan Docker Compose
docker-compose up -d

# Akses services:
# - FastAPI Inference: http://localhost:8000
# - Prometheus:        http://localhost:9090
# - Grafana:           http://localhost:3000

# Test inference API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Model status |
| `POST` | `/predict` | Predict heart disease |
| `GET` | `/metrics` | Prometheus metrics |

## 🔗 Related Repositories

| Component | Repository |
|-----------|------------|
| 🔬 Experimentation | [Eksperimen_SML_ardir](https://github.com/ardiradi/Eksperimen_SML_ardir) |
| 📦 Model Building | [Membangun-Model-SML](https://github.com/ardiradi/Membangun-Model-SML) |
| 🔄 CI/CD Workflow | [Workflow-CI](https://github.com/ardiradi/Workflow-CI) |

---

<p align="center">
  <b>Part of the Heart Disease MLOps Pipeline</b><br>
  Built as part of Dicoding — Membangun Sistem Machine Learning
</p>
