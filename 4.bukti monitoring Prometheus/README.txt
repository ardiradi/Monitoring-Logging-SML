PETUNJUK: Letakkan screenshot monitoring Prometheus di folder ini.

Minimal 3 metriks berbeda. Contoh file yang harus ada:
- 1.monitoring_predictions_total.jpg
- 2.monitoring_prediction_latency.jpg
- 3.monitoring_cpu_usage.jpg
- 4.monitoring_memory_usage.jpg (opsional, untuk lebih banyak poin)
- 5.monitoring_error_total.jpg (opsional)

Cara mendapatkan screenshot:
1. Jalankan prometheus_exporter: python 3.prometheus_exporter.py
2. Jalankan Prometheus: prometheus --config.file=2.prometheus.yml
3. Buka browser ke http://localhost:9090
4. Masuk ke tab "Graph"
5. Ketik query untuk setiap metriks:
   - model_predictions_total
   - model_prediction_latency_seconds_bucket
   - system_cpu_usage_percent
   - system_memory_usage_percent
   - model_error_total
6. Klik "Execute" dan screenshot hasilnya
7. Simpan di folder ini
