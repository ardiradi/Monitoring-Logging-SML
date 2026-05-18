PETUNJUK: Letakkan screenshot monitoring Grafana di folder ini.

Untuk Advanced: minimal 10 metriks berbeda. Contoh file:
- 1.monitoring_predictions_total.jpg
- 2.monitoring_prediction_latency.jpg
- 3.monitoring_cpu_usage.jpg
- 4.monitoring_memory_usage.jpg
- 5.monitoring_prediction_confidence.jpg
- 6.monitoring_predictions_by_class.jpg
- 7.monitoring_error_total.jpg
- 8.monitoring_data_drift.jpg
- 9.monitoring_uptime.jpg
- 10.monitoring_requests_in_progress.jpg

PENTING: Nama dashboard HARUS menggunakan username akun Dicoding Anda!

Cara mendapatkan screenshot:
1. Jalankan seluruh stack: docker-compose up -d
   ATAU jalankan masing-masing service secara manual
2. Buka Grafana di http://localhost:3000 (admin/admin)
3. Tambahkan Data Source Prometheus (http://prometheus:9090)
4. Buat Dashboard BARU dengan nama = username Dicoding Anda
5. Tambahkan Panel untuk setiap metriks
6. Screenshot setiap panel monitoring
7. Simpan di folder ini
