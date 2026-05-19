PETUNJUK: Letakkan screenshot alerting Grafana di folder ini.

Untuk Advanced: minimal 3 alerting rules. Contoh file:
- 1.rules_high_latency.jpg
- 2.notifikasi_high_latency.jpg
- 3.rules_high_error_rate.jpg
- 4.notifikasi_high_error_rate.jpg
- 5.rules_high_cpu_usage.jpg
- 6.notifikasi_high_cpu_usage.jpg

Cara membuat alerting:
1. Buka Grafana di http://localhost:3000
2. Masuk ke Alerting > Alert Rules
3. Buat rule baru:
   Rule 1: High Latency Alert
   - Condition: model_prediction_latency_seconds > 0.1
   - For: 1m
   
   Rule 2: High Error Rate Alert  
   - Condition: rate(model_error_total[5m]) > 0.05
   - For: 2m
   
   Rule 3: High CPU Usage Alert
   - Condition: system_cpu_usage_percent > 80
   - For: 5m

4. Screenshot setiap rule dan notifikasinya
5. Simpan di folder ini
