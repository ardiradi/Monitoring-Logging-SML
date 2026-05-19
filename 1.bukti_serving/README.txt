PETUNJUK: Letakkan screenshot bukti serving model di folder ini.

Contoh file:
- bukti_serving_mlflow.jpg (screenshot terminal mlflow models serve)
- bukti_serving_api.jpg (screenshot FastAPI docs /docs)
- bukti_serving_docker.jpg (screenshot docker ps jika menggunakan Docker)

Cara mendapatkan bukti serving:
1. Jalankan: mlflow models serve -m "runs:/<run_id>/model" -p 5001
   ATAU
2. Jalankan: python 7.Inference.py
3. Buka browser ke http://localhost:8000/docs
4. Screenshot terminal dan browser
5. Simpan screenshot di folder ini
