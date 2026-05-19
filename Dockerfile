FROM python:3.12-slim

WORKDIR /app

COPY Membangun_model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Monitoring\ dan\ Logging/7.Inference.py .
COPY Membangun_model/heart_disease_preprocessing/ ./heart_disease_preprocessing/

EXPOSE 8000

CMD ["python", "7.Inference.py"]
