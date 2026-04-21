FROM python:3.12-slim

# Zmienne środowiskowe
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KB_DB_PATH=/data/knowledge_base.db \
    KB_UPLOAD_DIR=/data/uploads

WORKDIR /app

# Zależności systemowe (pdfplumber wymaga libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Zależności Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kod aplikacji
COPY . .

# Katalog na dane (baza SQLite + uploady + cache modelu)
RUN mkdir -p /data/uploads /data/models

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
