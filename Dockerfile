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

# Pobierz model embeddingów podczas budowania obrazu
# (żeby nie pobierać przy każdym starcie kontenera)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Kod aplikacji
COPY . .

# Utwórz katalog na dane (baza SQLite + uploady)
RUN mkdir -p /data/uploads

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
