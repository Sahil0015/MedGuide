# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for building wheels (pymupdf) and runtime fonts/gl
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl libgl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Ensure data dirs exist at runtime
RUN mkdir -p data/knowledge_base/outputs data/knowledge_base/pdfs data/uploads data/lancedb

EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
