# Sarkari Mitra — Backend Dockerfile
# Deploys to Hugging Face Spaces (Docker) — port 7860

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/

# Copy FAISS index data (bundled into image — ~34MB)
COPY data/ ./data/

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/backend
# HF Spaces requires port 7860; Railway/local uses 8080 — both read from PORT env
ENV PORT=7860

# Expose HF Spaces port
EXPOSE 7860

# Run
CMD ["python", "backend/main.py"]
