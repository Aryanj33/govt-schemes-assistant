# Sarkari Mitra — Backend Dockerfile
# Deploys to Hugging Face Spaces (Docker) — port 7860
# Force rebuild trigger: LFS purge complete

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face prefers UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install Python dependencies
COPY --chown=user backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download ML models during build to avoid runtime DNS/init issues
# This also sets the default cache directory to one the user can write to
ENV SENTENCE_TRANSFORMERS_HOME=$HOME/.cache
RUN python3 -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy application code
COPY --chown=user backend/ ./backend/
COPY --chown=user config/ ./config/
# Copy and reassemble split data chunks
COPY --chown=user data.zip.part_* ./
RUN cat data.zip.part_* > data.zip && unzip data.zip && rm data.zip.part_* data.zip

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=$HOME/app:$HOME/app/backend
ENV PORT=7860
# Force transformers to stay offline since we pre-downloaded
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# Expose HF Spaces port
EXPOSE 7860

# Run
CMD ["python", "backend/main.py"]
