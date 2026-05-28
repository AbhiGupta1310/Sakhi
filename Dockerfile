FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (including data/chroma_db/)
COPY . .

# HF Spaces uses port 7860 by default
EXPOSE 7860

# Health check so HF Spaces knows when the app is ready
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the API on port 7860 (no reload in production)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
