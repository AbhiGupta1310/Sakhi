FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (including data/)
COPY . .

# HF Spaces uses port 7860 by default
EXPOSE 7860

# Start the API on port 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
