FROM python:3.11-slim-bullseye

# Install system dependencies for OpenCV, audio processing, and MediaPipe
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Render sets PORT env var)
EXPOSE 8000

# Run the application (use PORT env var from Render, fallback to 8000)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}

