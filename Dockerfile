# Use official Python 3.11 slim runtime as parent image
FROM python:3.11-slim

# Set system-level environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set container working directory
WORKDIR /usr/src/app

# Install basic OS-level libraries for Python dependency compilations
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt to leverage Docker build cache layers
COPY requirements.txt .

# Install dependencies without local package caching
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary application modules
COPY app/ app/
COPY pipeline/ pipeline/

# Create placeholder directories for database storage mounts
RUN mkdir -p data

# Expose Streamlit default network port
EXPOSE 8501

# Health check to ensure Streamlit server remains responsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit dashboard app
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

