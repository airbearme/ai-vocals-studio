FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    libsndfile1 \
    sox \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_minimal.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_minimal.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p outputs models dataset

# Expose port
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV APP_DATA_DIR=/app
ENV MODEL_CACHE_DIR=/app/models
ENV OUTPUT_DIR=/app/outputs
ENV DATASET_DIR=/app/dataset

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app_streamlit.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
