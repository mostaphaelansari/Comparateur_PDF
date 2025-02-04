# Use Python slim base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create swap file for memory optimization
RUN dd if=/dev/zero of=/swapfile bs=1M count=1024 && \
    chmod 600 /swapfile && \
    mkswap /swapfile && \
    swapon /swapfile

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8080
EXPOSE 8080

# Startup command
CMD ["streamlit", "run", "app/main.py", \
    "--server.port=8080", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false"]