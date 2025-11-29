# Use Python 3.11.5 base image
FROM python:3.11.5-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for yfinance, pandas, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
# RUN apt-get update && apt-get install -y build-essential gcc g++ libopenblas-dev

# Copy project files
COPY predictor.py news_sentiment.py app.py requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Dash
EXPOSE 8050

# Run the Dash app
CMD ["python", "app.py"]
