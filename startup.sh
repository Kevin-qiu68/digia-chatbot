#!/bin/bash

# Azure App Service startup script for Streamlit

echo "Starting Streamlit application..."

# Install dependencies if needed
# pip install -r requirements.txt

# Create necessary directories
# mkdir -p vectordb
# mkdir -p data/company_docs

PORT=${PORT:-8000}

# Start Streamlit
streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false