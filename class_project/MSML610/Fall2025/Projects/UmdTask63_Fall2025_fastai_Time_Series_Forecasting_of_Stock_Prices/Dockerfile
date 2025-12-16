FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY .env .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY fastai/fastai_Time_Series_Forecasting_of_Stock_Prices_utils.py .
COPY fastai/fastai_Time_Series_Forecasting_of_Stock_Prices.API.md .
COPY fastai/fastai_Time_Series_Forecasting_of_Stock_Prices.API.ipynb .
COPY fastai/fastai_Time_Series_Forecasting_of_Stock_Prices.example.md .
COPY fastai/fastai_Time_Series_Forecasting_of_Stock_Prices.example.ipynb .
COPY fastai/fastai_Time_Series_Forecasting_of_Stock_Prices_Sentiment.example.ipynb .

COPY README.md .

RUN mkdir -p /app/data /app/outputs /app/notebooks

EXPOSE 8888

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/api/status || exit 1

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]