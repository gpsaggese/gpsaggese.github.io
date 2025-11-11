FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1) upgrade pip first
RUN python -m pip install --upgrade pip

# 2) then install your deps
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888

CMD jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''

