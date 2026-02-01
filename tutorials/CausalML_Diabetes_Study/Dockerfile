FROM python:3.11-slim
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ls -la /root/.cargo/bin/ || echo "cargo bin dir not found" && \
    ls -la /root/.local/bin/ || echo "local bin dir not found"
ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN which uv && uv pip install --system -r requirements.txt
COPY . /app
EXPOSE 8888
CMD ["sh","-lc","jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*'"]
