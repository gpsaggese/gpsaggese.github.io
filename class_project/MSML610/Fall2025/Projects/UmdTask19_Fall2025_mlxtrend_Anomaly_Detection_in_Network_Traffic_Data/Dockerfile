# --- Base image ---
FROM python:3.9-slim

# --- Set working directory ---
WORKDIR /workspace

# --- Copy project files ---
COPY . /workspace

# --- Install dependencies ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Set default command ---
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
