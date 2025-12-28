# Simple CPU-only image for running the SBERT sentiment project
FROM python:3.11-slim

# Recommended Python settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Workdir inside the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Jupyter will run on this port
EXPOSE 8888

# Default command: launch Jupyter Lab in the project directory
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]