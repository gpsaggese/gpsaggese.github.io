# Use a lightweight Python base image
FROM python:3.11-slim

# Set a working directory inside the container
WORKDIR /app

# Copy only the requirements file first (better caching)
COPY requirements.txt .

# Install system packages needed for pandas, numpy, sklearn, matplotlib
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Make outputs directory (in case it doesn’t exist)
RUN mkdir -p outputs

# Command to run your pipeline
CMD ["python", "-m", "src.main"]
