FROM python:3.10

# Set the working directory inside the container
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Default command: open a shell (scripts will override this)
CMD ["bash"]
