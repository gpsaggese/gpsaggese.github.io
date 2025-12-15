FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir numpy==1.23.5
RUN pip install --no-cache-dir pandas==2.0.3
RUN pip install --no-cache-dir scipy==1.11.4
RUN pip install --no-cache-dir scikit-learn==1.3.2
RUN pip install --no-cache-dir tensorflow==2.15.0
RUN pip install --no-cache-dir keras==2.15.0
RUN pip install --no-cache-dir autokeras==1.1.0
RUN pip install --no-cache-dir matplotlib==3.7.2
RUN pip install --no-cache-dir seaborn==0.12.2
RUN pip install --no-cache-dir jupyter==1.0.0
RUN pip install --no-cache-dir notebook==7.0.6
RUN pip install --no-cache-dir statsmodels==0.14.0
RUN pip install --no-cache-dir python-dateutil==2.8.2

EXPOSE 8888
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

CMD ["/bin/bash"]
