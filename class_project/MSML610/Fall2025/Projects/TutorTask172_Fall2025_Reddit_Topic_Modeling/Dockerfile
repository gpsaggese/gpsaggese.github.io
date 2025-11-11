FROM python:3.10-slim

RUN apt-get update && apt-get install -y git wget

RUN pip install fasttext scikit-learn nltk matplotlib seaborn tqdm transformers jupyter

WORKDIR /app
COPY . /app

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
