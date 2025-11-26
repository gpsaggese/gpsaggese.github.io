FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn clip_embed_API:app --host 0.0.0.0 --port 8000 & streamlit run clip_embed.streamlit.py --server.port 8501 --server.address 0.0.0.0"]
