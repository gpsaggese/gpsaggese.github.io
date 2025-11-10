FROM python:3.11-slim

WORKDIR /app
COPY utils_data_io.py utils_post_processing.py ./
RUN pip install --no-cache-dir pandas==2.2.2 scikit-learn==1.5.2 jupyter==1.0.0

# Mount your data at runtime: -v $PWD:/app
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
