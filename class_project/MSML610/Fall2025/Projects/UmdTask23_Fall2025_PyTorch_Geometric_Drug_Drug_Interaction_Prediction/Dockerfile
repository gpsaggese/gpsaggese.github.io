FROM python:3.10-slim

WORKDIR /workspace

COPY . /workspace

RUN pip install --no-cache-dir \
    jupyter notebook pandas numpy scikit-learn

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
