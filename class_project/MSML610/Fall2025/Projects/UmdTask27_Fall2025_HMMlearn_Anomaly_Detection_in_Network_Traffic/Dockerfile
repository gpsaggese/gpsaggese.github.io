FROM python:3.8

RUN mkdir /workdir

WORKDIR /workdir
RUN mkdir /workdir/content

COPY . /workdir

RUN pip install --no-cache-dir -r requirements.txt
RUN curl -o ./content/UNSW_NB15_training-set.csv -O https://raw.githubusercontent.com/mvetlu/610_content/refs/heads/main/UNSW_NB15_training-set.csv
RUN curl -o ./content/42.csv -O https://raw.githubusercontent.com/mvetlu/610_content/refs/heads/main/42.csv

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]