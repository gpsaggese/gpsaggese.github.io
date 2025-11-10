FROM ubuntu:focal
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt install -y python3-pip python3-dev \
    postgresql postgresql-contrib libpq-dev \
    jupyter-notebook vim \
    openjdk-8-jdk
RUN apt install -y sudo curl systemctl gnupg
RUN pip3 install jupyter ipython-sql psycopg2 flask flask-restful flask_cors pymongo

RUN pip3 install nbconvert --upgrade

#ADD Assignment-0/smallRelationsInsertFile.sql Assignment-0/largeRelationsInsertFile.sql Assignment-0/DDL.sql Assignment-0/postgresql.conf /datatemp/
#ADD Assignment-5/sample_analytics/customers.json Assignment-5/sample_analytics/accounts.json Assignment-5/sample_analytics/transactions.json /datatemp/
#ADD Assignment-3/populate-se.sql /datatemp/
#ADD Assignment-4/log4j2.properties /datatemp/
#ADD Assignment-4/spark-3.3.0-bin-hadoop3/ /spark/

EXPOSE 8888
EXPOSE 5432

RUN cp /datatemp/postgresql.conf /etc/postgresql/12/main/postgresql.conf
RUN cp /datatemp/log4j2.properties /spark/conf

USER postgres

RUN /etc/init.d/postgresql start &&\
    createdb university &&\
    psql --command "\i /datatemp/DDL.sql;" university &&\
    psql --command "\i /datatemp/smallRelationsInsertFile.sql;" university &&\
    psql --command "alter user postgres with password 'postgres';" university &&\
    psql --command "create user root;" university &&\
    psql --command "alter user root with password 'root';" university &&\
    psql --command "alter user root with superuser;" &&\
    createdb stackexchange &&\
    psql --command "\i /datatemp/populate-se.sql" stackexchange &&\
    /etc/init.d/postgresql stop

USER root

RUN curl -fsSL https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add - && \
    echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list &&\
    apt update &&\
    ## curl -O http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb &&\
    ## dpkg -i ./libssl1.1_1.1.1f-1ubuntu2_amd64.deb &&\
    apt install -y mongodb-org

RUN systemctl enable mongod
RUN (/usr/bin/mongod --config /etc/mongod.conf &) && \
    mongoimport --db "analytics" --collection "customers" /datatemp/customers.json && \
    mongoimport --db "analytics" --collection "customers" /datatemp/accounts.json && \
    mongoimport --db "analytics" --collection "customers" /datatemp/transactions.json

ENV SPARKHOME=/spark/

ENTRYPOINT service postgresql start && \
        (/usr/bin/mongod --config /etc/mongod.conf &) && \
        (jupyter-notebook --port=8888 --allow-root --no-browser --ip=0.0.0.0 --NotebookApp.notebook_dir='/data' --NotebookApp.token='' 2>/dev/null &) && \
        /bin/bash
