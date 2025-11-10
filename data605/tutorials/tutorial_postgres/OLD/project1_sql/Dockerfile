FROM ubuntu:20.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt install -y python3-pip python3-dev postgresql postgresql-contrib libpq-dev jupyter-notebook
RUN pip3 install jupyter ipython-sql psycopg2

ADD smallRelationsInsertFile.sql largeRelationsInsertFile.sql DDL.sql /datatemp/

EXPOSE 8888
EXPOSE 5432

USER postgres

RUN /etc/init.d/postgresql start && \
    createdb university && \
    psql --command "\i /datatemp/DDL.sql;" university && \
    psql --command "\i /datatemp/smallRelationsInsertFile.sql;" university && \
    psql --command "\i /datatemp/smallRelationsInsertFile.sql;" university
