#!/bin/bash -xe
PYSPARK_PYTHON=/usr/bin/python3
PYSPARK_DRIVER_PYTHON="jupyter"
PYSPARK_DRIVER_PYTHON_OPTS="notebook --allow-root --no-browser --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --NotebookApp.password=''"
$SPARKHOME/bin/pyspark
