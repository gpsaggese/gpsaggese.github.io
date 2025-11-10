#!/bin/bash
echo "# Airflow"
docker images | grep airflow
echo "# Postgres"
docker images | grep postgres
echo "# Redis"
docker images | grep redis
