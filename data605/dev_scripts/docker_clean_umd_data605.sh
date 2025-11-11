#!/bin/bash -xe

TAG=umd_data605

docker container ls | \grep $TAG | awk '{print $1}' | xargs docker container rm --force

docker volume ls | \grep $TAG | awk '{print $2}' | xargs docker volume rm

docker network ls | \grep $TAG | awk '{ print $1 }' | xargs docker network rm
