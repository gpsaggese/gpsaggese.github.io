#!/bin/bash -xe

./docker_build.sh
./docker_cmd.sh '/data/run_psql_server.sh; /data/tutorial_university/init_small_psql_university_db.sh'
