#/bin/bash -xe
/data/run_psql_server.sh
/data/tutorial_university/init_small_psql_university_db.sh
/data/run_jupyter.sh
