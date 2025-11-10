#!/bin/bash -xe

createdb university
# Create the schema.
psql --command "\i /data/tutorial_university/DDL.sql;" university
# Insert some data in the DB.
psql --command "\i /data/tutorial_university/smallRelationsInsertFile.sql;" university
