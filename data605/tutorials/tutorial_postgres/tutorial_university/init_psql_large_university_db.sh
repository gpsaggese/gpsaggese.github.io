#!/bin/bash -xe

createdb large_university
psql --command "\i /data/tutorial_university/DDL.sql;" large_university
psql --command "\i /data/tutorial_university/largeRelationsInsertFile.sql;" large_university
