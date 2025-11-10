#!/usr/bin/env bash
(mongod --replSet book --dbpath ./mongo1 --port 27011 2>&1 | tee mongo_server1.log) &
(mongod --replSet book --dbpath ./mongo2 --port 27012 2>&1 | tee mongo_server2.log) &
(mongod --replSet book --dbpath ./mongo2 --port 27013 2>&1 | tee mongo_server3.log) &
