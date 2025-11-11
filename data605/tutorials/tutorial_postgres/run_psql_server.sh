#!/bin/bash -xe

service --status-all
/etc/init.d/postgresql start
service --status-all
