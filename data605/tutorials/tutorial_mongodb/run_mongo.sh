#!/bin/bash -x
sudo systemctl enable mongod
sudo systemctl start mongod
sudo systemctl status mongod

#sudo /usr/bin/mongod --config /etc/mongod.conf 2>&1 | tee ~/mongo.log &

# Test connection.
mongosh --eval "db"
#mongosh --eval "var status = db.serverStatus(); status.connections"
