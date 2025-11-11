- We want to spawn 3 different Mongo servers
- Go inside the container
  ```
  > docker_bash.sh
  ```
- Create 3 data directories
  ```
  docker> mkdir mongo1 mongo2 mongo3
  ```

- Start Mongo server
  ```
  docker> (mongod --replSet book --dbpath ./mongo1 --port 27011 2>&1 | tee mongo_server1.log) &
  docker> (mongod --replSet book --dbpath ./mongo2 --port 27012 2>&1 | tee mongo_server2.log) &
  docker> (mongod --replSet book --dbpath ./mongo2 --port 27013 2>&1 | tee mongo_server3.log) &
  ```
