# Assignment 2: MongoDB
- Start container and then MongoDB server

- Load database in MongoDB
  ```
  docker> /data/project_mongo2/init_mongo_customer_db.sh
  + SRC_DIR=/data/project_mongo2/sample_analytics
  + mongoimport --db analytics --collection customers /data/project_mongo2/sample_analytics/customers.json
  2022-12-11T09:18:41.072+0000    connected to: mongodb://localhost/
  2022-12-11T09:18:41.138+0000    500 document(s) imported successfully. 0 document(s) failed to import.
  + mongoimport --db analytics --collection customers /data/project_mongo2/sample_analytics/accounts.json
  2022-12-11T09:18:41.159+0000    connected to: mongodb://localhost/
  2022-12-11T09:18:41.201+0000    1746 document(s) imported successfully. 0 document(s) failed to import.
  + mongoimport --db analytics --collection customers /data/project_mongo2/sample_analytics/transactions.json
  2022-12-11T09:18:41.217+0000    connected to: mongodb://localhost/
  2022-12-11T09:18:42.013+0000    1746 document(s) imported successfully. 0 document(s) failed to import.
  ```

- Test that some queries from Python work
  ```
  docker> python3 MongoDBTesting.py
  ```


MongoDB is a database system, just like PostgreSQL, but has a different data
model (JSON) and a different query language.

### Installation
The provided Dockerfile (in the top-level directory) install MongoDB, and
`pymongo`, the driver to use MongoDB from within Python.

**NOTE: Queries written using pymongo look slightly different from queries
written directly in the `mongo` shell.**

### Dataset
We are using one of the test datasets provided by MongoDB itself, called
`sample-analytics`. It has three collections:
1. `customers`: Each document corresponds to a customer, and has basic
   information about the customer, as well as some information about the benefits
   they have. It also contains an array of `accounts` for that customer.
2. `accounts`: Each document contains information for an account, mainly which
   types of products it allows using.
3. `transactions`: Each document contains the transactions for a given account --
   each transaction being a stock trade.

You can see the raw JSONs (in MongoDB export format) in `sample_analytics`
directory.

The collections may already be loaded for you, but if not, you can do the
following (from the `Assignment-5` directory):
- Start the MongoDB server: `systemctl start mongod.service`
- Load customers: `mongoimport --db "analytics" --collection "customers"
  sample_analytics/customers.json`
- Load accounts: `mongoimport --db "analytics" --collection "accounts"
  sample_analytics/accounts.json`
- Load transactions: `mongoimport --db "analytics" --collection "transactions"
  sample_analytics/transactions.json`

### Assignment
As with the SQL assignment, the actual tasks are provided in the `queries.py`
file. You should fill out your answers in that file. You can use `python3
MongoDBTesting.py` to run all the queries and see the results. 

## Submission
Submit the `functions.py` file for Apache Spark part, and `queries.py` for MongoDB part.
