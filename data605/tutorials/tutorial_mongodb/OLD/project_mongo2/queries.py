#### Since the collection name must be separately specified, we will use "tuples" to simplify running the queries
### If the tuple only has two entries, then the first entry is the name of the collection, and the second entry is the 'find' query
### If the tuple has three entries, then the third entry is the "projection" array
collection_and_queries = [None] * 13

### 0. Return all the entries in accounts collection
collection_and_queries[0] = ('transactions', {})

### 1. Find all information for the customer with username 'fmiller'
collection_and_queries[1] = ('customers', {})

### 2. For all customers with first name 'Natalie', return their username, name, and address
### Use 'regex' functionality to do the matching
collection_and_queries[2] = ('customers', {}, {})

### 3. Find all accounts with a 'products' array containing 'Commodity' -- return the '_id' and 'account_id'
collection_and_queries[3] = ('accounts', {}, {})

### 4. Find all accounts with either limit <= 9000 or products array exactly ["Commodity", "InvestmentStock"] in that order -- return the entire accounts information
### Use "$or" to do a disjunction
collection_and_queries[4] = ('accounts', {})

### 5. Find all accounts with limit <= 9000 AND products array exactly ["Commodity", "InvestmentStock"] in that order -- return the entire accounts information
collection_and_queries[5] = ('accounts', {})

### 6. Find all accounts where the second entry in the products array is 'Brokerage' -- return the entire accounts information
collection_and_queries[6] = ('accounts', {})

### 7. On the customers collection, use aggregation and grouping to find the number of customers born in each month
### The output will contain documents like: {'_id': 7, 'totalCount': 42} 
### Use '$month' function to operate on the dates, and use '$sum' aggregate to do the counting
### https://database.guide/mongodb-month/
collection_and_queries[7] = ('customers', [])

### 8. Modify the above query to only count the customers whose name starts with any letter between 'A' and 'G' (inclusive). 
### The output will contain documents like: {'_id': 2, 'totalCount': 17}
###
### Use '$match' along with '$group' as above.
collection_and_queries[8] = ('customers', [])

### 9. In the 'transactions' collection, all transactions are inside a single array, making it difficult to operate on them. 
### However, we can use 'unwind' to create a separate document for each of the transactions. 
### The query below shows this for a single account to see how this work: 
###             [{'$match': {'account_id': 443178}}, {'$unwind': '$transactions'}]
###
### Use 'unwind' as above to create a list of documents where each document is simply: _id, account_id, symbol (inside the transaction), transaction_code ("sell" or "buy"), and amount (sold/bought)
### Restrict this to accounts with fewer than 10 transactions
###
### One of the outputs:
### {'_id': ObjectId('5ca4bbc1a2dd94ee58161cd5'), 'account_id': 463155, 'transactions': {'amount': 6691, 'symbol': 'amd', 'transaction_code': 'buy'}}
###
collection_and_queries[9] = ('transactions', [])

### 10. Use the result of the above query to compute the total number shares sold or bought for each symbol across the entire collection of accounts
### However, DO NOT restrict it to only accounts with < 10 transactions.
### Use '$sort' to sort the outputs in descending order by the total count of shares
### First few outputs look like this:
###       {'_id': 'adbe', 'totalCount': 27463715}
###       {'_id': 'ebay', 'totalCount': 27232371}
###       {'_id': 'crm', 'totalCount': 27099929}
###       {'_id': 'goog', 'totalCount': 27029894}
###       {'_id': 'nvda', 'totalCount': 26108705}
collection_and_queries[10] = ('transactions', [])

### 11. Use $lookup to do a "join" between customers and accounts to find, for each customer the number of accounts they have with 'InvestmentFund' as a product (i.e., the number of their accounts where 'InvestmentFund' is in the 'products' array).
### Sort the final output by 'username' in the ascending order
### First few outputs
###        {'_id': 'abrown', 'totalCount': 1}
###        {'_id': 'alexandra72', 'totalCount': 2}
###        {'_id': 'alexsanders', 'totalCount': 2}
###        {'_id': 'allenhubbard', 'totalCount': 2}
###        {'_id': 'alvarezdavid', 'totalCount': 3}

collection_and_queries[11] = ('customers', [])

### 12. We want to find all accounts that have exactly 3 products and <= 10 transactions associated with them.
### Use '$lookup' and '$group' to do so by joining accounts and transactions. This would be a multi-stage pipeline, possibly with multiple groups and matches.
###
### Output all the information for the matching accounts as shown below.
### {'_id': ObjectId('5ca4bbc7a2dd94ee58162576'),
### 'account_id': 154391,
### 'products': ['Brokerage', 'Commodity', 'InvestmentStock'],
### 'transaction_count': 4}
###
### Use unwind and addFields to add the top-level field 'transaction_count' to accounts
### Sort the final output by account_id
collection_and_queries[12] = ('accounts', [])
