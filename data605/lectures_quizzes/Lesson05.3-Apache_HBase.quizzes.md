1. What does HBase stand for?
   - A) Hadoop Basic
   - B) High Base
   - C) High Batch
   - D) **Hadoop DataBase**
   - E) Hadoop Binaries

2. Which storage solution is HBase built on?
   - A) Google File System
   - B) Microsoft File System
   - C) **Hadoop File System (HDFS)**
   - D) Amazon S3
   - E) Flat File Storage

3. HBase is predominantly what type of database?
   - A) Document-oriented
   - B) Relational
   - C) **Column-oriented**
   - D) Key-value
   - E) Graph

4. What kind of workloads does HBase mainly support?
   - A) OLAP
   - B) Data Warehousing
   - C) **OLTP**
   - D) Batch Processing
   - E) Big Data Analysis

5. Which of the following is a characteristic of HBase?
   - A) Fixed schema
   - B) **Data versioning**
   - C) Centralized configuration
   - D) Transactional consistency
   - E) Low fault tolerance

6. What does the term "cell" in HBase refer to?
   - A) A unit of data storage
   - B) A row identifier
   - C) The database name
   - D) **A unique data point identified by (table, row, family:column)**
   - E) A backup point in the database

7. How does HBase ensure atomicity at the row level?
   - A) By backing up data
   - B) **By updating entire rows at once or not at all**
   - C) By using complex transactions
   - D) By splitting rows into microtransactions
   - E) By holding multiple session locks

8. What is the purpose of a Bloom filter in HBase?
   - A) To index all data entries
   - B) **To check row or column existence without querying**
   - C) To compress data
   - D) To partition tables across servers
   - E) To maintain data backups

9. Why is a Write-Ahead Log (WAL) important for HBase?
   - A) It speeds up read operations
   - B) It stores data permanently
   - C) **It ensures atomicity and durability against failures**
   - D) It allows real-time analytics
   - E) It implements complex queries

10. How does HBase handle variable-length data?
    - A) **Uses pointers to the actual data location**
    - B) Stores fixed-length records
    - C) Requires data to be padded
    - D) Stores it in relational tables
    - E) Uses designated formats like JSON only

11. Which of the following is true about adding columns versus adding column families in HBase?
    - A) **Columns can be added at runtime, column families cannot**
    - B) Both can be added at will
    - C) Both require table recreation
    - D) Column families can be easily removed
    - E) Columns require complex indexing

12. How does HBase support data compression?
    - A) It compresses data using SQL methods
    - B) It does not support compression
    - C) **It compresses and decompresses data on-the-fly**
    - D) It compresses data manually
    - E) It only compresses during data retrieval

13. In the HBase data model, what does a timestamp provide?
    - A) Row prioritization
    - B) **Version control for data**
    - C) Row assignment
    - D) Row identification
    - E) Automatic backups

14. What does OLAP stand for, and how does it relate to HBase?
    - A) Online Local Analysis
    - B) **On-Line Analytical Processing**
    - C) Overload Application
    - D) Offline Analytical Processing
    - E) Operational Long-Access Processing

15. In HBase, what is a 'region'?
    - A) A data type
    - B) **A chunk of rows within a table**
    - C) A table schema
    - D) A database server
    - E) A user access level

16. What is a primary benefit of HBase's in-memory tables?
    - A) Higher permanent storage capacity
    - B) They are centralized in one location
    - C) **Faster read and write operations**
    - D) Elimination of disk dependency
    - E) Reduced storage costs

17. Which function allows HBase to retrieve a specific value?
    - A) update()
    - B) **get()**
    - C) fetch()
    - D) load()
    - E) access()

18. What significantly differentiates HBase from traditional relational databases?
    - A) HBase supports multiple users simultaneously
    - B) HBase allows predefined column types
    - C) HBase has a schema-on-read approach
    - D) **HBase uses a row as a mini-database with varied structures**
    - E) HBase offers compartmentalized security frameworks

19. When would you choose to use HBase?
    - A) For small-scale applications
    - B) With less than 5 nodes
    - C) For fixed-data models
    - D) **For large databases with high write and read performance**
    - E) For transaction-heavy databases

20. What storage mechanism does HBase primarily utilize?
    - A) Flat file storage
    - B) **Hadoop's distributed storage system**
    - C) Local disk storage
    - D) Cloud-based file systems
    - E) Relational storage files