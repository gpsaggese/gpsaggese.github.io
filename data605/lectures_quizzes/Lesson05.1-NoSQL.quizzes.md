1. What does the term "NoSQL" originally express?
   - A) Not Only SQL
   - B) No SQL
   - C) New SQL
   - D) None of the above
   - **E) Both A and B**

2. Which of the following describes a trade-off that NoSQL databases provide compared to relational databases?
   - A) Strong consistency
   - B) Schema enforcement
   - C) Rich query ability
   - **D) Schema flexibility**
   - E) Single data model

3. Why might a developer utilize multiple database technologies in a single project?
   - A) To reduce complexity
   - B) To ensure all data is normalized
   - **C) To leverage the strengths of each database type**
   - D) To increase costs
   - E) To avoid coding

4. What is a crucial drawback of relational databases mentioned in the material?
   - A) High scalability
   - **B) Application-DB impedance mismatch**
   - C) Lack of flexibility
   - D) Complex query languages
   - E) Weak data integrity

5. Which solution can address the application-DB impedance mismatch?
   - A) Normalization
   - B) Denormalization
   - C) Using only flat tables
   - **D) Object-relational mapping (ORM)**
   - E) Using NoSQL databases exclusively

6. In regards to schema flexibility, what is a common problem with relational databases?
   - A) They are too flexible
   - **B) Data may not fit into a rigid schema**
   - C) They do not support multiple schemas
   - D) They ensure strong consistency at all times
   - E) They use only one data type

7. What does ACID stand for in the context of databases?
   - A) Atomic, Consistent, Integrity, Durable
   - **B) Atomicity, Consistency, Isolation, Durability**
   - C) Automatic, Consistent, Integrative, Durable
   - D) Atomic, Concurrent, Integrated, Durable
   - E) None of the above

8. What does the CAP theorem state regarding distributed databases?
   - A) They can achieve all three: consistency, availability, and partition tolerance
   - **B) They can have at most two of the three properties**
   - C) They are inherently consistent
   - D) They can only be available or consistent
   - E) They operate best without partitions

9. Which option represents a common trade-off under the CAP theorem?
   - A) More consistency for less availability
   - **B) Sacrificing consistency for availability**
   - C) Consistency without partition tolerance
   - D) High availability with no durability
   - E) None of the above

10. In primary-secondary replication, which statement is true?
    - **A) The application always communicates with the primary server.**
    - B) All replicas are independent of each other.
    - C) Any server can update without a designated primary.
    - D) It guarantees no single point of failure.
    - E) It allows direct updates from secondary servers.

11. What characterizes asynchronous replication?
    - A) Immediate consistency across all nodes
    - B) Each replica updates before the transaction completes
    - **C) The primary node updates replicas after the transaction completes**
    - D) It requires a two-phase commit to operate
    - E) All updates are synchronous

12. In the NoSQL context, what is a common characteristic of schema-less databases?
    - **A) They allow for varying data structures within the same dataset.**
    - B) They require predefined schemas.
    - C) They must adhere to strict rows and columns.
    - D) They cannot accommodate nested data.
    - E) They are incompatible with ACID properties.

13. What is a benefit of table denormalization in databases?
    - **A) Faster read operations by reducing joins.**
    - B) Ensures consistent data across all replicas.
    - C) Guarantees ACID compliance.
    - D) Maintains complex relationships without issues.
    - E) Reduces data redundancy.

14. What problem arises from the use of strict locking mechanisms in databases?
    - A) Increased read performance
    - B) Enhanced data integrity
    - C) Growth of available storage
    - **D) Increased latency in updates**
    - E) No impact on performance

15. How does MongoDB maintain schema flexibility?
    - A) By enforcing strict schemas
    - **B) By not enforcing schemas at the database level**
    - C) Through data normalization
    - D) Using only numerical data types
    - E) By requiring foreign key relationships

16. What is the major disadvantage of using synchronous replication?
    - **A) It can result in a single point of failure.**
    - B) It allows faster updates to every node simultaneously.
    - C) It does not require a primary server.
    - D) It ensures immediate data consistency.
    - E) It is simpler to implement than asynchronous replication.

17. When is eventual consistency acceptable in a database application?
    - A) In banking applications
    - B) In all database scenarios
    - **C) In social networking applications where immediate consistency is not critical.**
    - D) In applications requiring high transactional volume
    - E) In systems that require strict ACID properties

18. Which of the following is a solution for scalability issues in relational databases?
    - A) Maintain strict consistency for all operations
    - **B) Relax consistency for higher performance**
    - C) Use multiple normalization processes
    - D) Ensure all data is kept on a single server
    - E) Differentiate data types strictly

19. Which statement is true about NoSQL databases?
    - **A) They prioritize availability and partition tolerance over consistency in many scenarios.**
    - B) They always offer strong consistency.
    - C) They require extensive schema design.
    - D) They are not suitable for large-scale applications.
    - E) They strictly avoid data denormalization.

20. The use of which replication model generally avoids a single point of failure?
    - **A) Update-anywhere replication**
    - B) Primary-secondary replication
    - C) Two-phase commit
    - D) Quorum-based replication
    - E) Synchronous replication