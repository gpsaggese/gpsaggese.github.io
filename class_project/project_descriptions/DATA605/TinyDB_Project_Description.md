# Description

TinyDB is a lightweight, pure-Python document database that stores schema-less JSON
documents in a single file, with no server to install. It solves the problem of
persisting and querying small collections of records in a prototype, where a full
database is too heavy. It is worth a 60-minute tutorial because the full API fits in
one hour, and it shows clearly what a document store gives up compared with an
indexed database.

## Technologies Used

TinyDB

- Tables of schema-less JSON documents, with `insert`, `insert_multiple`, `update`,
  `upsert`, and `remove`
- The `Query()` object and `where` to compose conditions with `&` and `|`
- Pluggable storage backends: `JSONStorage`, `MemoryStorage`, and custom storages
- Middlewares such as `CachingMiddleware` to speed up the writes

# Tutorial

- Implement the tutorial "Learn TinyDB in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of `.claude/skills/tutorial_in_60_mins.rules.md`
  - No earlier tutorial or project uses TinyDB, so read the closest database work
  - Read `data605/tutorials/tutorial_mongodb/` for the document store view
  - Read the `README.md` of the Spring2025 SQLAlchemy project for the relational
    view, and reuse what is good
    - `class_project/data605/Spring2025/projects/TutorTask121_Spring2025_Real-Time_Bitcoin_Analysis_with_SQLAlchemy/`
- Create `tutorials/TinyDB/`, since it does not exist yet
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare briefly with SQLite and MongoDB from the query and storage point of view,
  e.g., indexes, concurrency, and file size
- Deliverables:
  - `tinydb_utils.py`
  - `tinydb.API.ipynb`
  - `tinydb.example.ipynb`

# Project

## Project 1: Personal Book Collection Manager

- **Difficulty**: 1 (Easy)
- **Project Objective**: Build a book collection manager on TinyDB that supports CRUD
  operations and queries, and measure how its speed scales with the size of the
  collection
- **Dataset Suggestions**: [Goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k)
- **Tasks**:
  - **Load the Data**: Read `books.csv` with pandas, and insert the 10,000 books into
    a TinyDB table with `insert_multiple`
  - **Define the Queries**: List eight questions the collection must answer, e.g.,
    the books of an author, or the books after 2000 with a rating above 4.2, and
    write each as a `Query()`
  - **Implement CRUD**: Write add, get, update, and delete functions with `insert`,
    `search`, `update`, and `remove`, and check each with an assertion
  - **Query the Collection**: Answer the eight queries with `where` and the `&` and
    `|` operators, and check the counts against pandas
  - **Benchmark the Storage**: Measure the insert and query time for 1,000, 5,000,
    and 10,000 documents with `JSONStorage` and with `CachingMiddleware`
  - **Visualize the Collection**: Plot the distribution of the publication years and
    the ten authors with the most books
- **Bonus Ideas (Optional)**: Recommend similar books from the shared tags; add a
  command-line interface; export and import the collection as CSV

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Load the Data
  - Result: `tutorials/TinyDB/` container running, and a table with the 10,000 book
    documents
- Milestone 2: API notebook
  - Project tasks: Implement CRUD, Query the Collection
  - Result: `tinydb.API.ipynb` covering `TinyDB`, tables, `insert_multiple`, `Query`,
    `update`, `upsert`, `remove`, the storages, and `CachingMiddleware`
- Milestone 3: Example notebook
  - Project tasks: Define the Queries, Implement CRUD, Query the Collection,
    Benchmark the Storage, Visualize the Collection
  - Result: `tinydb.example.ipynb` running end to end

## Project 2: Sensor Readings Store vs. SQLite

- **Difficulty**: 2 (Medium)
- **Project Objective**: Store hourly air quality readings as documents, and find
  where TinyDB stops being competitive with SQLite
- **Dataset Suggestions**:
  [UCI - Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality)
- **Tasks**:
  - **Load the Readings**: Parse the timestamps, replace the `-200` markers with
    missing values, and insert one document per hour
  - **Query a Range**: Retrieve the readings of a date range with a CO level above a
    threshold, using `Query().test()` on the ISO timestamps
  - **Aggregate the Readings**: Compute the daily mean CO by looping over the
    `search` results, and check the values against pandas
  - **Compare with SQLite**: Load the same data into SQLite, and time three queries
    on the original data and on a copy with ten times more rows
  - **Report the Trade-Offs**: Plot the query time vs. the number of documents for
    both stores
- **Bonus Ideas (Optional)**: Add a manual index, i.e., a dictionary from date to
  document id, and measure the gain

## Project 3: Custom Storage and Concurrent Writers

- **Difficulty**: 3 (Hard)
- **Project Objective**: Extend TinyDB with a compressed storage, and measure its
  robustness with concurrent writers and the trade-off between size and speed
- **Dataset Suggestions**:
  [Open Library Search API](https://openlibrary.org/developers/api)
- **Tasks**:
  - **Fetch the Data**: Download 2,000 works from the search API, and cache the JSON
    responses on disk
  - **Write a Custom Storage**: Subclass `Storage` to save the database as
    gzip-compressed JSON, implementing `read` and `write`
  - **Test Concurrent Writers**: Run four processes that insert documents at the same
    time, and count the lost or corrupted documents
  - **Add a File Lock**: Protect the writes with a `filelock` lock, and repeat the
    concurrency test
  - **Compare the Storages**: Report the file size, write time, and read time of
    `JSONStorage`, the gzip storage, and `MemoryStorage`
  - **Report the Trade-Offs**: Plot the size vs. the time for each storage, and
    summarize when TinyDB is not a good choice
- **Bonus Ideas (Optional)**: Add an encrypted storage with the `cryptography`
  package
