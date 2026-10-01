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
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
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

## Project 1 (Fall2026): Insider Trading Filings Store vs. SQLite

- **Project Objective**: Store SEC Form 4 insider filings as nested documents, and
  find where TinyDB stops being competitive with SQLite on insider trade queries
- **Dataset Suggestions**:
  [SEC Insider Transactions Data Sets](https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets),
  using the file of one quarter, e.g., `2024q1_form345.zip`; the SEC asks for a
  `User-Agent` header with contact information on downloads
- **Tasks**:
  - **Build the Documents**: Join `SUBMISSION.tsv`, `REPORTINGOWNER.tsv`, and
    `NONDERIV_TRANS.tsv` on `ACCESSION_NUMBER` with pandas, and insert one nested
    document per Form 4 filing with the issuer, the owners, and the trades
  - **Make the Load Idempotent**: Reload the file with `upsert` on the accession
    number, check that the document count does not change, and `remove` the filings
    without trades
  - **Query the Trades**: Retrieve the filings with an open-market purchase
    (`TRANS_CODE` `P`) above 100,000 USD by an officer or director, using
    `Query().test()` and `any` on the nested trades
  - **Aggregate the Trades**: Compute the weekly value of insider buys and sells per
    issuer ticker by looping over the `search` results, and check the totals against
    pandas
  - **Compare with SQLite**: Load the same data into three SQLite tables, and time
    three queries on 5,000, 20,000, and 50,000 filings, for TinyDB with
    `JSONStorage` and with `MemoryStorage`, and for SQLite
  - **Report the Trade-Offs**: Plot the query time vs. the number of documents for
    the three setups, and report the file sizes
- **Bonus Ideas (Optional)**: Add a manual index, i.e., a dictionary from issuer
  ticker to document id, and measure the gain

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Build the Documents
  - Result: project dir created and container running, and a table with the number
    of nested Form 4 documents, owners, and trades built from the quarter file
- Milestone 2: API notebook
  - Project tasks: Make the Load Idempotent, Query the Trades, Aggregate the Trades
  - Result: `tinydb.API.ipynb` covering `TinyDB`, `insert`, `upsert`, `remove`,
    `Query().test()`, `any`, `search`, `JSONStorage`, and `MemoryStorage`
- Milestone 3: Example notebook
  - Project tasks: Make the Load Idempotent, Query the Trades, Aggregate the
    Trades, Compare with SQLite, Report the Trade-Offs
  - Result: `tinydb.example.ipynb` running end to end

## Project 2: Macro Indicator Watchlist with Alerts

- **Project Objective**: Build a watchlist of macroeconomic indicators on TinyDB that
  supports CRUD operations and alert rules, and measure how its speed scales with the
  number of observations
- **Dataset Suggestions**:
  [FRED Economic Data](https://fred.stlouisfed.org/), using ten series since 2010,
  e.g., `UNRATE`, `CPIAUCSL`, `FEDFUNDS`, `PAYEMS`, `INDPRO`, `UMCSENT`, `ICSA`,
  `DGS10`, `DGS2`, and `T10Y2Y`, downloaded one by one from
  `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>`
- **Tasks**:
  - **Load the Series**: Read the ten CSV files with pandas, drop the missing values,
    and insert the 15,000 observations as documents with `series_id`, `date`, and
    `value` into a TinyDB table with `insert_multiple`
  - **Define the Alert Rules**: List eight rules the watchlist must check, e.g., the
    unemployment rate above 6%, or the 10-year minus 2-year spread below 0, and store
    them in an `alerts` table
  - **Implement CRUD**: Write add, get, update, upsert, and delete functions with
    `insert`, `search`, `update`, `upsert`, and `remove`, e.g., `upsert` a revised
    observation by `series_id` and `date`, and check each with an assertion
  - **Query the Series**: Evaluate the eight rules with `where` and the `&` and `|`
    operators, and check the counts of the triggering dates against pandas
  - **Benchmark the Storage**: Measure the insert and query time for 1,000, 5,000, and
    15,000 documents with `JSONStorage` and with `CachingMiddleware`
  - **Visualize the Alerts**: Plot the spread and the unemployment rate with the
    triggering dates shaded
- **Bonus Ideas (Optional)**: Compare the dates with an inverted spread against the
  recession indicator `USREC`; add a command-line interface; export the alerts as CSV

## Project 3: Custom Storage and Concurrent Writers for Crypto Candles

- **Project Objective**: Extend TinyDB with a compressed storage, and measure its
  robustness with concurrent writers and the trade-off between size and speed
- **Dataset Suggestions**:
  [Binance Public Data](https://github.com/binance/binance-public-data), using the
  monthly spot klines with the `1h` interval of `BTCUSDT`, `ETHUSDT`, `BNBUSDT`, and
  `SOLUSDT` for 2024, from `https://data.binance.vision/`
- **Tasks**:
  - **Fetch the Data**: Download the 48 monthly zip files, cache them on disk, and
    convert each row into a candle document with an ISO `open_time` and the prices
  - **Write a Custom Storage**: Subclass `Storage` to save the database as
    gzip-compressed JSON, implementing `read` and `write`
  - **Test Concurrent Writers**: Run four processes, one per symbol, that insert the
    monthly batches with `insert_multiple` into the same file, and count the lost or
    corrupted documents against the row count of the CSV files
  - **Add a File Lock**: Protect the writes with a `filelock` lock, and repeat the
    concurrency test
  - **Compare the Storages**: Report the file size, write time, read time, and the time
    of a date range query for one symbol, for `JSONStorage`, the gzip storage, and
    `MemoryStorage`
  - **Report the Trade-Offs**: Plot the size vs. the time for each storage, and
    summarize when TinyDB is not a good choice
- **Bonus Ideas (Optional)**: Add an encrypted storage with the `cryptography`
  package
