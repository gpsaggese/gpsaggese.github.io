# PostgreSQL Tutorial

## Overview

PostgreSQL is a full-featured, open-source relational database system. It is
used for several tutorials and for the class project in `umd_data605`.

In this tutorial you will:

- Start a pre-configured Docker container that includes PostgreSQL
- Start the PostgreSQL server inside the container
- Create and populate example databases
- Connect to PostgreSQL from the container shell, from your laptop, and from a
  Jupyter notebook
- Run basic SQL queries using the `psql` command-line client

## Workflow

The four phases of this tutorial are:

1. **Start the container** — launch a Docker container with PostgreSQL installed
2. **Start the Postgres server** — run the server process inside the container
3. **Create databases** — load example datasets into PostgreSQL
4. **Query the data** — connect via `psql` or Jupyter and run SQL

## Set up environment

- Read the instructions in the top-level `README.md` to clone the class repo
  `umd_data605` and set up the Docker environment
- Make sure the Docker daemon is running on your computer (e.g., Docker Desktop
  for Mac)
  - See https://www.docker.com/products/docker-desktop for installation
    instructions

- Open a terminal and navigate to the class repo:
  ```bash
  # E.g., GIT_ROOT=~/src/umd_data605
  > cd $GIT_ROOT
  > ls
  Dockerfile    LICENSE       README.md     __pycache__   dev_scripts   project_template lectures      tutorials

  > cd tutorials/tutorial_postgres
  > ls
  Dockerfile        docker_bash.sh    docker_build.sh   docker_clean.sh   docker_exec.sh    docker_push.sh
  README.md         bashrc            etc_sudoers       pg_hba.conf       postgresql.conf   run_psql_server.sh
  run_jupyter.sh    tutorial_basics   tutorial_university   tutorial_seven_dbs
  ```

## Build container (optional)

- The provided scripts (e.g., `docker_bash.sh`) pull and use the pre-built
  image from Docker Hub. Building the container yourself is optional.

- To examine how the container is built:
  ```bash
  > cd $GIT_ROOT/tutorials/tutorial_postgres
  > vi Dockerfile docker_*.sh
  ```

- Build the Docker container (this takes approximately 10 minutes):
  ```bash
  > ./docker_build.sh
  ```

## Run container

- Run a bash shell inside the Docker container:
  ```bash
  > ./docker_bash.sh
  ```

- You should see the prompt from Docker showing the user and container ID:
  ```bash
  postgres@09913bf19d81:/$
  ```

- Verify that your local directory is mounted inside the container:
  ```bash
  docker> ls /data
  Dockerfile  README.md  docker_bash.sh   docker_clean.sh  etc_sudoers
  pg_hba.conf      run_psql_server.sh  tutorial_basics     tutorial_university
  ...
  ```

- The files from `tutorials/tutorial_postgres` on your computer appear under
  `/data` inside the container because `docker_bash.sh` mounts the directory:
  ```bash
  docker run ... -v /Users/saggese/src/umd_data605/tutorials/tutorial_postgres:/data
  ```

## PostgreSQL

- PostgreSQL is already installed in the container. The following sections
  explain the setup.

- PostgreSQL runs in client-server mode:
  - The **server** is a continuously running process that listens on a specific
    port (default: `5432`)
  - A **client** connects to the server over that port to send SQL commands and
    receive results
  - In this setup the client and server run inside the same container, but they
    could be on separate machines

- Connecting using the `psql` client is the easiest way to interact with
  PostgreSQL:
  - It provides command-line access to the database
  - GUI clients (e.g., pgAdmin, DBeaver) are also available

- PostgreSQL has a default superuser called `postgres`:
  - You can do everything under that username, or create a separate username
  - If you run a command (e.g., `createdb`) without specifying a user, it uses
    the Linux username you are logged in as
  - If no matching PostgreSQL user exists, the command will fail — use
    `-U postgres` to run as the superuser

- Note: no password is required here because PostgreSQL uses **peer
  authentication** by default. Peer authentication means PostgreSQL trusts the
  operating system user without requiring a password — the `postgres` OS user
  maps directly to the `postgres` database superuser. Password authentication
  is required for network connections from external applications.

- The `psql` client accepts many options (user, port, host, etc.). See the
  documentation: http://www.postgresql.org/docs/current/static/app-psql.html

- The `University` dataset used in examples is provided on the book website:
  http://www.db-book.com

## Start Postgres

- Start the PostgreSQL server inside the container.

- First, examine the startup script:
  ```bash
  docker> vi /data/run_psql_server.sh
  ```
  The script contains:
  ```bash
  # -x: print each command before executing; -e: exit on first error
  service --status-all
  /etc/init.d/postgresql start
  service --status-all
  ```

- Run the script to start the PostgreSQL service:
  ```bash
  docker> /data/run_psql_server.sh
  + service --status-all
  [ - ]  cron
  [ ? ]  hwclock.sh
  [ - ]  postgresql
  [ - ]  procps
  [ - ]  sysstat
  + /etc/init.d/postgresql start
  * Starting PostgreSQL 14 database server [ OK ]
  + service --status-all
  [ - ]  cron
  [ ? ]  hwclock.sh
  [ + ]  postgresql
  [ - ]  procps
  [ - ]  sysstat
  ```
  The `[ + ]` next to `postgresql` confirms the server is running.

## Creating example databases

### Creating the small university database

- With the Postgres server running, examine the initialization script:
  ```bash
  docker> more /data/tutorial_university/init_small_psql_university_db.sh
  #!/bin/bash -xe

  createdb university
  # Create the schema.
  psql --command "\i /data/tutorial_university/DDL.sql;" university
  # Insert some data in the DB.
  psql --command "\i /data/tutorial_university/smallRelationsInsertFile.sql;" university
  ```

- Examine the schema definition:
  ```bash
  docker> more /data/tutorial_university/DDL.sql
  ...
  ```

- Examine the data insert file:
  ```bash
  docker> more /data/tutorial_university/smallRelationsInsertFile.sql
  ...
  ```

- Populate the `university` database by running the script:
  ```bash
  docker> /data/tutorial_university/init_small_psql_university_db.sh
  + createdb university
  + psql --command '\i /data/tutorial_university/DDL.sql;' university
  psql:/data/tutorial_university/DDL.sql:1: NOTICE:  table "prereq" does not exist, skipping
  DROP TABLE
  ...
  ```

- The script runs these three steps:

  1. Create a database called `university`:
     ```bash
     docker> createdb university
     ```

  2. Create the tables using the DDL schema:
     ```bash
     docker> psql --command "\i DDL.sql" university
     ```

  3. Populate the tables with the small dataset:
     ```bash
     docker> psql --command "\i smallRelationsInsertFile.sql" university
     ```

### Creating the university_large database

- You can create a second database `university_large` for the larger dataset
  from the book website: http://www.db-book.com
- Because the table names are identical to the small dataset, a separate
  database is required to avoid conflicts
- The process mirrors the small university setup: create a new database, load
  the DDL schema, then load the larger insert file:
  ```bash
  docker> createdb university_large
  docker> psql --command "\i /data/tutorial_university/DDL.sql;" university_large
  docker> psql --command "\i /data/tutorial_university/largeRelationsInsertFile.sql;" university_large
  ```

## Connecting to Postgres

You can connect to the Postgres server in three ways:

| Method | When to use |
|---|---|
| `psql` inside the container | Quickest for ad-hoc queries and setup |
| `psql` from your laptop | Demonstrates the client-server model over a network port |
| Jupyter notebook | Recommended for running the tutorial notebooks |

### Connecting from inside the container

- With the server running, connect to it from the same container shell:
  ```bash
  docker> psql -U postgres
  psql (14.5 (Ubuntu 14.5-0ubuntu0.22.04.1))
  Type "help" for help.

  postgres=#
  ```

- Switch to the `university` database:
  ```bash
  postgres=# \c university
  SSL connection (protocol: TLSv1.3, cipher: TLS_AES_256_GCM_SHA384, bits: 256, compression: off)
  You are now connected to database "university" as user "postgres".

  university=#
  ```

### Using a second terminal in the same container

- If the server is started in one terminal, you can open a second terminal
  attached to the same running container using `docker_exec.sh`:

  - In terminal 1, start the container and the server:
    ```bash
    > ./docker_bash.sh
    docker> /data/run_psql_server.sh
    ```

  - In terminal 2, attach to the running container:
    ```bash
    > ./docker_exec.sh
    FULL_IMAGE_NAME=gpsaggese/umd_data605_postgres
    CONTAINER ID   IMAGE                            COMMAND   CREATED         STATUS         PORTS                                            NAMES
    2ea21580ffb9   gpsaggese/umd_data605_postgres   "bash"    5 minutes ago   Up 5 minutes   0.0.0.0:5432->5432/tcp, 0.0.0.0:8888->8888/tcp   umd_data605_postgres
    CONTAINER_ID=2ea21580ffb9
    postgres@2ea21580ffb9:/$
    ```

  - You can now run queries in terminal 2 while the server runs in terminal 1.

### Connecting from your laptop

- From your laptop (outside Docker) you need the Postgres client installed:
  ```bash
  > brew install postgresql
  ```

- Connect to the server running inside Docker (which exposes port 5432):
  ```bash
  > psql -U postgres -h localhost
  SSL connection (protocol: TLSv1.3, cipher: TLS_AES_256_GCM_SHA384, bits: 256, compression: off)
  You are now connected to database "postgres" as user "postgres".

  postgres=#
  ```

- This demonstrates the client-server model: the `psql` client on your laptop
  connects over port 5432 to the PostgreSQL server running inside Docker.

### Connecting from Jupyter notebook

- Start Jupyter inside the container:
  ```bash
  docker> /data/run_jupyter.sh
  ```

- Open a browser on your laptop and navigate to:
  ```
  http://localhost:8888/tree/data/tutorial_university
  ```

- You will see three tutorial notebooks:
  ```
  sql_basics.ipynb
  sql_joins.ipynb
  sql_nulls_and_unknown.ipynb
  ```

- The notebooks connect to your local PostgreSQL instance and let you run SQL
  queries directly from notebook cells.

- Queries use Jupyter **magic commands** — special directives that extend
  notebook cells beyond plain Python:
  - `%sql` — run a single-line SQL command
  - `%%sql` — run a multi-line SQL block

- **Important**: the `university` database must exist before running the
  notebooks. Follow the steps in the "Creating example databases" section above
  first.

## Using psql CLI

Once inside `psql`, you can use backslash (`\`) commands to manage the
database. These are client commands, not SQL — they control the `psql` session
itself.

### Quick reference

| Command | Description |
|---|---|
| `\h` | Help on SQL command syntax |
| `\?` | Help on psql backslash commands |
| `\l` | List all databases |
| `\c <db>` | Connect to a database |
| `\d` | List tables in the current database |
| `\d <table>` | Show schema for a table |
| `\q` | Quit psql |

All commands are documented at:
http://www.postgresql.org/docs/current/static/app-psql.html

### Examples

- Get general help:
  ```
  postgres=# help
  You are using psql, the command-line interface to PostgreSQL.
  Type:  \copyright for distribution terms
         \h for help with SQL commands
         \? for help with psql commands
         \g or terminate with semicolon to execute query
         \q to quit
  ```

- List psql backslash commands:
  ```
  postgres=# \?
  General
    \copyright             show PostgreSQL usage and distribution terms
    \crosstabview [COLUMNS] execute query and display results in crosstab
    \errverbose            show most recent error message at maximum verbosity
    \g [(OPTIONS)] [FILE]  execute query (and send results to file or |pipe);
  ...
  ```

- List available SQL commands:
  ```
  postgres=# \h
  Available help:
    ABORT                            ALTER SYSTEM                     CREATE FOREIGN DATA WRAPPER ...
    ALTER AGGREGATE                  ALTER TABLE                      CREATE FOREIGN TABLE        ...
  ...
  ```

- Get help for a specific SQL command:
  ```
  postgres=# \h create
  Command:     CREATE ACCESS METHOD
  Description: define a new access method
  Syntax:
  CREATE ACCESS METHOD name
      TYPE access_method_type
      HANDLER handler_function
  ```

- Show all databases:
  ```
  postgres=# \l
                                     List of databases
         Name            |  Owner   | Encoding | Collate |  Ctype  |   Access privileges
  ----------------------+----------+----------+---------+---------+-----------------------
   postgres             | postgres | UTF8     | C.UTF-8 | C.UTF-8 |
   template0            | postgres | UTF8     | C.UTF-8 | C.UTF-8 | =c/postgres          +
                        |          |          |         |         | postgres=CTc/postgres
   template1            | postgres | UTF8     | C.UTF-8 | C.UTF-8 | =c/postgres          +
                        |          |          |         |         | postgres=CTc/postgres
   university           | postgres | UTF8     | C.UTF-8 | C.UTF-8 |
   university_large     | postgres | UTF8     | C.UTF-8 | C.UTF-8 |
  (5 rows)
  ```
  The `postgres` database is created automatically by PostgreSQL for internal
  use.

- Connect to the `university` database (note the prompt changes):
  ```
  postgres=# \c university
  SSL connection (protocol: TLSv1.3, cipher: TLS_AES_256_GCM_SHA384, bits: 256, compression: off)
  You are now connected to database "university" as user "postgres".

  university=#
  ```

- List tables in the current database:
  ```
  university=# \d
             List of relations
   Schema |    Name    | Type  |  Owner
  --------+------------+-------+----------
   public | advisor    | table | postgres
   public | classroom  | table | postgres
   public | course     | table | postgres
   public | department | table | postgres
   public | instructor | table | postgres
   public | prereq     | table | postgres
   public | section    | table | postgres
   public | student    | table | postgres
   public | takes      | table | postgres
   public | teaches    | table | postgres
   public | time_slot  | table | postgres
  (11 rows)
  ```

- Query a table:
  ```
  university=# select * from instructor;
    id   |    name    | dept_name  |  salary
  -------+------------+------------+----------
   10101 | Srinivasan | Comp. Sci. | 65000.00
   12121 | Wu         | Finance    | 90000.00
   15151 | Mozart     | Music      | 40000.00
   22222 | Einstein   | Physics    | 95000.00
   32343 | El Said    | History    | 60000.00
   33456 | Gold       | Physics    | 87000.00
   45565 | Katz       | Comp. Sci. | 75000.00
   58583 | Califieri  | History    | 62000.00
   76543 | Singh      | Finance    | 80000.00
   76766 | Crick      | Biology    | 72000.00
   83821 | Brandt     | Comp. Sci. | 92000.00
   98345 | Kim        | Elec. Eng. | 80000.00
  (12 rows)
  ```

## Other notebook examples

The `tutorial_university` directory contains three Jupyter notebooks. Open them
at `http://localhost:8888/tree/data/tutorial_university` after starting
Jupyter with `/data/run_jupyter.sh`.

- **`sql_basics.ipynb`** — introductory SQL queries (SELECT, WHERE, ORDER BY,
  aggregation) on the university dataset
- **`sql_joins.ipynb`** — JOIN types and multi-table queries
- **`sql_nulls_and_unknown.ipynb`** — handling NULL values and three-valued
  logic in SQL
