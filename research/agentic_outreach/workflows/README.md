# ck_marketing/workflows

This directory contains the core workflow orchestration and data processing pipelines
for CK Marketing operations, including contact management, campaign selection, and CRM
database operations.

## Structure of the Dir

- `notebooks/`
  - Jupyter notebooks for interactive workflows and data processing pipelines
- `test/`
  - Unit tests for workflow modules
- `test/outcomes/`
  - Golden file test outputs for regression testing

## Description of Files

- `workflows2.py`
  - GoogleSheetsHelper class for Google Sheets integration and DataFrame operations
- `campaign_utils.py`
  - Campaign selection utilities for filtering and sampling contact subsets
- `yamm_utils.py`
  - YAMM schema normalization and campaign results aggregation utilities
- `data_loaders.py`
  - Functions to load contact data from Google Sheets with caching support
- `contact_df_utils.py`
  - Contact DataFrame processing utilities including hashing and filtering operations
- `crm_db.py`
  - SQLite database operations library for Contact and LinkedIn table management
- `manage_crm_db.py`
  - Command-line interface for CRM database management operations
- `yamm_utils.py`
  - YAMM campaign schema normalization and results processing utilities
- `save_cache.sh`
  - Shell script for backing up email verification and enrichment caches

### Notebook Files

- `Master.process_contacts.py`
  - Master notebook for processing and managing contact data workflows
- `Master.cold_outreach.py`
  - Pipeline for extracting and validating profiles with PhantomBuster, DropContact, HunterIO APIs
- `Master.plugin_gallery.py`
  - Demonstration notebook showcasing available plugin integrations
- `Process_GP_LIn_contacts.py`
  - Workflow for processing GP's LinkedIn connections and enriching contact data
- `Process_hedge_fund_list.py`
  - Pipeline for processing hedge fund contact lists
- `Process_super_networking_sheet.py`
  - Workflow for processing super networking sheet contacts
- `debug_hunters_io.py`
  - Debugging utilities for HunterIO API integration

## Description of Executables

### `manage_crm_db.py`

#### What It Does

- Provides command-line interface for managing SQLite CRM database with Contact and
  LinkedIn tables
- Supports database creation, data insertion, querying, and CSV export operations
- Validates data schemas and enforces primary key constraints on insert operations

#### Examples

**Create a new CRM database**

```bash
> python manage_crm_db.py --db_path contacts.db --action create_db
```

**Insert contact data from JSON file**

```bash
> python manage_crm_db.py --db_path contacts.db --action insert_contact --contact_json contact_data.json
```

**Query contacts with filtering and limit**

```bash
> python manage_crm_db.py --db_path contacts.db --action query_contacts --where "company_name='Acme Corp'" --limit 10
```

**Export database tables to CSV files**

```bash
> python manage_crm_db.py --db_path contacts.db --action export_csv --output_dir ./exports
```

**Insert LinkedIn profile data**

```bash
> python manage_crm_db.py --db_path contacts.db --action insert_linkedin --linkedin_json linkedin_data.json
```

### `crm_db.py`

#### What It Does

- Core library module providing SQLite database operations for CRM tables
- Implements Contact_table and LinkedIn_table schema definitions and CRUD operations
- Provides functions for database creation, data insertion with conflict resolution, and
  querying
- Can be imported as a library or executed directly for database operations

#### Examples

**Create database programmatically**

```bash
> python -c "import ck_marketing.workflows.crm_db as crm; crm.create_db('contacts.db')"
```

**Get contact count from database**

```bash
> python -c "import ck_marketing.workflows.crm_db as crm; print(crm.get_contact_count('contacts.db'))"
```

**Export contacts to DataFrame**

```bash
> python -c "import ck_marketing.workflows.crm_db as crm; df = crm.get_contacts_as_df('contacts.db'); print(df.head())"
```

### `save_cache.sh`

#### What It Does

- Backs up email verification and enrichment cache files to backup directory
- Creates timestamped backups and makes them read-only to prevent accidental
  modification
- Preserves cache state for debugging and recovery purposes

#### Examples

**Run cache backup**

```bash
> ./save_cache.sh
```

**View backed up cache files**

```bash
> ls -lh ~/src/backup/cache.*.json
```
