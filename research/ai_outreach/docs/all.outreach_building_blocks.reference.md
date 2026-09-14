# Outreach Building Blocks

<!-- toc -->

- [Overview](#overview)
- [Architecture](#architecture)
  * [Directory Structure](#directory-structure)
  * [Layered Design](#layered-design)
  * [Data Flow](#data-flow)
  * [Design Principles](#design-principles)
- [Scripts Layer](#scripts-layer)
  * [`outreach_search`](#outreach_search)
  * [`outreach_enrich`](#outreach_enrich)
  * [`outreach_draft`](#outreach_draft)
  * [`outreach_run_all`](#outreach_run_all)
  * [`scripts/utils`](#scriptsutils)
- [Plugins Layer](#plugins-layer)
  * [`plugins/hunterio/hunterio_api`](#pluginshunteriohunterio_api)
  * [`plugins/linkedin/phantombuster_api`](#pluginslinkedinphantombuster_api)
- [Workflows Layer](#workflows-layer)
  * [`workflows/contact_df_utils`](#workflowscontact_df_utils)
  * [`workflows/data_loaders_utils`](#workflowsdata_loaders_utils)
  * [`workflows/classify_utils`](#workflowsclassify_utils)
  * [`workflows/crm_db`](#workflowscrm_db)

<!-- tocstop -->

## Overview

`ai_outreach` is a pipeline for finding contacts, enriching them with emails,
and drafting personalized outreach using LLMs. This document describes the
architecture and each building block.

## Architecture

### Directory Structure

```
ai_outreach/
  scripts/             # CLI layer: executable pipeline scripts
    outreach_search     #   Stage 1: persona -> contacts
    outreach_enrich     #   Stage 2: contacts -> emails
    outreach_draft      #   Stage 3: emails -> LLM drafts
    outreach_run_all    #   Orchestrator: chains all stages
    utils               #   Shared CSV I/O and Google Sheets upload
  plugins/             # Integration layer: third-party API wrappers
    hunterio/           #   Email finding + verification
    linkedin/           #   PhantomBuster scraping
  workflows/           # Library layer: data transformations
    contact_df_utils    #   Contact cleaning and validation
    data_loaders_utils  #   Schema normalization
    classify_utils      #   LLM-based contact classification
    campaign_utils      #   Campaign selection logic
    crm_db              #   SQLite CRM (optional)
  cli/                 # CRM utility commands (optional)
  docs/                # Documentation
  tests/               # Unit tests
```

### Layered Design

The codebase is organized into three layers. Each layer depends only on the
layers below it.

```
  Scripts (CLI layer)
      |
      | orchestrates
      v
  Plugins (integration layer)      Workflows (library layer)
      |                                |
      | Hunter.io, PhantomBuster       | contact cleaning, schema
      | LLM via helpers.hllm_cli       | normalization, classification
      v                                v
  External APIs                    pandas DataFrames
```

- **Scripts** are thin orchestrators. They parse CLI arguments, load CSVs, call
  into plugins and workflows, and save results. Each script has a class-based
  executor (e.g., `ContactSearcher`, `ContactEnricher`, `EmailDrafter`)
  following the pattern used by DataMap scripts in `causal_kg`
- **Plugins** wrap external APIs with caching, rate limiting, and DataFrame
  interfaces. They have no knowledge of the pipeline structure
- **Workflows** are pure data transformations on DataFrames. They have no
  external API dependencies and no knowledge of the pipeline structure

### Data Flow

Each stage reads a CSV, adds columns, and writes a CSV. The intermediate files
are self-contained and can be inspected or edited between stages.

```
              persona (str)
                  |
                  v
         [outreach_search]
          LLM -> SN query
          PhantomBuster -> scrape
                  |
                  v
            1_contacts.csv
       (name, title, company, linkedin_url, ...)
                  |
                  v
         [outreach_enrich]
          Hunter.io -> find emails
          Hunter.io -> verify
                  |
                  v
            2_enriched.csv
       (+ email, email_verification)
                  |
                  v
         [outreach_draft]
          LLM -> personalized subject + body
                  |
                  v
            3_drafts.csv
       (+ email_draft_subject, email_draft_body)
                  |
                  v
           Google Sheets
       (review, edit, send manually)
```

### Design Principles

- **CSV-in, CSV-out**: Files are the interface between stages. No database
  required
- **Incremental**: Enrichment and verification skip contacts that already have
  data. Safe to re-run
- **Resumable**: `outreach_run_all --skip_to <stage>` restarts from any
  intermediate file
- **Review before send**: The pipeline produces drafts, not sent emails. A human
  reviews in Google Sheets and sends from any platform

## Scripts Layer

### `outreach_search`

- **Purpose**: Persona description -> Sales Navigator query -> PhantomBuster
  scrape -> contacts CSV
- **Class**: `ContactSearcher`
  - `persona_to_query(persona)`: LLM generates an SN boolean query + filters
    from a natural-language persona description. Returns JSON with `query`,
    `filters`, `explanation`
  - `scrape_profiles(sn_query_url, linkedin_cookie)`: Creates a PhantomBuster
    Sales Navigator Search Export agent, launches it, polls for results, returns
    a DataFrame
  - `run(...)`: Chains both stages. Supports `--query_only` to stop after query
    generation
- **Dependencies**: `helpers.hllm_cli`, PhantomBuster API

### `outreach_enrich`

- **Purpose**: Contacts CSV -> find emails + verify -> enriched CSV
- **Class**: `ContactEnricher`
  - `find_emails(df)`: Looks up emails via Hunter.io for contacts that have
    `company_domain` but no `email`. Incremental
  - `verify_emails(df)`: Verifies deliverability via Hunter.io. Returns status:
    `valid`, `invalid`, `accept_all`, `unknown`. Incremental
  - `run(df, verify, verify_only)`: Chains find + verify
- **Dependencies**: Hunter.io API

### `outreach_draft`

- **Purpose**: Enriched CSV -> LLM-generated personalized emails -> drafts CSV
- **Class**: `EmailDrafter`
  - `run(df, campaign_goal, sender_name, sender_title, ...)`: For each contact,
    extracts context (name, title, company, bio) and prompts the LLM to generate
    JSON with `subject` and `body`. Adds `email_draft_subject` and
    `email_draft_body` columns
- **Dependencies**: `helpers.hllm_cli`
- **Output columns**: `email_draft`, `email_draft_subject`, `email_draft_body`

### `outreach_run_all`

- **Purpose**: Single entry point chaining search -> enrich -> draft -> Google
  Sheets upload
- **Stages**: `search`, `enrich`, `draft`
- **Resume**: `--skip_to <stage> --in_path <csv>` to restart from any stage
- **Skip**: `--no_draft` to skip drafting, `--no_upload` to skip Google Sheets

### `scripts/utils`

- `load_contacts_csv(path)`: Load CSV, fill NaN with empty strings, cast to str
- `save_contacts_csv(df, path)`: Save DataFrame to CSV
- `filter_mailing_list(df)`: Filter by email presence and verification status
- `update_google_sheet(df, sheet_name)`: Upload DataFrame to Google Sheets with
  timestamp

## Plugins Layer

### `plugins/hunterio/hunterio_api`

Hunter.io API wrapper for email finding and verification. All single-contact
functions are cached via `hcache_simple`.

- `hunterio_email_finder(domain, first_name, last_name)`: Find email by name +
  domain
- `hunterio_find_emails_from_df(df)`: Batch email finding for a DataFrame.
  Incremental (skips rows that already have an email)
- `hunterio_verify_email(email)`: Verify a single email
- `hunterio_verify_emails_from_df(df)`: Batch verification for a DataFrame.
  Incremental
- `hunterio_domain_search(domain)`: Search all emails at a domain
- `get_hunterio_account_info()`: Check quota and reset date

### `plugins/linkedin/phantombuster_api`

PhantomBuster API wrapper for LinkedIn automation.

- `Phantom` class:
  - `create_sales_nav_phantom(name, query, cookie)`: Create a Sales Navigator
    Search Export agent
  - `launch_and_get_df(agent_id)`: Launch agent, poll for completion, download
    CSV, return DataFrame
  - `create_linkedIn_info_extractor_phantom(gsheet_url, name, cookie)`: Create a
    LinkedIn Profile Scraper agent from a Google Sheet of URLs
  - `create_linkedIn_url_finder_phantom(name, gsheet_url, cookie)`: Create a
    LinkedIn URL Finder agent from a Google Sheet of names

## Workflows Layer

### `workflows/contact_df_utils`

Contact DataFrame cleaning and validation. Operates on DataFrames following the
canonical contact schema (21 fields defined in `data_loaders_utils`).

- `add_hash(df)`: Compute MD5 hash from `first_name` + `last_name` +
  `linkedin_url`
- `clean_up_contact_df(df)`: Multi-phase cleanup:
  - Deduplicate emails
  - Remove invalid emails
  - Filter names with Chinese characters
  - Remove empty first names
  - Move misplaced URLs/emails between columns
  - Strip `mailto:` prefixes
- `sanity_check_contact_df(df)`: Validate without modifying (LinkedIn URLs,
  emails, verification status, email-domain matching)
- `print_contact_df_stats(df)`: Summary statistics (duplicates, emails,
  verification, origins)

### `workflows/data_loaders_utils`

Schema definition and normalization for contact data from heterogeneous sources.

- `get_contact_df_schema()`: Returns the 21-field canonical contact schema:
  `hash`, `origin`, `origin_timestamp`, `first_name`, `last_name`, `email`,
  `email_timestamp`, `email_verification`, `email_verification_timestamp`,
  `linkedin_url`, `job_title`, `linked_timestamp`, `company_name`,
  `company_domain`, `company_type`, `industry`, `city`, `country`,
  `enrichment_timestamp`, `type`, `biography`
- `normalize_contact_schema(df, cols_map)`: Rename and reorder columns to match
  the canonical schema. Fill missing columns with empty strings
- `fuzzy_column_matching(cols)`: Match arbitrary column names to schema fields
  using aliases (e.g., "LinkedIn URL" -> `linkedin_url`, "company" ->
  `company_name`, "title" -> `job_title`)

### `workflows/classify_utils`

LLM-based contact classification using `helpers.hllm_cli`.

- `classify_industry_type_executive(df)`: Classify each contact into:
  - **Industry** (33 categories): Financial Services, IT - Software, Healthcare,
    etc.
  - **Type** (12 categories): VC, Angel Investor, Corporate Development,
    Technology, etc.
  - **Executive title** (13 categories): CEO, CTO, Partner, Founder, etc.
- Uses `hllmcli.apply_llm_prompt_to_df` for batched LLM calls

### `workflows/crm_db`

SQLite-based CRM database. Optional; not required by the pipeline scripts.

- Tables: `Contact`, `Campaign_info`, `Campaign_targets`, `LinkedIn`
- `create_db(db_path)`: Initialize the database with schema
- `insert_contact_df(db_path, df, mode)`: Insert contacts with overlap handling:
  - `assume_no_overlap`: assert no duplicates
  - `assume_idempotent`: check existing data matches
  - `keep_new`: overwrite existing with new
  - `keep_existing`: skip duplicates
- `get_as_df(db_path, table_name)`: Read a table as DataFrame
- `export_db_to_csv(db_path, output_dir)`: Export all tables to CSV
- CLI access via `python -m ai_outreach.cli.main`
