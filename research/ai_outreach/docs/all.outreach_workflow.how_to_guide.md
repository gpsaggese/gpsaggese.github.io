# Outreach Workflow

<!-- toc -->

- [Overview](#overview)
- [Pre-Pipeline: Environment Setup](#pre-pipeline-environment-setup)
- [Stage 1: Search for Contacts](#stage-1-search-for-contacts)
  * [Generate SN Query From Persona](#generate-sn-query-from-persona)
  * [Scrape Profiles via PhantomBuster](#scrape-profiles-via-phantombuster)
- [Stage 2: Enrich With Emails](#stage-2-enrich-with-emails)
  * [Find Emails](#find-emails)
  * [Verify Emails](#verify-emails)
- [Stage 3: Draft Personalized Emails](#stage-3-draft-personalized-emails)
- [Stage 4: Upload to Google Sheets](#stage-4-upload-to-google-sheets)
- [Running the Full Pipeline](#running-the-full-pipeline)
  * [End-to-End Run](#end-to-end-run)
  * [Resuming From a Stage](#resuming-from-a-stage)
  * [Partial Runs](#partial-runs)

<!-- tocstop -->

## Overview

The outreach pipeline finds contacts matching a target persona, enriches them
with emails, generates personalized email drafts using an LLM, and uploads the
results to Google Sheets for manual review and sending.

The pipeline consists of four stages, each reading a CSV and writing a CSV:

```
outreach_search -> outreach_enrich -> outreach_draft -> Google Sheets
```

Each stage can be run independently or chained via `outreach_run_all`.

## Pre-Pipeline: Environment Setup

- Set the required API keys as environment variables:
  - `Phantom_API_KEY`: PhantomBuster API key (for `outreach_search`)
  - `HUNTER_API_KEY`: Hunter.io API key (for `outreach_enrich`)
- Have a LinkedIn Sales Navigator account and a valid `li_at` session cookie for
  PhantomBuster scraping

## Stage 1: Search for Contacts

This stage converts a persona description into a list of matching LinkedIn
profiles. It runs in two sub-steps that can be decoupled.

### Generate SN Query From Persona

- Provide a natural-language persona description
- The LLM generates an optimized Sales Navigator boolean search query with
  filters (geography, seniority, industry, etc.)
- Review the query before scraping

```bash
> outreach_search \
    --persona "Series A fintech founders in NYC" \
    --query_only \
    --out_path sn_query.json
```

- Output: `sn_query.json` with `query`, `filters`, and `explanation` fields

### Scrape Profiles via PhantomBuster

- Copy the generated query into Sales Navigator, get the search URL
- Pass the URL to scrape matching profiles

```bash
> outreach_search \
    --sn_query_url "https://www.linkedin.com/sales/search/people?..." \
    --linkedin_cookie "$LI_AT" \
    --out_path contacts.csv
```

- Output: `contacts.csv` with names, titles, companies, LinkedIn URLs, and
  other profile fields from PhantomBuster

## Stage 2: Enrich With Emails

This stage finds email addresses and verifies their deliverability. Both
operations are incremental: contacts that already have an email or verification
status are skipped, so the script is safe to re-run.

### Find Emails

- For each contact that has a `company_domain` but no `email`, Hunter.io is
  queried to find the email
- Results are cached locally to avoid redundant API calls

### Verify Emails

- Each found email is checked via Hunter.io for deliverability
- Status values: `valid`, `invalid`, `accept_all`, `unknown`

```bash
> outreach_enrich \
    --in_path contacts.csv \
    --out_path enriched.csv
```

- Output: `enriched.csv` with `email`, `email_verification`,
  `email_timestamp`, and `email_verification_timestamp` columns added
- To skip verification, pass `--no_verify`
- To only verify existing emails without finding new ones, pass `--verify_only`

## Stage 3: Draft Personalized Emails

This stage uses an LLM to generate a personalized subject line and email body
for each contact. The LLM receives the contact's name, title, company, and
biography as context, along with the campaign goal.

```bash
> outreach_draft \
    --in_path enriched.csv \
    --campaign_goal "Introduce our causal AI platform to VC partners" \
    --sender_name "GP" \
    --sender_title "CEO, Causify" \
    --out_path drafts.csv
```

- Output: `drafts.csv` with `email_draft_subject` and `email_draft_body`
  columns added
- The default model is `gpt-5-nano`; override with `--model`
- Batch size is configurable with `--batch_size` (default: 20)

## Stage 4: Upload to Google Sheets

- Upload drafts to Google Sheets for review, editing, and manual sending from
  your preferred platform (YAMM, SendGrid, Gmail, etc.)

```bash
> outreach_draft \
    --in_path enriched.csv \
    --campaign_goal "Cold outreach to fintech founders" \
    --upload_gsheet \
    --sheet_name "Q1 outreach drafts" \
    --out_path drafts.csv
```

- The `outreach_run_all` orchestrator uploads automatically at the end unless
  `--no_upload` is passed

## Running the Full Pipeline

### End-to-End Run

`outreach_run_all` chains all stages and writes intermediate files to
`--out_dir`:

```bash
> outreach_run_all \
    --persona "Partners at crypto VC funds" \
    --sn_query_url "https://www.linkedin.com/sales/search/people?..." \
    --linkedin_cookie "$LI_AT" \
    --campaign_goal "Introduce our causal AI platform to VCs" \
    --sender_name "GP" \
    --sender_title "CEO, Causify" \
    --out_dir ./campaign_q1/
```

- Output files in `--out_dir`:
  - `1_contacts.csv`: raw scraped profiles
  - `2_enriched.csv`: with emails and verification
  - `3_drafts.csv`: with LLM-generated email drafts
- Final results are uploaded to Google Sheets

### Resuming From a Stage

If a stage has already been completed (e.g., you already have contacts), pass
`--skip_to` with the intermediate CSV:

```bash
# Start from enrich (already have contacts).
> outreach_run_all \
    --skip_to enrich \
    --in_path ./campaign_q1/1_contacts.csv \
    --campaign_goal "Introduce our AI platform" \
    --out_dir ./campaign_q1/

# Start from draft (already have enriched contacts).
> outreach_run_all \
    --skip_to draft \
    --in_path ./campaign_q1/2_enriched.csv \
    --campaign_goal "Follow up after demo" \
    --out_dir ./campaign_q1/
```

### Partial Runs

```bash
# Search + enrich only (no drafts).
> outreach_run_all \
    --persona "VP Engineering at B2B SaaS" \
    --sn_query_url "https://..." \
    --linkedin_cookie "$LI_AT" \
    --no_draft \
    --out_dir ./leads_q1/

# Skip Google Sheets upload.
> outreach_run_all \
    --skip_to draft \
    --in_path ./campaign_q1/2_enriched.csv \
    --campaign_goal "Follow up" \
    --no_upload \
    --out_dir ./campaign_q1/
```
