Marketing data extraction and cold outreach automation platform for B2B lead
generation, investor relations, and contact database management.

# Workflows

- The overall flow is as below

## Phase A: Import contacts in CRM
- Each data source (e.g., all the VCs, LinkedIn connections, ...) has a Master
  notebook that cleans the data and import it into the CRM, e.g.,
  - `ck_marketing/workflows/notebooks/Master.A1.process_VC_contacts.ipynb`
     - Data scraped from many VC databases
  - `ck_marketing/workflows/notebooks/Master.A2.process_pitchbook_contacts.ipynb`
     - Data from PitchBook exports
  - `ck_marketing/workflows/notebooks/Master.A3.process_my_lin_connections.ipynb`
     - Data from LinkedIn connection exports
  - `ck_marketing/workflows/notebooks/Master.A4.process_SN_search_export.ipynb`
     - Data from Sales Navigator Searches
  - `ck_marketing/workflows/notebooks/Master.A5.process_LinkedIn_Profile_Scraper.ipynb`
     - Data from LinkedIn Profile Scraper (e.g., when enriching)

- The plugins are described in `ck_marketing/workflows/notebooks/Master.plugin_gallery.ipynb`

## Phase B: Manage CRM
- `ck_marketing/workflows/notebooks/Master.B1.manage_crm_db.ipynb`
  - Build and manage the CRM (e.g., create a new CRM, query, create campaign)

- `ck_marketing/workflows/notebooks/Master.B2.enrich_contacts.ipynb`
  - The CRM can be enriched with:
     - LinkedIn information from PB LinkedIn Profile Scraper
     - Updated emails (e.g., from hunter.io)
     - Validated emails (e.g., from hunter.io)

- `ck_marketing/workflows/notebooks/Master.B3.update_crm_db.ipynb`
  - Update CRM with the results of a campaign (e.g., bounced emails)

## Phase C: Manage outreach campaigns
- `ck_marketing/workflows/notebooks/Master.C1.generate_campaign.ipynb`

- `ck_marketing/workflows/notebooks/Master.C2.patch_up_campaign.ipynb`
  - Read a Gsheet with email / linkedin
  - Patch it up

- `ck_marketing/workflows/notebooks/Master.C3.analyze_gsheet.ipynb`
  - Read a Gsheet
  - Understand what kind of data contains

# Structure of the code

- `misc/`
  - One-off data extraction scripts for target companies and databases
- `notebooks/`
  - Task-specific Jupyter notebooks for ad-hoc marketing campaigns
- `plugins/`
  - Third-party integration plugins for various marketing data sources
- `workflows/`
  - Integrated workflows combining multiple APIs for cold outreach campaigns
- `test/`

# Description of Files

## Root Directory

- `ck_marketing.how_to_guide.md`
  - Comprehensive guide covering all modules, APIs, workflows, and usage patterns

## plugins

### plugins/docsend/

- `plugins/docsend/notebooks/Postprocess_Docsend_Data.py`
  - Post-processing script for DocSend document sharing analytics data

### plugins/dropcontact/

- `plugins/dropcontact/drop_contact_api.py`
  - DropContact API client for finding and verifying emails using names and companies
- `plugins/dropcontact/notebooks/SorrTask606_Get_email_from_dropcontact.py`
  - Workflow notebook demonstrating DropContact email extraction process

### plugins/hunterio/

- `plugins/hunterio/hunterio_api.py`
  - Hunter.io API wrapper with email finding, verification, and Google Sheets integration

### plugins/linkedin/

- Code
  - `plugins/linkedin/phantombuster_api.py`
    - PhantomBuster API wrapper for LinkedIn automation, profile extraction, and agent management
  - `plugins/linkedin/sales_navigator_query.py`
    - Sales Navigator query builder with LinkedIn filter mappings and codes
  - `plugins/linkedin/linkedin_utils.py`
    - Utility functions for filtering, normalizing, and processing LinkedIn profile data

- Notebooks
  - `plugins/linkedin/notebooks/clean_up_names.py`
    - Script for cleaning and normalizing LinkedIn profile names
  - `plugins/linkedin/notebooks/process_profile_export.py`
    - Process and structure exported LinkedIn profile data
  - `plugins/linkedin/notebooks/process_search_export.py`
    - Process LinkedIn search results into structured DataFrames
  - `plugins/linkedin/notebooks/create_gsheets_and_fetch_result_csv.py`
    - Google Sheets integration for PhantomBuster results export

### plugins/signal_nfx/

- `plugins/signal_nfx/extract_investors_from_signal_list.py`
  - Selenium-based scraper for Signal NFX investor lists
- `plugins/signal_nfx/CmampTask11104_Scrape_SignalNFX.py`
  - Task script for automated Signal NFX data scraping
- `plugins/signal_nfx/CmampTask11142_Process_SignalNFX_records.py`
  - Post-processing and normalization of scraped Signal NFX data
- `plugins/signal_nfx/notebooks/SorrTask612_Get_information_from_Signal.py`
  - Interactive workflow for Signal NFX investor information extraction

### plugins/tracxn/

- `plugins/tracxn/extract_VCs_from_Tra_search_mhtml.py`
  - Parse Tracxn MHTML files to extract VC firm information
- `plugins/tracxn/extract_people_from_Tra_company_mhtml.py`
  - Extract team members and founders from Tracxn company pages
- `plugins/tracxn/notebooks/SorrTask601_Extract_VCs_from_Tra_search_mhtml.py`
  - Workflow notebook for VC extraction from Tracxn search results
- `plugins/tracxn/notebooks/SorrTask601_Extract_people_from_Tra_company_html.py`
  - Workflow notebook for people extraction from Tracxn company pages

### plugins/VC_Sheet/

- `plugins/VC_Sheet/VCsheet_scrape.py`
  - Web scraper for VC firm data sheets

## misc/

- `misc/CmampTask11363_Target_companies_list.py`
  - Script to build and manage target company lists
- `misc/CmampTask11363_Growjo_data.py`
  - Growjo database data extraction and processing
- `misc/CmampTask11363_Fortune500_data.py`
  - Fortune 500 company data extraction and structuring

## notebooks/

- `notebooks/CmampTask11189_Create_flow_to_find_decision_makers_for_a_firm.py`
  - Workflow for identifying decision makers at target companies
- `notebooks/CmampTask8908_Find_GP_VC_connections.py`
  - Map and analyze connections between GPs and VC firms
- `notebooks/CmampTask8909_Collect_emails_Tier1_VCs.py`
  - Email collection workflow for Tier 1 venture capital firms
- `notebooks/CmampTask8811_Find_Bloomberg_decision_makers.py`
  - Identify and extract decision maker contacts from Bloomberg data
- `notebooks/CmampTask11283_Scrape_airtable_into_a_CSV.py`
  - Airtable database scraping and CSV export utility
- `notebooks/CmampTask11450_Find_names_of_decision_makers_for_a_list_of_companies.py`
  - Bulk decision maker identification for company lists
- `notebooks/CmTask9143_Build_target_VC_investor_lists.py`
  - Comprehensive VC investor list compilation workflow
- `notebooks/CmTask10499_Scrape_the_money_20-20_attendees.py`
  - Money 20/20 conference attendee list scraping
- `notebooks/CmTask10499_Scrape_the_money_20-20_attendees_Selenium.py`
  - Selenium-based Money 20/20 attendee scraper
- `notebooks/CmTask11085_Money_20_extract_email.py`
  - Email extraction for Money 20/20 conference attendees
- `notebooks/CmampTask11473_Errors_in_HunterIO.py`
  - Debugging and error analysis for Hunter.io API issues
- `notebooks/CmTask9112_AI_VC_list_from_David.py`
  - AI-focused VC list processing from external sources

## workflows/

- `workflows/data_loaders.py`
  - Data loading module for scraping and importing contacts from various sources
    (LinkedIn, VCSheet, EuroVC, FolkApp, etc.)
  - Handles Google Sheets integration for data extraction
  - Contact schema normalization and validation
- `workflows/hyamm.py`
  - Core module for contact processing, YAMM integration, and campaign management
  - Contact DataFrame manipulation and statistics
  - YAMM campaign results processing and analytics
- `workflows/workflows.py`
  - Google Sheets helper class for programmatic sheet operations
- `workflows/notebooks/Master.cold_outreach.py`
  - Master orchestration script for end-to-end cold outreach campaigns
- `workflows/notebooks/Master_process_contacts.py`
  - Process and organize existing contact databases
- `workflows/notebooks/Process_GP_LIn_contacts.py`
  - Process contacts from specific LinkedIn and GP sources
- `workflows/notebooks/Process_hedge_fund_list.py`
  - Specialized workflow for hedge fund contact extraction
- `workflows/notebooks/Process_super_networking_sheet.py`
  - Process super-connector and networking event attendee data
- `workflows/notebooks/debug_hunters_io.py`
  - Debugging utilities and diagnostics for Hunter.io API

## test/

- `test/test_hyamm.py`
  - Unit tests for data loading functions from data_loaders module (data scraping and import)
- `test/test_hyamm_normalization.py`
  - Tests for contact schema normalization functions in data_loaders module
- `test/test_hyamm_process_df.py`
  - Tests for DataFrame processing operations in hyamm module
- `test/test_hyamm_yamm.py`
  - Tests for YAMM campaign integration and processing in hyamm module
