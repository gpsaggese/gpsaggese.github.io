# LinkedIn Plugin

This plugin provides tools for LinkedIn data extraction, processing, and analysis
using PhantomBuster automation and Sales Navigator queries.

## Structure of the Dir

- `notebooks/`
  - Jupyter notebooks for LinkedIn data processing workflows
- `test/`
  - Unit tests for LinkedIn utilities
- `test/outcomes/`
  - Test output files and golden test results

## Description of Files

- `__init__.py`
  - Package initialization for LinkedIn plugin module
- `linkedin_utils.py`
  - Utilities for filtering dataframes, cleaning names, and processing LinkedIn profile data
- `phantombuster_api.py`
  - API wrapper for PhantomBuster automation platform to scrape LinkedIn data
- `sales_navigator_query.py`
  - Parser and generator for LinkedIn Sales Navigator search query URLs

## Description of Notebooks

- `notebooks/clean_up_names.py`
  - Cleans and normalizes first and last names from CSV files using regex and LLM
- `notebooks/create_gsheets_and_fetch_result_csv.py`
  - Creates Google Sheets structure and downloads PhantomBuster result CSVs to Google Drive
- `notebooks/process_profile_export.py`
  - Filters LinkedIn profile export data by keywords like volunteer, adjunct, consultant
- `notebooks/process_search_export.py`
  - Filters LinkedIn search export data by location and prepares for profile scraping
