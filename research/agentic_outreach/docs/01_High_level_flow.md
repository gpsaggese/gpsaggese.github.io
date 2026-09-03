# Intro

## Brief Description
- We want to automate the process of running a cold outreach campaign

- We have built a master Jupyter notebook with the entire flow that we can use
  to go from LinkedIn / Sales Navigator query to a Google Sheet with the
  information for an outreach (names, emails, positions)
  - [https://github.com/cryptokaizen/cmamp/blob/master/ck_marketing/process_automation/Master.cold_outreach.ipynb](https://github.com/cryptokaizen/cmamp/blob/master/ck_marketing/process_automation/Master.cold_outreach.ipynb)

- For specific tasks, there is a single notebook with only what's needed for
  that task, which is a particular version of the master notebook

- We stitched together several services
  - E.g., LinkedIn scraping, PhantomBuster, a little of ML, hunter.io,
    dropcontact, YAMM
  - At some point, we should also add a CRM to the flow

## Abbreviations
SN = SalesNavigator

PB = PhantomBuster

## Workflow
- Come up with a list of sectors to target and decision makers
  - E.g., David's
    [TO DELETE Kaizen ICP](https://docs.google.com/document/d/1IFrrUiA7vciZzkQU9-mhMM0-PfTdTxRCeIkw1yYLl4g/edit#heading=h.i63h21soi3hx))

- Come up with lists of target companies for each sector
  - E.g.,
    [! Company lists](https://docs.google.com/spreadsheets/d/10_znRzH4jV54bCeTWYqwTkt9fLJIgpfw/edit?usp=drive_web&ouid=101788911614446619403&rtpof=true)

- Create a SalesNavigator (SN) query for the company with the decision makers
  - [https://www.linkedin.com/sales/home](https://www.linkedin.com/sales/home)
  - We have templates of SN queries that we can customize
    - E.g., "decision makers for a VC firm", "decision makers for a consulting
      firm"

- We can analyze an SN query to the list of query to process
  - E.g.,
    [Cold outreach - LIN searches](https://docs.google.com/document/d/1sP9J8CWltdYKsYQxghXWCKMKHade6Xps/edit?usp=drive_web&ouid=101788911614446619403&rtpof=true)

- Use PhantomBuster to extract the candidates (manually at first and then using
  API)
  - [https://phantombuster.com/8472730339660855/phantoms](https://phantombuster.com/8472730339660855/phantoms)
  - E.g.,
    [VC leads - Shaunak](https://drive.google.com/drive/folders/1lx7iqPp43_iNvhWHvxse3t1U551BrqGW)
  - E.g.,
    [Flutter_decision_makers](https://docs.google.com/spreadsheets/d/11s8ayk_fvMEy2AcQM8bjlWoDA3uoA3NVrezmpc8yZes/edit#gid=1146488863)

- The structure of the notebook pipeline is
  - Given a sector and a company
  - Generate the SN query automatically
  - The output is in
    [VC leads - Shaunak](https://drive.google.com/drive/folders/1lx7iqPp43_iNvhWHvxse3t1U551BrqGW)
  - Run PB (only export profiles) -> linked in gsheet
    - E.g.,
    - TODO: Automate
  - Read Gsheet "export_search"
  - Filter by certain criteria (e.g., "remove lawyers from the list")
    - Goal is to keep only the decision makers
    - We have a set of functions in the lib that help doing filtering
    - Every company is different and so we pick the functions that are best and
      / or we customize it
  - Save in Gsheet tab "export_search.filtered"
  - Extract emails
    - Use hunter.io
    - Use Dropcontact (already implemented)
  - Save in Gsheet tab "email"
  - Filter/remove dead emails
    - It's totally ok to use hunter.io
    - Neverbounce
  - Save Gsheet as "email.cleaned" tab
    - We want to get to 20-50 people per company
  - Create customized email
  - Run Yamm campaign

### Outreach Flow
```graphviz
flowchart TD
   LIn_Search --> PB_Email[[PB_Email]]
   GP_LIn_contacts --> PB_Email
   Folkapp_data --> X
   VCSheet_data --> X
   NFX_data --> X
   PB_Email --> X
   HedgeFundList --> PB_Email
   Yamm_data --> X
   X[(Contact_df)]
       --> Clean_up --> Clean_up_names
       --> Email_check --> Tag_type --> Y[(Contact_df2)]
   Y --> Extract_Campaign --> Z[(Campaign_Gsheet)]
   Z --> Yamm[[Yamm]] --> Yamm_data[(Yamm_data)]
   Z --> PB_LIn_Outreach[[PB_LIn_Outreach]]
```

### Main Flow
- Load Contact data
  - Read all the scraped Gsheet (e.g., FolkApp, VCSheet, ...)
  - Read LIn connections from people LIn accounts
  - Read mixed data sources
    - HedgeFundList
    - Super networking spreadsheet
  - Each data loader
    - Reads data from a different format
    - Maps data from original schema to Contact schema
    - Normalizes data into the Contact schema
  - Concat all data
    - Merge all the data
    - Clean up
    - Compute stats
    - Serialize it as a Gsheet

- Process Contact data
  - Clean up first / last names
  - Enrich with Hunters.io data
    - Query data from Hunters.io
    - Merge the data back
    - We should do the emails too
  - Validate emails
    - Use Hunters.io to check that all emails are valid
  - Assign a category to leads
    - E.g., investor, VC, customer

- YAMM pipeline
  - We read the data from previous YAMM campaigns

- Update the state of each contact

- Extract YAMM / LIn campaign

### Problems / Guidelines:
The Contact schema is often changing

- E.g., we find out that we want to track some new information that we
  originally ignored

- Solutions
  - We need to be able to re-run everything from the original data

Some fields are separated in some cases, e.g.,

- "series type = {series_seed, series_A, ...}"

- "geography"

- Solutions
  1.  Have multiple tables (not sure it's worth it)

  2.  Use key-value db

  3.  Make it simple to recreate the dataset once we change something (this
      seems the best)

Not all the original data sets have all the info

- Solutions
  - We can encode multiple values in a single cell
  - Just make the schema large and have easy ways to slice the data (best)

Sometimes one wants to open a pipeline stage and look into it, other times you
want to run it as an atomic block

- Solution:
  - Have multiple notebooks with different level of details
  - Serialize data to exchange data
  - Have multiple functions back-to-back
  - Put all the functions in a single function once it's debugged
  - Have switches to run in verbose mode or not

The notebook is a pipeline of pipelines

- The main pipeline has several stages
  - E.g., Master_process_contacts

- Each pipeline stage has several stages
  - Some notebooks are used to analyze a pipeline, e.g.,
    - Process_hedge_fund_list
    - GP_LIn_contacts

Add sanity checks to make sure the data is well-formed or abort

Each transform can add debug information

- E.g., stats

- Mark the changed rows as `is_changed`

- Have a function to debug the data

There is a config controlling all the stages

There is a stage to serialize / deserialize

if serialize:

- Save

- Assert 0

else:

- Load

Each function should be idempotent

- Make a copy, modify, and return the data

- It's ok to keep assigning stages in sequence
  - Contact_df = f1(contact_df)
  - Contact_df = f2(contact_df)

- Once in a while we want to assign to a different var to split the computation
  - Contact_df = f1(contact_df)
  - Contact_df2 = f1(contact_df)
  - It's better in this case to use a meaningful name "cleaned_contact_df",
    rather than contact_df2

The expensive / slow phases are cached, either as Gsheet or through the caching
code

# Conventions
- Each SN search corresponds to a single Gsheet with multiple tabs for each step
  of the process

- Once the tab is complete we copy into a Gsheet corresponding to each vertical,
  e.g., "Cold outreach - VC"
  - The format/schema of each tab should be always the same so we can append
    data
  - We also add extra columns to the gsheet to represent the fact that a lead
    was acted upon, etc
  - In other words, the tab for VCs is called "VC - DB"

- Every time we do a YAMM campaign we
  - Add another tab tagged with the date
  - Copy-paste a set of rows from the "DB" tab

# Workflows
The goal is to create an email list from a LinkedIn search that represents a
"target customer" and then automatically send an email.

The target customers are:

- VCs

- Hospitality management

- Crypto Mining

- Real-estate management

- Online betting

- Oil and gas

- Banks

- Insurance

- ...

For each vertical we want to target the decision makers

GP's method

- LinkedIn/SN search

- PhantomBuster -> Hunter.io
  - This creates a Gsheet

- Filtering based on certain criteria
  - This adds another

- YAMM emails

- Email journey
  - [https://docs.google.com/document/d/1awxPYKZP23EMiOPJSb9jJDOLRh8vvfMZHftEh-i8nuI/edit](https://docs.google.com/document/d/1awxPYKZP23EMiOPJSb9jJDOLRh8vvfMZHftEh-i8nuI/edit)

# Automated Workflow
Make sure to give necessary permissions to Gsheets and folders for the code to
access them

- Input all API keys and credential files to access Gsheet via code - -
  [gsheet_into_pandas.how_to_guide.md](https://github.com/cryptokaizen/cmamp/blob/master/docs/coding/all.gsheet_into_pandas.how_to_guide.md)

<img
src="ai_outreach_process_figs/image28.png"
style="width:5.78646in;height:2.27635in" />

Look at all the phantom agents present and select our required agent

<img
src="ai_outreach_process_figs/image45.png"
style="width:6.5in;height:1.27778in" />

Input the required agent information from above to initialize the container

<img
src="ai_outreach_process_figs/image21.png"
style="width:5.64063in;height:2.81094in" />

Initialize the container, process the results and collect a dataframe

<img
src="ai_outreach_process_figs/image31.png"
style="width:6.5in;height:2.25in" />

Create a Gsheet in given folders with name 'agent name'+'search.export'

<img
src="ai_outreach_process_figs/image15.png"
style="width:6.5in;height:2.59722in" />

Extract emails from hunter io bulk email extraction

<img
src="ai_outreach_process_figs/image22.png"
style="width:6.5in;height:2.04167in" />

The results are stored in a new tab of same Gsheet named `hunter_results`.

- Below is how the file looks

<img
src="ai_outreach_process_figs/image51.png"
style="width:5.63021in;height:4.70833in" />

Computing some Stats!

<img
src="ai_outreach_process_figs/image18.png"
style="width:6.5in;height:2.01389in" />

-

## SN Query Automation
1.Benchmark Capital -
https://www.linkedin.com/sales/search/people?query=(spellCorrectionEnabled%3Atrue%2CrecentSearchParam%3A(id%3A3566053282%2CdoLogHistory%3Atrue)%2Cfilters%3AList((type%3AFUNCTION%2Cvalues%3AList((id%3A12%2Ctext%3AHuman%2520Resources%2CselectionType%3AEXCLUDED)%2C(id%3A16%2Ctext%3AMedia%2520and%2520Communication%2CselectionType%3AEXCLUDED)%2C(id%3A1%2Ctext%3AAccounting%2CselectionType%3AEXCLUDED)%2C(id%3A15%2Ctext%3AMarketing%2CselectionType%3AEXCLUDED)%2C(id%3A18%2Ctext%3AOperations%2CselectionType%3AEXCLUDED)%2C(id%3A13%2Ctext%3AInformation%2520Technology%2CselectionType%3AEXCLUDED)))%2C(type%3AREGION%2Cvalues%3AList((id%3A103644278%2Ctext%3AUnited%2520States%2CselectionType%3AINCLUDED)))%2C(type%3ASENIORITY_LEVEL%2Cvalues%3AList((id%3A320%2Ctext%3AOwner%2520%252F%2520Partner%2CselectionType%3AINCLUDED)))%2C(type%3ACURRENT_COMPANY%2Cvalues%3AList((id%3Aurn%253Ali%253Aorganization%253A18077%2Ctext%3AGreylock%2CselectionType%3AINCLUDED))))%2Ckeywords%3APartner)&sessionId=9crXfVSbSneO74cLlQLLNg%3D%3D

2. Greylock Partners -

https://www.linkedin.com/sales/search/people?query=(spellCorrectionEnabled%3Atrue%2CrecentSearchParam%3A(id%3A3566053282%2CdoLogHistory%3Atrue)%2Cfilters%3AList((type%3AFUNCTION%2Cvalues%3AList((id%3A12%2Ctext%3AHuman%2520Resources%2CselectionType%3AEXCLUDED)%2C(id%3A16%2Ctext%3AMedia%2520and%2520Communication%2CselectionType%3AEXCLUDED)%2C(id%3A1%2Ctext%3AAccounting%2CselectionType%3AEXCLUDED)%2C(id%3A15%2Ctext%3AMarketing%2CselectionType%3AEXCLUDED)%2C(id%3A18%2Ctext%3AOperations%2CselectionType%3AEXCLUDED)%2C(id%3A13%2Ctext%3AInformation%2520Technology%2CselectionType%3AEXCLUDED)))%2C(type%3AREGION%2Cvalues%3AList((id%3A103644278%2Ctext%3AUnited%2520States%2CselectionType%3AINCLUDED)))%2C(type%3ASENIORITY_LEVEL%2Cvalues%3AList((id%3A320%2Ctext%3AOwner%2520%252F%2520Partner%2CselectionType%3AINCLUDED)))%2C(type%3ACURRENT_COMPANY%2Cvalues%3AList((id%3Aurn%253Ali%253Aorganization%253A18077%2Ctext%3AGreylock%2CselectionType%3AINCLUDED))))%2Ckeywords%3APartner)&sessionId=9crXfVSbSneO74cLlQLLNg%3D%3D

**Differences:**

The primary difference between the two queries lies in the **CURRENT_COMPANY**
filter:

**Query 1:**
id%3Aurn%253Ali%253Aorganization%253A14467%2Ctext%3ABenchmark%2520%2CselectionType%3AINCLUDED

This corresponds to the company "Benchmark".

**Query 2:**
id%3Aurn%253Ali%253Aorganization%253A18077%2Ctext%3AGreylock%2CselectionType%3AINCLUDED

This corresponds to the company "Greylock".

**Common Parameters**

1.  SpellCorrectionEnabled: true

2.  RecentSearchParam: (id%3A3566053282%2CdoLogHistory%3Atrue)

3.  Filters:

  - FUNCTION: Excludes Human Resources, Media and Communication, Accounting,
      Marketing, Operations, and Information Technology.

  - REGION: Includes United States.

  - SENIORITY_LEVEL: Includes Owner / Partner.

1.  Keywords: Partner

2.  SessionId: Same in both queries.

## Create Flow to Find Decision Makers for a Firm

### Option 1
- We can have 1 SN query for multiple companies. Accuracy of data/ people
  extracted might decrease.

- Manual SN query generation (up to 10 companies)

- To Create Multi-Company Queries in Sales Navigator
  - Add Multiple Companies In the Current Company filter:

### Option 2
- We can loop on companies and their given SN queries

- We have 5 slots for SN extraction from PB

- PR:
  [https://github.com/causify-ai/cmamp/pull/11204](https://github.com/causify-ai/cmamp/pull/11204)

# Tools and Comparison
<table style="width:95%;">
<colgroup>
<col style="width: 22%" />
<col style="width: 34%" />
<col style="width: 38%" />
</colgroup>
<thead>
<tr>
<th>Tool</th>
<th>Features</th>
<th>Pricing</th>
</tr>
<tr>
<th>Phantombuster</th>
<th>Web scraping, automation, social media data extraction, API
integration</th>
<th>Free: Limited features; Pro: $30/month; Team: $70/month; Business:
$200/month</th>
</tr>
<tr>
<th>Apify</th>
<th>aWeb scraping, data extraction, automation, integration with APIs,
cloud infrastructure</th>
<th>Free: Up to 10,000 results/month; Personal: $49/month; Team:
$499/month</th>
</tr>
<tr>
<th>Octoparse</th>
<th>Visual web scraping, no coding required, cloud-based, data
extraction to various formats</th>
<th>Free: Limited features; Standard: $75/month; Professional:
$209/month; Enterprise: Custom pricing</th>
</tr>
<tr>
<th>ParseHub</th>
<th>Visual data extraction, no coding required, works with dynamic
websites</th>
<th>Free: 200 pages/month; Standard: $149/month; Professional:
$499/month</th>
</tr>
<tr>
<th>Scrapy Cloud</th>
<th>Web scraping, scheduling, data storage, API integration</th>
<th>Free: Limited resources; Professional: $90/month; Business:
$350/month</th>
</tr>
<tr>
<th>Zyte (formerly Scrapinghub)</th>
<th>Web scraping, data extraction, proxies, automatic retries</th>
<th>Free: Limited features; Developer: $99/month; Enterprise: Custom
pricing</th>
</tr>
<tr>
<th>Import.io</th>
<th>Data extraction, integration with APIs, scheduling, no coding
required</th>
<th>Free: Limited features; Essential: $299/month; Professional:
$499/month; Enterprise: Custom pricing</th>
</tr>
</thead>
<tbody>
</tbody>
</table>
