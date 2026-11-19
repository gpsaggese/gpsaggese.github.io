# GP's Startup Network Google Sheet

- This dir stores a local copy of the Google Sheet
  [GP's Startup Network](https://docs.google.com/spreadsheets/d/1Qa9g-j0NSdStV46W_dc85TMrwywIEPvJAhzV4Ymokb0/edit)
- Each tab of the Google Sheet is one CSV file
  `startup_resources/gp_resources.<ID>.<tab_name>.csv`
  - `<ID>` is the position of the tab in the Google Sheet, starting from 0
  - E.g., `startup_resources/gp_resources.0.FriendlyAngels.csv`
- The tab names have no spaces, so that `from_gsheet.py` and `to_gsheet.py` do
  a roundtrip
- The tabs are:

  | ID | Tab name              | CSV file                                                   |
  | :- | :-------------------- | :--------------------------------------------------------- |
  | 0  | `FriendlyAngels`      | `startup_resources/gp_resources.0.FriendlyAngels.csv`      |
  | 1  | `Accelerators`        | `startup_resources/gp_resources.1.Accelerators.csv`        |
  | 2  | `FriendlyVCs`         | `startup_resources/gp_resources.2.FriendlyVCs.csv`         |
  | 3  | `VC`                  | `startup_resources/gp_resources.3.VC.csv`                  |
  | 4  | `FriendlySalezBizDev` | `startup_resources/gp_resources.4.FriendlySalezBizDev.csv` |
  | 5  | `Grants`              | `startup_resources/gp_resources.5.Grants.csv`              |
  | 6  | `Credits`             | `startup_resources/gp_resources.6.Credits.csv`             |
  | 7  | `Support`             | `startup_resources/gp_resources.7.Support.csv`             |
- The `Focus` column of the `Grants` tab tells the AI grants, the crypto grants,
  and the other grants apart: `AI`, `Crypto`, or `General`
- `startup_resources/ai_grants.md` is the notes on the grant programs, not a tab

# Summary of Commands

| Action       | Command          | Direction                      |
| :----------- | :--------------- | :----------------------------- |
| **Download** | `from_gsheet.py` | Google Sheet to the CSV files  |
| **Upload**   | `to_gsheet.py`   | CSV files to the Google Sheet  |

# Download

- Run `startup_resources/download_gsheet.sh` to do the command below
  - Extra arguments go to `from_gsheet.py`, e.g., `--overwrite`
- Download all the tabs of the Google Sheet to the CSV files:
  ```bash
  > helpers_root/dev_scripts_helpers/google/from_gsheet.py \
    --url "https://docs.google.com/spreadsheets/d/1Qa9g-j0NSdStV46W_dc85TMrwywIEPvJAhzV4Ymokb0/edit" \
    --tabname _ALL_ \
    --output_file startup_resources/gp_resources.csv
  ```
- `--tabname _ALL_` downloads all the tabs, one CSV file for each tab
- `--output_file` is the prefix of the CSV files, not the name of one file
- The command fails if a CSV file already exists
  - Add `--overwrite` to replace the existing CSV files
  - This also deletes the local changes that are not uploaded

# Upload

- Run `startup_resources/upload_gsheet.sh` to do the command below
- Upload the CSV files to the tabs of the Google Sheet:
  ```bash
  > helpers_root/dev_scripts_helpers/google/to_gsheet.py \
    --input_file startup_resources/gp_resources.csv \
    --url "https://docs.google.com/spreadsheets/d/1Qa9g-j0NSdStV46W_dc85TMrwywIEPvJAhzV4Ymokb0/edit" \
    --tabname _ALL_ \
    --overwrite
  ```
- `--tabname _ALL_` uploads all the CSV files, one tab for each file
- `--input_file` is the prefix of the CSV files, not the name of one file
- `--overwrite` replaces the content of the tabs that already exist
  - Without it, the command fails if a tab already exists
  - This cannot be undone, except with the version history of Google Sheets
- The CSV files store only the values, so the formulas and the formatting of the
  Google Sheet are lost

# Definitions

- These definitions apply to the `Difficulty` field in the sections of
  `startup_resources/detailed_info.md`
- They also apply to the `Difficulty`, `Amount`, `AmountSize`, `GeoFocus`, and
  `Stage` columns of the CSV files `startup_resources/gp_resources.*.csv`
  - The angel, accelerator, VC, grant, and credit files have the
    `Difficulty`, `Amount`, `AmountSize`, and `GeoFocus` columns
  - Only the accelerator and `FriendlyVCs` files have the `Stage` column
  - Only the accelerator and `Grants` files have the `AcceptanceRate` column
    - In the `Grants` file it is filled for the NSF TIP programs only
    - Older rows keep the `Selectivity` in `Notes`
  - `gp_resources.4.FriendlySalezBizDev.csv` is a contact list and has none of
    these columns
  - `gp_resources.7.Support.csv` has advice and help, but no money, so it has
    none of the money columns

## Tabs

- Each CSV file is one tab of the Google Sheet, as listed at the top of this
  file
- Put a row in a tab by the type of money, not by who gives it
  - Equity (SAFE, convertible note, common stock, or a loan that converts):
    use the `VC` tab
  - Non-dilutive (grant, prize, or fellowship): use the `Grants` tab
    - Set `Focus` to `Crypto` for a crypto foundation grant, to `AI` for a
      grant for AI work, and to `General` for any other grant
    - Use `Credits` for cloud, GPU, or API credits
  - Advice, help, or support with no money and no credits: use the `Support`
    tab
    - E.g., an incubator, an advising office, a tech transfer office, or a
      directory of grants
  - A person who invests goes in `FriendlyAngels` or `FriendlyVCs`, not in a
    program tab
- Contents of each tab:
  - `0.FriendlyAngels` (People): angel investors and family offices in the
    network, who invest their own money
  - `1.Accelerators` (Programs): cohort accelerators that give cash and mentoring
    for equity, e.g., Alchemist, PearX, YCombinator
  - `2.FriendlyVCs` (People): partners and staff of VC firms in the network
  - `3.VC` (Programs): programs and funds that invest for equity and take
    applications
    - E.g., Y Combinator, AI Grant, USM Momentum Fund
    - It also holds state, regional, and university seed funds that take equity
      or convertible notes
      - E.g., TEDCO Builder Fund, AlphaLab, Virginia Venture Partners
  - `4.FriendlySalezBizDev` (Contacts): people for sales and business
    development, with no money columns
  - `5.Grants` (Programs): non-dilutive programs that are not credits
    - Federal, corporate, foundation, and student grants and prizes
    - State, regional, and university programs that give grants or training
      with some money, e.g., NSF I-Corps, TEDCO MII, VIPC CCF
    - Grants and fellowships from crypto foundations, e.g., Ethereum Foundation
    - `Focus` has three values: `AI` (a program for AI work, e.g., NSF AI
      Institutes), `Crypto` (a crypto foundation or ecosystem program), and
      `General` (any other topic, e.g., student funding or state programs)
    - `Category` has its own list of values for each `Focus`: the crypto rows
      use the ecosystem types, e.g., `L1/L2 Ecosystem Grants`, and the other
      rows use the funder types, e.g., `Federal / Government`
  - `6.Credits` (Programs): cloud, GPU, and API credits, e.g., NVIDIA Inception
  - `7.Support` (Programs): advice, help, and support with no money and no
    credits
    - Categories: `Advising`, `Proposal Help`, `Incubator`,
      `Regional Ecosystem`, `University Tech Transfer Office`, `Directory`
    - E.g., SBTDC, TEDCO Proposal Lab, bwtech@UMBC, UNC Innovate Carolina,
      Grantmatch
    - For a row whose amount is a value of the help, not cash, write the
      amount in `Notes`
    - It has no money columns, no `Difficulty`, and no `Deadline`
- A program with both equity and non-dilutive money goes in the tab of its main
  product
  - Write the other product in `Notes`, with the tab where it is listed
  - E.g., `TEDCO` is in `VC`, and its `Notes` says that MII and BII are in
    `Grants`
- If the type of money is not known, put the row in `Grants`
  - Write `assumed, not checked` in `Notes`
  - Move the row to `VC` when the program takes equity

## Schema

- The same data has the same column name in every CSV file
  - E.g., `Notes`, not `Notes / sources` in one file and `Notes` in another
- Column names are PascalCase, with no spaces and no punctuation
  - E.g., `ApplicationPage`, `AmountSize`, `GeoFocus`
- There are three row types
  - Each row type has one column set, in one fixed order
  - Every file of a row type has the columns of the set, in the same order
  - A column with no data in any row of a file is removed from the file
    - To fill it later, add it back in its place in the order of the set
- Row types and column sets:
  - **People**: the angels and the VCs, in CSV files 0 and 2
    - `Name`, `Organization`, `Email`, `LinkedIn`, `Website`, `Focus`, `Stage`,
      `GeoFocus`, `Amount`, `AmountSize`, `Difficulty`, `Notes`
  - **Contacts**: the people for sales and business development, in CSV file 4
    - `Name`, `Organization`, `Email`, `LinkedIn`, `SchoolAffiliation`, `Notes`
    - It has no money columns
  - **Programs**: the accelerators, grants, support offices, and programs that
    take applications, in CSV files 1, 3, 5, 6, and 7
    - `Name`, `Category`, `Eligibility`, `Focus`, `Stage`, `GeoFocus`, `Amount`,
      `AmountSize`, `Difficulty`, `AcceptanceRate`, `Website`, `ApplicationPage`,
      `Deadline`, `Notes`
    - `gp_resources.7.Support.csv` has only `Name`, `Category`, `Eligibility`,
      `GeoFocus`, `Website`, `ApplicationPage`, and `Notes`: the other columns
      have no data in it
- Order of the columns: who or what it is, then fit and money, then links, then
  `Notes`
- Meaning of the columns:

  | Column              | Meaning                                                    |
  | :------------------ | :--------------------------------------------------------- |
  | `Name`              | Person or people (People, Contacts) or program (Programs)  |
  | `Organization`      | Fund, firm, or company of the person                       |
  | `Email`             | Email address of the person                                |
  | `LinkedIn`          | URL of the LinkedIn profile                                |
  | `SchoolAffiliation` | School of the contact                                      |
  | `Category`          | Type of program, e.g., `Student Funding`, `Startup-only`   |
  | `Eligibility`       | Who can apply                                              |
  | `Focus`             | Sector or theme, e.g., `Fintech`, `AI`, `Crypto`           |
  | `Stage`             | Company stage: see `Stage` below                           |
  | `GeoFocus`          | Location: see `GeoFocus` below                             |
  | `Amount`            | Check or award size, as stated by the source               |
  | `AmountSize`        | Size level of `Amount`: see `AmountSize` below             |
  | `Difficulty`        | How hard it is to get the money or the meeting             |
  | `AcceptanceRate`    | Share of applicants accepted, as stated by the program     |
  | `Website`           | Home page of the firm or program                           |
  | `ApplicationPage`   | Page to apply or to submit a pitch                         |
  | `Deadline`          | Deadline, or the status of the call                        |
  | `Notes`             | Free text and sources                                      |

- Put each fact in its own column
  - E.g., the check size goes in `Amount`, the sector in `Focus`, the stage in
    `Stage`, the location in `GeoFocus`
  - Put a fact that has no column, e.g., `$100K+ MRR`, in `Notes`
- To add a column, add it to all the files of the row type
- Old column names were renamed to the names above:

  | Old name                  | New name            | CSV files     |
  | :------------------------ | :------------------ | :------------ |
  | `Application page`        | `ApplicationPage`   | 1, 3, 5, 6    |
  | `Acceptance rate`         | `AcceptanceRate`    | 1             |
  | `Who it's for`            | `Eligibility`       | 3, 5, 6       |
  | `Company`, `Firm`         | `Organization`      | 0, 2          |
  | `Notes / sources`         | `Notes`             | 0, 2          |
  | `LinkedIn Profile Link`   | `LinkedIn`          | 4             |
  | `School Affiliation`      | `SchoolAffiliation` | 4             |
  | `First Name`, `Last Name` | `Name`              | 4             |

## Difficulty

- `Difficulty` rates how hard it is to get the money or the meeting
  - Rate the odds and the effort for a typical startup or lab
  - Use four levels: easy, medium, hard, very hard
- Levels:
  - **easy**: Open to almost any applicant, with no competitive selection
    - Rolling or always open, short form, no gatekeeper
    - E.g., NAIRR Pilot, NVIDIA Inception, Starknet Seed Grants
  - **medium**: Open call that needs a real application and a good fit
    - Competitive, or limited to a restricted pool of applicants
    - E.g., Google Academic Research Awards, NSF ExpandAI (MSIs only)
  - **hard**: Highly selective, or access is limited
    - Acceptance rate ~2-10%, a fixed deadline with heavy paperwork, a warm intro,
      an invitation, or a program that is between rounds
    - E.g., NSF SBIR/STTR, Schmidt Sciences AI2050 Fellowship, Alchemist (~3%)
  - **very hard**: Almost no chance without a special edge
    - Acceptance rate under ~2%, a large consortium, or a person who rarely
      invests
    - E.g., NSF National AI Research Institutes, Y Combinator (~1%), A16z Speedrun
      (under 0.4%)
- The `Difficulty` column of every CSV file uses one of these labels:
  `Easy`, `Medium`, `Hard`, `Very hard`, `N/A`
  - It is the column right after `AmountSize` in each CSV file
  - Angels and VCs use the same labels as programs
  - For a person, the level is how hard it is to get a reply and a check
  - E.g., `Easy`: Rajeev Madhavan, who already invested $100K via SAFE
  - E.g., `Medium`: UMD Dingman Center Angels, with a formal application and pitch
  - E.g., `Hard`: Gokul Rajaram, a prolific angel with a fund
  - E.g., `Very hard`: Naval Ravikant, who made few new investments recently
- Old labels were renamed to the labels above:

  | Old label   | New label   |
  | :---------- | :---------- |
  | `Low`       | `Easy`      |
  | `Moderate`  | `Medium`    |
  | `High`      | `Hard`      |
  | `Very high` | `Very hard` |

- Special values:
  - A range (`Moderate–Hard`, `Easy–moderate`) means the level depends on the
    track or the tier
    - Use the level of the track you apply to
    - If no track is chosen, use the harder level
  - A level with a status (`Hard: between rounds`) keeps the level
    - Put the status in the `Deadline` or `Notes` field
  - `N/A` means that the program is closed to outsiders, paused, or
    discontinued, or that the row is a directory or a placeholder, not a program
    - Do not assign a level
    - Put the reason in the `Deadline` or `Notes` field

## AmountSize

- `AmountSize` classifies the size of the money that one winner receives
  - It is read from the `Amount` column, which grants, angels, VCs, and
    accelerators all use
  - It is the column right after `Amount` in each CSV file
- Levels, in USD:

  | Level      | Amount per winner | Examples                               |
  | :--------- | :---------------- | :------------------------------------- |
  | Very small | under $10K        | Women Founders Grant ($5K)             |
  | Small      | $10K to $100K     | Z Fellows ($10K), EF PhD Fellowship    |
  | Medium     | $100K to $500K    | Techstars ($220K), Thiel Fellowship    |
  | Large      | $500K and above   | Y Combinator ($500K), Hult Prize ($1M) |

- The `AmountSize` column of every CSV file uses one of these labels:
  `Very small`, `Small`, `Medium`, `Large`, `N/A`
- The lower bound is included in each level
  - E.g., $10K is `Small`, $100K is `Medium`, $500K is `Large`
- Rules for the amount:
  - Use one award or one check, not the total pool
    - E.g., "$2M pool per round" is not an award size
    - E.g., a VC fund size (`$200M Fund III`) is not a check size
  - Use the typical first award or check if the source states one
    - If the source gives only a range, use the lower end
    - E.g., `$100K–$250K` is `Medium` and `$250K–$1M+` by phase is `Medium`
  - Count credits and in-kind support at the dollar value stated
    - E.g., `Up to ~$150K in Azure credits` is `Medium`
    - Say in the notes that the amount is credits and not cash
  - Convert other currencies to USD at a rough rate
  - Use `N/A` when the source says `Varies`, `Not stated`, or gives no figure
  - Use `N/A` when the program is closed, paused, or discontinued

## GeoFocus

- `GeoFocus` says where the applicant must be based, or must work, to use the
  opportunity
  - Read it as: "Can a Maryland company use it, and must it move?"
  - For an angel or a VC, it says where they invest
  - It is not the sector: put the sector in `Focus`
- Use the location the applicant needs, not the location of the funder
  - E.g., the Ethereum Foundation is based in Switzerland, but its grants are
    `Global`
- Values:

  | Value                 | Meaning                                       | Examples                      |
  | :-------------------- | :-------------------------------------------- | :---------------------------- |
  | `Global`              | No location limit. Remote is fine             | Hello Tomorrow, Hult Prize    |
  | `US`                  | US entity or US-based applicant only          | NSF SBIR/STTR, Dorm Room Fund |
  | `US: <state or city>` | One US place only                             | `US: MD`, `US: Baltimore`     |
  | `Non-US`              | Applicant must be outside the US              | USAID Program                 |
  | `<country>`           | One non-US country only                       | `UAE`                         |
  | `<country>: <place>`  | One place in a non-US country                 | `UAE: Dubai`                  |
  | `On site: <place>`    | Applicant works at the place, from any origin | YC: `On site: San Francisco`  |
  | `N/A`                 | The source does not say                       |                               |

- Rules:
  - Use `Global` only when the source states no limit
    - A permissionless on-chain program (any builder can apply) is `Global`
      unless the source names a limit
  - Use `N/A` when the source is silent
    - Do not leave the cell empty
    - Do not guess from the sponsor name
  - List several places with `;`
    - E.g., `US: MD; US: DC; US: VA`
    - After `On site:` the applicant picks one place, e.g., Techstars:
      `On site: Baltimore; Boston; London`
  - Use the narrowest true value
    - E.g., `US: MD`, not `US`, for a Maryland-only program
  - If the tracks have different locations, use the track you apply to
    - If no track is chosen, list all the places
  - Start every item with one of the values above, so that a filter on the first
    word works
- Current state of the CSV files:
  - `GeoFocus` is a column in the angel, accelerator, VC, grant, and credit
    files
    - It was `Geographical focus` in the accelerator file and `Geo` in the VC file
  - `gp_resources.4.FriendlySalezBizDev.csv` is a contact list and has no
    `GeoFocus` column
  - Many rows are `N/A`: the angel, VC, and grant files do not state the
    location for those rows
    - Fill them from the program or fund page

## Stage

- `Stage` says the stage of the companies that the funder or the program accepts
  - It is not the stage of the funder
  - It is not the check size: put the size in `Amount`
  - It is not the sector: put the sector in `Focus`
- Values, from the earliest stage to the latest:

  | Value       | Meaning                                              |
  | :---------- | :--------------------------------------------------- |
  | `Idea`      | No product yet, only an idea or a team               |
  | `Pre-seed`  | First product or prototype, first small round        |
  | `Seed`      | Product in the market, early traction                |
  | `Series A`  | Repeatable traction, scaling the first model         |
  | `Series B`  | Scaling a proven model                               |
  | `Series C`  | Expansion of a company with a large customer base    |
  | `Growth`    | Growth equity in a mature or profitable company      |

- Other values:
  - `Early-stage`: the source says "early-stage" and names no round
  - `Later stage`: the source says "later stage" and names no round
  - `Any`: no stage limit
  - `N/A`: the source does not say
- A range uses `to`, with the earlier stage first
  - E.g., `Pre-seed to Seed`, `Seed to Series A`, `Series A to Series C`
- An open range uses `onward`
  - E.g., `Idea onward`
- Rules:
  - Use the stage names of the source, mapped to the closest value
    - E.g., `Seed-at-scale` is `Seed`, and `Seed and Series A` is
      `Seed to Series A`
  - Put only the stage in `Stage`
    - Move the check size to `Amount`, the sector to `Focus`, and a trait of the
      company, e.g., `$100K+ MRR`, to `Notes`
  - Use `N/A` when the source is silent
    - An empty cell means that nobody has reviewed the row yet
- Current state of the CSV files:
  - `Stage` is a column only in the accelerator file and in
    `gp_resources.2.FriendlyVCs.csv`
    - The other files had no `Stage` data, so the column was removed from them
  - It is filled for 5 of 8 rows of the accelerator file and for 9 rows of the
    VC file
  - It is empty in the other rows: fill them from the program or fund page
