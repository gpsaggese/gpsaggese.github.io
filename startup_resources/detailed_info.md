# Search engines

https://www.grants.gov/
https://www.thegrantportal.com/
https://maryland.grantwatch.com/
https://www.instrumentl.com/
https://www.linkedin.com/posts/ymlnv_i-found-30-web3-grant-programs-funding-builders-activity-7486356023230844929-_fHG/
https://coach-carter.beehiiv.com/archive

- The last link is the archive of The Scouting Report, a weekly newsletter
  of student and startup funding opportunities (grants, competitions,
  accelerators, fellowships, credits)
- Use it as a source of links: each issue lists the sponsor, amount,
  deadline, and application URL
- Issues #011-#018 (Oct 6 - Nov 17, 2026) are captured in the sections
  below, marked `Source: The Scouting Report #NNN`. Issue #014 and the
  issues before #011 (archive pages 2 and 3+) are not reviewed yet
- Amounts and dates are as stated in the newsletter. Confirm them on the
  sponsor page before applying

# Definitions

- These definitions apply to the `Difficulty` field in the sections below
- They also apply to the `Difficulty`, `Amount`, `AmountSize`, `GeoFocus`, and
  `Stage` columns of the CSV files `startup_resources/gp_resources.*.csv`
  - The angel, accelerator, VC, grant, and credit files have the
    `Difficulty`, `Amount`, `AmountSize`, and `GeoFocus` columns
  - Only the accelerator and `FriendlyVCs` files have the `Stage` column
  - `gp_resources.4.FriendlySalezBizDev.csv` is a contact list and has none of
    these columns
  - `gp_resources.7.Support.csv` has advice and help, but no money, so it has
    none of the money columns

## Tabs

- Each CSV file is one tab of the Google Sheet, as listed in
  `startup_resources/README.md`
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

# Detail

## Federal / Government

### NSF National AI Research Institutes

- **Website**:
  <https://www.nsf.gov/funding/opportunities/national-artificial-intelligence-research-institutes>
- **Submission**: <https://www.research.gov/> (proposals prepared and submitted
  through Research.gov per the program solicitation)
- **Award size**: ~$20M each
- **Eligibility**: Multidisciplinary, multi-stakeholder university-led teams
- **Status**: The program page shows no upcoming due dates: waiting on a new
  solicitation. This is the window to watch
- **Deadline**: None open. No active solicitation as of Aug 2026, no announced
  reopening
- **Difficulty**: Very hard. Requires assembling a large consortium

### NSF ExpandAI (NSF 23-506)

- **Website**:
  <https://www.nsf.gov/funding/opportunities/expandai-expanding-ai-innovation-through-capacity-building/506090/nsf23-506/solicitation>
- **Submission**: N/A. Proposers must first submit a Concept Outline and receive
  an official NSF invitation before a full proposal can be submitted; no direct
  public application page
- **Award size**: Track 1 (Capacity Building Pilots) up to $400,000 over two
  years; Track 2 (Partnerships) connects MSIs with existing AI Institutes. $7.5M
  allocated for FY 2026
- **Eligibility**: Minority-serving institutions only: HBCUs, HSIs, Alaska Native
  Serving Institutions, Predominantly Black Institutions
- **Deadline**: Closed/archived. NSF's solicitation page is marked "Archived
  funding opportunity," no open call or announced reopening
- **Difficulty**: Moderate. Restricted eligibility means a smaller applicant pool
- **Precedent**: Clark Atlanta University used a Track 2 award to create the first
  graduate AI degree programs at an HBCU

### AI Forge (DARPA + NSF)

- **Website**: <https://www.darpa.mil/research/programs/ai-forge>
- **Submission**:
  <https://sam.gov/workspace/contract/opp/a077021d13f843a5aa5f0baee00b1097/view>
  (RFI response submitted via SAM.gov)
- **Award size**: $750,000–$3M per Project Venture, one-year terms
- **Focus areas**: AI interpretability, AI control, adversarial robustness
- **Eligibility**: US universities and military service academies only. One
  authorized submission per institution
- **Status**: Announced June 1, 2026. Administered by a nonprofit launching
  summer 2026. The initial RFI has closed
- **Deadline**: RFI (DARPA-SN-26-80) closed June 22, 2026. No formal Project
  Ventures proposal deadline published yet; watch for the nonprofit's launch
- **Note**: IP is expected to be shared across forum participants, preferably
  under open-source licensing

### DOE Genesis Mission AI

- **Website**:
  <https://www.energy.gov/undersecretaryforscience/genesis-mission/genesis-mission>
- **Submission**:
  <https://simpler.grants.gov/opportunity/0228b895-9cb3-4160-8acc-58709e75c3c7>
- **Award size**: Varies by funding opportunity announcement
- **Eligibility**: Depends on the specific FOA: universities, nonprofits, and
  national labs are typical; some announcements are lab-restricted
- **Process**: Pre-applications or letters of intent, then full proposals via
  Grants.gov or PAMS, then multi-stage review
- **Deadline**: FY26 Phase I/II deadlines (through May 19, 2026) have passed.
  Next upcoming: December 17, 2026, 11:59 PM ET, for Phase II applications from
  FY26 Phase I awardees
- **Difficulty**: Hard. Budget significant lead time for PAMS registration and
  institutional sign-off if you have no prior DOE relationship

### NAIRR Pilot

- **Website**: <https://nairrpilot.org/about>
- **Submission**: <https://submit-nairr.xras.org/>
- **What you get**: Free computational, data, and training resources: GPUs,
  pre-trained models, AI-ready datasets. Not cash
- **Eligibility**: US-based researchers and educators at academic institutions
  (including grad students with a faculty support letter), nonprofits, federal
  agencies, FFRDCs, state/local/tribal agencies, and startups with federal
  grants
- **Tracks**: Start-Up Projects (3 months, one resource, decision in ~2 weeks),
  full Resource Allocations (12 months), education calls, deep partnerships
- **Deadline**: Rolling; open until resources are exhausted
- **Difficulty**: Easiest entry point on this list. Rolling, open until resources
  are exhausted
- **Requirement**: Project results must be open and shared with the community. PI
  name, affiliation, title, and abstract are posted publicly

### NSF CISE Core Programs (incl. US–Italy AI collaboration track)

- **Website**:
  <https://new.nsf.gov/funding/opportunities/computer-information-science-engineering-core/nsf23-561/solicitation>
- **Submission**: <https://www.research.gov/> or <https://grants.gov/>
- **What it is**: NSF 23-561, the standard CISE core-programs solicitation, with a
  bilateral track for joint US–Italy AI proposals
- **Deadline**: NSF 23-561 is archived, superseded by NSF 25-543 "Future CoRe."
  Current target dates: 2nd Thursday of September annually (next: Sept 10,
  2026) and 1st Thursday of February annually (next: ~Feb 4, 2027); proposals
  also accepted anytime. The US–Italy AI Lead Agency DCL (NSF 24-055) is itself
  archived/closed, no fixed deadline
- **Difficulty**: Hard. Broad, competitive core program

### NSF 24-554: Artificial Intelligence, Formal Methods, and Mathematical Reasoning

- **Website**:
  <https://new.nsf.gov/funding/opportunities/artificial-intelligence-formal-methods/nsf24-554/solicitation>
- **Submission**: <https://grants.gov/>
- **Background**: Webinar slides at
  <https://nsf-gov-resources.nsf.gov/files/mps-aiming-webinar-slides_2024-04-11-508.pdf>
- **Deadline**: Active solicitation, but all three published full-proposal
  deadlines (June 3, 2024; Feb 5, 2025; Feb 5, 2026) have passed. No deadline
  beyond Feb 5, 2026 published yet
- **Difficulty**: Hard

### Advancements in Artificial Intelligence for Science

- **Website**: <https://www.grants.gov/search-results-detail/352320>
- **Submission**: <https://grants.gov/>
- **Deadline**: Closed. This is a DOE Office of Science FOA (DE-FOA-0003264);
  full-application deadline was May 21, 2024 (pre-application March 19, 2024).
  No new cycle found
- **Difficulty**: Hard

### ACED: Accelerating Computing-Enabled Scientific Discovery

- **Website**:
  <https://new.nsf.gov/funding/opportunities/aced-accelerating-computing-enabled-scientific>
  (also indexed at <https://www.grants.gov/search-results-detail/352340>)
- **Submission**: <https://grants.gov/>
- **Deadline**: NSF's site marks this solicitation (NSF 24-541) "archived," yet
  the document itself still lists a Track II (Discovery Proposals) deadline of
  September 17, 2026 (Track I and a prior Track II deadline have both passed).
  Conflicting signals — verify directly with the NSF program officer before
  relying on the Sept 17, 2026 date
- **Difficulty**: Hard

### NSF 23-518: Advanced Computing Systems & Services

- **Website**:
  <https://new.nsf.gov/funding/opportunities/advanced-computing-systems-services-adapting-rapid>
- **Submission**: <https://www.research.gov/> or <https://grants.gov/>
- **What it is**: "Adapting to the Rapid Evolution of Science and Engineering
  Research"
- **Deadline**: Archived, superseded by NSF 24-583, which itself states it is
  "awaiting a new solicitation and is not currently accepting proposals." No
  open deadline currently published
- **Difficulty**: Hard

### NSF 24-569: Mathematical Foundations of Artificial Intelligence

- **Website**:
  <https://new.nsf.gov/funding/opportunities/mathematical-foundations-artificial-intelligence/nsf24-569/solicitation>
- **Submission**: <https://www.research.gov/> or <https://grants.gov/>
- **Deadline**: October 9, 2026 (recurring annual full-proposal deadline: Oct 10
  2024; Oct 10 2025; Oct 9 2026)
- **Difficulty**: Hard

### NSF 24-111 Dear Colleague Letter: Planning Grants for AI-Ready Test Beds

- **Website**: <https://www.nsf.gov/pubs/2024/nsf24111/nsf24111.jsp>
- **Submission**: N/A standing portal. Inquiries/proposals via
  AI-Ready_testbeds@nsf.gov per the DCL
- **What it is**: DCL inviting planning-grant proposals to create AI-ready test
  beds
- **Deadline**: Closed. Was November 13, 2024, 5:00 PM submitting organization's
  local time (concept outline required beforehand). No new call found
- **Difficulty**: Hard

### Resource: NSF SBIR/STTR Fast-Track Office Hours

- **Website**: <https://www.sbir.gov/about>
- **Registration**:
  <https://nsf.zoomgov.com/webinar/register/WN_i0kwrKCVTZOey_XKMoc6ow#/registration>
- **What it is**: Informational webinar on the NSF SBIR/STTR Fast-Track pilot, not
  a grant itself. Pair with the NSF SBIR/STTR entry below
- **Deadline**: N/A, informational resource, not a grant

## Corporate / Foundation

### Google Academic Research Awards

- **Website**:
  <https://research.google/programs-and-events/google-academic-research-awards/>
- **Submission**:
  <https://google.submittable.com/submit/b1fa9f63-c58c-4a70-a838-ec01ed6083c1/google-academic-research-awards-gara-application>
- **Award size**: Up to $100,000
- **Eligibility**: Professors at degree-granting institutions who advise students
  and do research in technology/computing
- **Cadence**: Rolling cycles; a recent window closed July 23, 2026
- **Deadline**: Rolling; most recent window closed July 23, 2026, next cycle TBD
- **Difficulty**: Moderate. Broad program, thousands of applicants per cycle

### Google.org Impact Challenge: AI for Science

- **Website**: <https://www.google.org/impact-challenges/ai-science/>
- **Submission**:
  <https://googlenewaccount2.submittable.com/submit/2eacc5bf-5e78-4f06-b14e-546c8a895981/google-org-impact-challenge-ai-for-science>
- **Award size**: $500K–$3M per grantee, $30M total pool
- **Eligibility**: Nonprofits, social enterprises, and academic institutions using
  AI for health or climate science
- **Extras**: Technical mentorship from Google AI researchers, Google Cloud
  compute, multi-month accelerator program
- **Status**: 2026 round deadline (May 1) has passed. Watch for the next round
- **Deadline**: Closed. 2026 round deadline was May 1, 2026; next round TBD
- **Difficulty**: Hard. Global open call, highly competitive

### Amazon Research Awards

- **Website**: <https://www.amazon.science/research-awards>
- **Submission**: <https://home.academic-research-portal.amazon.science/login>
- **Award size**: Varies by track. AWS Agentic AI: ~$70K unrestricted funds plus
  ~$50K AWS credits. Robotics: up to $50K plus $50K credits. Trainium: $50K–
  $250K+ in credits
- **Eligibility**: Faculty/researchers at universities or research institutions,
  worldwide
- **Cadence**: Two calls per year (Spring/Fall). Spring 2026 window closed May 6,
  decisions expected August
- **Deadline**: Fall 2026 window not yet announced; Spring 2026 closed May 6,
  2026
- **Difficulty**: Moderate

### Cisco Research: University Gift Funding

- **Website**: <https://research.cisco.com/research-funding>
- **Submission**: <https://cisco-research.versaic.com>
- **Award size**: Typically $50,000–$100,000 per project, unrestricted gift funds;
  multi-year follow-on possible for high-impact teams
- **Eligibility**: PI must be full-time faculty or research staff at a university
  or research institution (students/adjuncts may be co-investigators only)
- **Focus areas**: Open rolling RFPs, currently AI/ML, generative AI, embodied AI,
  distributed systems security for GenAI
- **Cadence**: Rolling; current cycle runs through December 31, 2026
- **Deadline**: Rolling through December 31, 2026
- **Precedent**: UMD ECE's Ang Li received $75K for "Federated Fine-tuning of
  Black-box Large Language Models via Proxy Models at the Edge"
- **Difficulty**: Moderate

### Schmidt Sciences: AI2050 Fellowship

- **Website**: <https://ai2050.schmidtsciences.org/fellows/>
- **Submission**: N/A. Fellows are chosen via a closed nomination process; the
  program page states there is no application
- **Award size**: $300,000–$500,000 over 3 years
- **Eligibility**: Early-career researchers worldwide working on AI safety,
  alignment, robustness, or beneficial deployment
- **Status**: 2026 cohort applications have closed; the program recurs annually
- **Deadline**: Closed for 2026. No public application (closed nomination
  process); recurs annually
- **Difficulty**: Hard. Highly selective

### Schmidt Sciences: Science of Trustworthy AI RFP

- **Website**: <https://www.schmidtsciences.org/trustworthy-ai-research-agenda/>
- **Submission**:
  <https://schmidtsciences.smapply.io/prog/science_of_trustworthy_ai_rfp_2026/>
- **Award size**: Tier 1 up to $1M for focused work; Tier 2 $1M–$5M+ for
  multi-year interdisciplinary teams
- **Eligibility**: Individual researchers, teams, or multi-institution
  collaborations at universities, national labs, or nonprofits, worldwide
- **Extras**: GPU/CPU compute and storage, software engineering support, frontier
  model API credits
- **Status**: 2026 RFP deadline was May 17. Expect a future round
- **Deadline**: Closed. 2026 RFP deadline was May 17, 2026; no next round
  announced
- **Difficulty**: Hard

### Coefficient Giving (formerly Open Philanthropy): Technical AI Safety

- **Website**:
  <https://coefficientgiving.org/funds/navigating-transformative-ai/request-for-proposals-technical-ai-safety-research/>
- **Submission**: <https://airtable.com/app1NkoJMO9OvaGYS/pagDeq1lbSlmYdJRG/form>
- **Award size**: Exploratory grants under $100K; standard $100K–$1M; large grants
  $1M–$10M
- **Eligibility**: Academic researchers, grad students, independent researchers,
  and nonprofits with demonstrated ML experience
- **Process**: Start with a 300-word expression of interest via the general "Apply
  for Funding" pathway
- **Status**: The dedicated $40M Technical AI Safety RFP has closed, but general
  AI-safety funding recommendations remain open
- **Deadline**: Rolling (general funding pathway); dedicated $40M RFP has closed,
  no next round announced
- **Difficulty**: Hard

### OpenAI Superalignment Fast Grants

- **Website**: <https://openai.com/blog/superalignment-fast-grants>
- **Submission**: <https://airtable.com/appnIXmOlWAJBzrJp/paghnoKL6EHiKmKbf/form>
- **What it is**: Fast-turnaround grants for weak-to-strong generalization /
  superalignment research, originally announced 2023
- **Status**: Check the Airtable form directly for whether this round is still
  accepting applications
- **Deadline**: Closed. Deadline was Feb 18, 2024; the OpenAI Superalignment team
  that ran it was dissolved in May 2024, no new round announced
- **Difficulty**: Hard

### a16z Open Source AI Grant

- **Website**: <https://a16z.com/announcing-our-latest-open-source-ai-grants/>
- **Submission**: N/A. No public apply link found; past rounds appear to be
  sourced by a16z directly rather than via open intake
- **Deadline**: N/A, no public application process; recipients sourced/selected
  directly by a16z (most recent batch mid-2025)
- **Difficulty**: Hard. No standing open portal

### Microsoft AI for Accessibility

- **Website**: <https://www.microsoft.com/en-us/ai/ai-for-accessibility-grants>
- **Submission**: <https://sway.cloud.microsoft/JE0PoxRpU6FQwmMy>
- **What it is**: Grants (cash, Azure credits, technical support) for AI projects
  that improve accessibility for people with disabilities
- **Deadline**: Not confirmed on Microsoft's official page; third-party
  aggregators conflict. Treat as rolling/periodic rounds with no confirmed
  official date
- **Difficulty**: Moderate

### Llama Impact Grants (Meta)

- **Website**: <https://llama.meta.com/llama-impact-grants/>
- **Submission**: N/A. No public apply link found on the page
- **What it is**: Meta grants for open-source Llama-based projects with social
  impact
- **Deadline**: Closed. Global Llama 3.1 cohort deadline was Nov 22, 2024; a
  regional Africa track closed June 27, 2025. No evidence of an open 2026 cohort
- **Difficulty**: Hard

### Google.org: Digital Futures Project

- **Website**:
  <https://blog.google/outreach-initiatives/google-org/launching-the-digital-futures-project-to-support-responsible-ai/>
- **Submission**: N/A. Announcement post only, no public apply link found
- **What it is**: Google.org initiative funding responsible-AI research and
  safeguards
- **Deadline**: N/A, announcement/partnership-based, no public application
- **Difficulty**: Hard. Appears invitation/partnership-based

### Artists + Machine Intelligence Grants (Google)

- **Website**: <https://experiments.withgoogle.com/ami-grants>
- **Submission**: <https://labs.google/submit-experiment>
- **What it is**: Google program funding artists collaborating with engineers on
  AI-driven creative experiments, not a traditional research grant
- **Deadline**: Rolling; submitted via labs.google/submit-experiment, no fixed
  cycle found
- **Difficulty**: Hard

## University Internal Seed Grants (Not Open to Outside PIs)

Listed for reference: shows the "internal AI seed grant" pattern that other
universities, including UMD, could replicate.

### Stanford HAI Seed Grants

- **Website**:
  <https://seedfunding.stanford.edu/opportunities/2023-institute-human-centered-artificial-intelligence-seed-grants>
- **Submission**: N/A. No public apply link found; internal Stanford program
- **Eligibility**: Stanford faculty only
- **Deadline**: N/A, internal Stanford program only
- **Difficulty**: N/A, not open to outside applicants

### Vanderbilt Generative AI Seed Grant

- **Website**:
  <https://www.vanderbilt.edu/research-development-and-support/internal-funding-programs/generative-ai-seed-grant/>
- **Submission**: N/A. No public apply link found; internal Vanderbilt program
- **Eligibility**: Vanderbilt faculty only
- **Deadline**: N/A, internal Vanderbilt program only
- **Difficulty**: N/A, not open to outside applicants

## International: UAE

Outside the US, the UAE is the most active government-backed AI-grant ecosystem,
split between Abu Dhabi (ATRC/ASPIRE, TII, MBZUAI) and Dubai (Dubai Future
Foundation). Most tracks require a UAE-based host institution or PI; a few
(small individual grants, fellowships) are open directly to applicants
worldwide

### ASPIRE / Advanced Technology Research Council (ATRC): Abu Dhabi

- **Website**: <https://www.aspireuae.ae/>
- **Submission**: N/A. AARE/AYIA/VRI/ViP grants are awarded via periodic
  competitive calls announced to Abu Dhabi research institutions, not a standing
  public application portal; inquiries via hello@aspireuae.ae
- **What it is**: ATRC's program-management arm. Four standing tracks: ASPIRE
  Award for Research Excellence (AARE), ASPIRE Young Investigator Award (AYIA),
  Virtual Research Institutes (VRIs), and Visiting International Professorship
  (ViPs). Plus challenge programs like ASPIRE Vantage (2026: autonomous systems,
  front-loaded co-development funding)
- **Track record**: ~$81M (AED 297M) awarded across 114 research projects to
  date; ~$2.7M+ (AED 10M+) used specifically to build international research
  networks
- **Eligibility**: VRIs and ViPs are explicitly designed to bring in international
  researchers/institutes as partners or visiting appointments. AARE/AYIA
  typically route through a UAE host institution (e.g., Khalifa University, NYU
  Abu Dhabi)
- **Focus areas**: ICT (incl. AI), aerospace, energy, environment, food &
  agriculture, health, manufacturing
- **Deadline**: No current public deadline. AARE/AYIA run periodic annual calls
  (no 2026 cycle announced yet); ViP is rolling (EOI anytime, invited proposals
  due within 4 months); VRI has no published deadline; ASPIRE Vantage 2026 is a
  multi-stage challenge — check ChallengeON (challengeon.atrc.ae) for live stage
  dates
- **Difficulty**: Hard. No public rolling portal; entry is usually via an existing
  UAE partner institution or direct outreach to hello@aspireuae.ae

### Sandooq Al Watan (SWARD, Fellowship, Grants programmes)

- **Website**: <https://sandooqalwatan.ae/>
- **Submission**: <https://www.researcher.ae/>
- **SWARD programme**: Co-funded, institution-level research. Example: ~$272K
  (AED 1M) Khalifa University/UAEU project on phase-change-material nanowires
  for AI/in-memory computing. Explicitly supports "international and Emirati
  experts from universities and research institutions in the country"
- **Grants programme**: Small individual grants, ~$1.4K–$6.8K (AED 5,000–25,000),
  open to any innovator, academic, entrepreneur, or scientist with a research
  idea
- **Fellowship programme**: Separate track for individual researchers
- **Portal**: researcher.ae, Sandooq Al Watan's national research-support platform
- **Deadline**: No single fixed deadline published. SWARD runs ~3 periodic
  cycles/year (Circular Economy, Climate Change, Social Sustainability) with no
  confirmed next date; Grants and Fellowship programmes (via researcher.ae)
  appear continuous/rolling
- **Difficulty**: Moderate for the small Grants track; harder for SWARD, which
  needs a UAE institutional host

### Dubai Future Foundation: Dubai RDI Grant Initiative

- **Website**: <https://dubairdi.ae/>
- **Submission**: <https://dubairdi.ae/application/>
- **Award size**: ~$136K–$408K (AED 500,000–1.5M) per project (annual budget
  capped at ~$136K/AED 500K), 1–3 year projects
- **Eligibility**: PI must be fully affiliated with a Dubai-based HEI, research
  centre, or institute, and hold a PhD. International co-investigators are
  explicitly permitted, but at least 50% of grant funds must be spent in Dubai
- **Focus areas**: Cognitive Cities and Health & Life Sciences, with AI, robotics,
  IoT, quantum computing, and AR/VR as cross-cutting technologies
- **Related track**: International Collaborative Research Funding runs bilateral
  calls with a foreign funder (first partner: South Korea's National Research
  Foundation): a model a US collaboration could piggyback on if a similar
  bilateral call opens with a US funder
- **Also**: Dubai Resilience Challenge Fund (TRL 4–7, commercialization-focused)
  and Dubai Future Accelerators (no equity taken)
- **Deadline**: Periodic annual rounds; the 2nd Research Projects Funding round
  closed June 13, 2025. No verified next-round deadline published as of Aug
  2026 (unconfirmed secondary reports cite a June 9, 2026 LOI date, already
  passed if accurate)
- **Difficulty**: Moderate–Hard. Needs a Dubai-based PI/host; a US researcher
  joins as international co-investigator

### Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)

- **Website**: <https://mbzuai.ac.ae/ugrip/>
- **Submission**: <https://apply.mbzuai.ac.ae/ApplicantPortal>
- **Primarily funds its own grad students**: fully funded MSc/PhD scholarships,
  plus the Undergraduate Research Internship Program (UGRIP), fully funded, open
  to undergrads worldwide, runs roughly May–June
- Faculty-level research funding flows mainly through philanthropic partners:
  Google.org gave $1M in February 2026 to fund an MBZUAI-led research initiative
- Not a standing open grants program for outside PIs, but worth watching for
  future funder partnerships, and as a potential UAE-side collaborator for joint
  proposals (e.g., to Dubai RDI or ASPIRE)
- **Deadline**: 2026 UGRIP cycle closed Feb 28, 2026, 5pm GST (program ran
  June 1–26, 2026); no 2027 deadline published yet. Not a standing grants
  program for outside PIs
- **Difficulty**: Hard to access directly as an external PI; easier as a partner
  institution

### Technology Innovation Institute (TII) / AI71: Abu Dhabi

- **Website**: <https://www.tii.ae/partnership-and-programmes>
- **Submission**: N/A. No public grant application portal; collaboration/funding
  inquiries routed through the "Let's talk" contact form at
  <https://www.tii.ae/contact-us>
- Not a grants program. TII is ATRC's applied-research arm (9 centers, incl. AI
  & Digital Science) and gives away its Falcon LLM family as open-weight models;
  AI71 is the commercial spin-off for enterprise deployment
- Relevant mainly as (a) a source of free, capable open models for research, and
  (b) a potential collaboration/co-PI partner for postgrad placements and joint
  projects, not as a source of direct cash grants to external teams
- **Deadline**: N/A, no application process. Relationship-driven
- **Difficulty**: N/A, no application process. Relationship-driven

## Startup-only (listed for Completeness: Not Applicable to University Applicants)

### NSF SBIR/STTR

- **Website**: <https://seedfund.nsf.gov/>
- **Submission**: <https://seedfund.nsf.gov/apply/get-started/>
- **Award**: $250,000 to $1M+ depending on phase, for early-stage small businesses
- **Deadline**: Phase I full proposals due Nov 4, 2026; Mar 4, 2027; Jul 7, 2027
  (recurring: 1st Wed of Nov, 1st Thu of Mar, 1st Wed of Jul). A Project Pitch
  must be submitted and approved first; Pitches are accepted on a rolling basis

### NVIDIA Inception

- **Website**: <https://www.nvidia.com/en-us/startups/>
- **Submission**: <https://programs.nvidia.com/phoenix/application>
- **Award**: GPU credits, technology access, and network support. No application
  deadlines
- **Deadline**: Rolling; no application deadlines

### AI Grant

- **Website**: <https://aigrant.com/>
- **Submission**:
  <https://airtable.com/appG3EC7hiKlTmV5y/pagE0AXR3bcw3dJ8V/form>
- **Award size**: $250,000 investment via SAFE (no cap, no discount), plus
  $350,000 in Azure credits and $250,000+ in additional credits from partners
  (Anthropic, OpenAI, Replicate, etc.), plus a summit and Demo Day
- **Eligibility**: Any founder or team with a working, useful AI product demo; no
  prior funding or credentials required. Company must be (or become) a
  Delaware C-corp
- **Status**: Batch 4 applications are closed; next batch's timing isn't posted
- **Deadline**: Closed (Batch 4); next batch timing not published
- **Difficulty**: Hard. Competitive, investment-style program, not a grant in the
  traditional sense

### Build in Tulsa: Arch Accelerator

- **Website**: <https://www.buildintulsa.com/program/arch>
- **Submission**: <https://www.buildintulsa.com/program/arch>
- **Award size**: $70K non-dilutive plus a stipend
- **Eligibility**: Revenue-stage tech founders. In person in Tulsa, so
  relocation is required
- **Deadline**: Nov 20, 2026
- **Source**: The Scouting Report #018

### Hello Tomorrow Global Challenge 2026

- **Website**: <https://hello-tomorrow.org/global-challenge/>
- **Submission**: <https://hello-tomorrow.org/global-challenge/>
- **Award size**: EUR 100K equity-free grand prize
- **Eligibility**: Early-stage deep-tech founders, open globally
- **Deadline**: Nov 30, 2026
- **Note**: Finalists pitch at the Hello Tomorrow Summit in Paris
- **Source**: The Scouting Report #017, #018

### SET Award 2027 (dena, Start Up Energy Transition)

- **Website**:
  <https://www.startup-energy-transition.com/set-award/set-award-2027/>
- **Submission**:
  <https://www.startup-energy-transition.com/set-award/set-award-2027/>
- **Award size**: Mainly recognition and exposure, not cash
- **Eligibility**: Climate and energy-transition startups worldwide
- **Deadline**: Nov 30, 2026
- **Note**: Top 100 join the SET100 list. Finalists pitch at the SET Tech
  Festival in Berlin
- **Source**: The Scouting Report #017, #018

### The Great Oral Health Pitch

- **Website**: <https://shilsfund.org/the-great-oral-health-pitch/>
- **Submission**: <https://shilsfund.org/the-great-oral-health-pitch/>
- **Award size**: $2,000 regional prizes feeding a grand prize package worth
  $50K+
- **Eligibility**: Oral-health ideas; national pitch competition
- **Deadline**: Nov 30, 2026
- **Note**: Sponsor not named in the issue (domain: shilsfund.org). Narrow focus
- **Source**: The Scouting Report #017, #018

### BioTools Innovator 2027

- **Website**: <https://biotoolsinnovator.org/apply/>
- **Submission**: <https://biotoolsinnovator.org/apply/>
- **Award size**: Up to $300K
- **Eligibility**: Life-science and biotech tool startups; recruits globally
- **Deadline**: Dec 1, 2026
- **Note**: Sponsor not named in the issue (domain: biotoolsinnovator.org)
- **Source**: The Scouting Report #018

### F. M. Kirby Impact Prize 2027 (Duke CASE)

- **Website**: <https://kirbyimpactprize.smapply.org/prog/2027kirbyprize/>
- **Submission**: <https://kirbyimpactprize.smapply.org/prog/2027kirbyprize/>
- **Award size**: $150K unrestricted
- **Eligibility**: Social-impact ventures
- **Deadline**: Dec 2, 2026
- **Note**: Run by Duke's Center for the Advancement of Social Entrepreneurship.
  May cap at 250 applications
- **Source**: The Scouting Report #018

### D-Prize Global Competition

- **Website**: <https://d-prize.org/>
- **Submission**: <https://d-prize.org/>
- **Award size**: Up to $20K
- **Eligibility**: First-time founders (US eligible) delivering tested
  anti-poverty solutions in lower- and middle-income nations
- **Deadline**: Final extension Dec 6, 2026 (first round was Oct 25)
- **Source**: The Scouting Report #013

### 9Fresh Founder Residency

- **Website**: <https://9fresh.co/founder-residency/>
- **Submission**: <https://9fresh.co/founder-residency/>
- **Award size**: Not a grant. Rooms about $550-$750 a month, utilities and wifi
  included
- **Eligibility**: Founders, teams, independent builders, mentors, investors,
  digital nomads. 12-month on-campus residency in Cazenovia, NY
- **Deadline**: Round 1 closes Dec 11, 2026 (midnight ET)
- **Note**: Move-in March 2027. Residents cover meals and parking
- **Source**: The Scouting Report #018

### SEED SPOT Impact Accelerator

- **Website**: <https://seedspot.org/>
- **Submission**: <https://seedspot.org/>
- **Award size**: Not stated. Takes no equity
- **Eligibility**: Impact-driven founders; national virtual accelerator
- **Deadline**: Jan 11, 2027
- **Source**: The Scouting Report #018

### NIA Start-Up Challenge 2027 (NIH)

- **Website**:
  <https://www.nih.gov/challenges/2027-nia-start-challenge-accelerator>
- **Submission**:
  <https://www.nih.gov/challenges/2027-nia-start-challenge-accelerator>
- **Award size**: Not stated in #018. #016 mentions $994K for an NIH
  aging/longevity challenge (unverified, may be the same program)
- **Eligibility**: Aging- and longevity-focused startups
- **Deadline**: Jan 15, 2027
- **Note**: Application portal opens Dec 1, 2026
- **Source**: The Scouting Report #016, #018

### South Summit Startup Competition 2027

- **Website**: <https://www.southsummit.io/en/content/startup-competition>
- **Submission**: <https://www.southsummit.io/en/content/startup-competition>
- **Award size**: Not stated
- **Eligibility**: Not stated
- **Deadline**: Jan 31, 2027
- **Note**: Finalists pitch at South Summit in Madrid
- **Source**: The Scouting Report #018

### Claude for Startups (Anthropic)

- **Website**: <https://claude.com/programs/startups>
- **Submission**: <https://claude.com/programs/startups>
- **Award size**: Free Claude API credits plus priority rate limits (no dollar
  value stated)
- **Eligibility**: Startups (eligibility not stated in the newsletter)
- **Deadline**: Rolling; no deadline listed
- **Source**: The Scouting Report #011, #012, #018

### Microsoft for Startups Founders Hub

- **Website**: <https://www.microsoft.com/en-us/startups>
- **Submission**: <https://www.microsoft.com/en-us/startups>
- **Award size**: Up to ~$150K in Azure credits plus AI model access and dev
  tools
- **Eligibility**: Startups; no equity or VC introduction required
- **Deadline**: Rolling; no deadline listed
- **Source**: The Scouting Report #011, #012, #018

### AWS Activate

- **Website**: <https://aws.amazon.com/activate/>
- **Submission**: <https://aws.amazon.com/activate/>
- **Award size**: $1K-$100K in AWS credits, valid for two years
- **Eligibility**: Early-stage startups
- **Deadline**: Rolling; no deadline listed
- **Source**: The Scouting Report #011, #012, #018

### ElevenLabs Startup Grants

- **Website**: <https://elevenlabs.io/startup-grants>
- **Submission**: <https://elevenlabs.io/startup-grants>
- **Award size**: 33M free voice-AI credits (~$10K value) for 12 months
- **Eligibility**: Startups (eligibility not stated)
- **Deadline**: Rolling; no deadline listed
- **Source**: The Scouting Report #016, #017, #018

### Google for Startups Cloud Program

- **Website**: <https://cloud.google.com/startup>
- **Submission**: <https://cloud.google.com/startup>
- **Award size**: Google Cloud credits; amount not in the email (public info: up
  to ~$200K, more for AI startups; verify)
- **Eligibility**: Early-stage startups (tier by funding stage)
- **Deadline**: Rolling; no deadline listed
- **Source**: Inbound email from Google Cloud Startups SDR. Amount and URL not
  in the email, verify on site

### SoGal, iFundWomen, Google for Startups (named only)

- **Website**: N/A, the newsletter gives no URLs
- **Award size**: Not stated
- **Eligibility**: Not stated
- **Deadline**: Rolling all fall
- **Note**: Named with no details. Also mentioned: "a deep bench of
  accelerators"
- **Source**: The Scouting Report #015, #016, #017, #018

### Techstars Anywhere

- **Website**: <https://www.techstars.com/accelerators/anywhere>
- **Submission**: <https://www.techstars.com/accelerators/anywhere>
- **Award size**: Not stated
- **Eligibility**: Remote-first; no relocation required
- **Deadline**: Closed (last call was Nov 18, 2026)
- **Note**: Program begins March 2027
- **Source**: The Scouting Report #013, #018

### Y Combinator Winter 2027

- **Website**: <https://www.ycombinator.com/apply>
- **Submission**: <https://www.ycombinator.com/apply>
- **Award size**: $125K for 7% plus $375K uncapped SAFE
- **Eligibility**: Open globally
- **Deadline**: Closed (was Nov 2, 2026)
- **Note**: Three-month program ending in Demo Day
- **Source**: The Scouting Report #013, #015

### Supernova Challenge (Expand North Star, Dubai)

- **Website**:
  <https://expandnorthstar.com/supernova-challenge-pitch-competition>
- **Submission**:
  <https://expandnorthstar.com/supernova-challenge-pitch-competition>
- **Award size**: Equity-free pitch pool; no cash amount stated
- **Eligibility**: International reach; travel to Dubai required
- **Deadline**: Closed (was Nov 2, 2026)
- **Note**: Held at GITEX in Dubai, Dec 8-10, 2026
- **Source**: The Scouting Report #013, #015

### Unearthodox Exploration Co-Lab 2027

- **Website**:
  <https://opportunitydesk.org/2026/10/02/unearthodoxs-exploration-co-lab-2027-edges-of-possibility-for-innovators/>
- **Submission**:
  <https://opportunitydesk.org/2026/10/02/unearthodoxs-exploration-co-lab-2027-edges-of-possibility-for-innovators/>
- **Award size**: CHF 10K (~$12K) plus 10-month online incubation
- **Eligibility**: Global early-stage ventures addressing root causes of
  environmental and social problems; must be past the idea stage
- **Deadline**: Closed (was Nov 16, 2026)
- **Note**: Link is to an aggregator (Opportunity Desk), not the sponsor
- **Source**: The Scouting Report #017

## Student Funding

### VentureWell E-Teams

- **Website**: <https://venturewell.org/e-team-program/>
- **Submission**: <https://venturewell.org/e-team-program/>
- **Award size**: Non-dilutive, in two stages: Pioneer $5,000 and Propel
  $20,000, up to $25,000 per team
- **Eligibility**: Teams of 2+ students with a faculty advisor, currently
  enrolled at a US institution
- **Deadline**: Spring 2027 cycle: applications opened Nov 9, 2026 and are due
  Jan 25, 2027. Three cycles per year
- **Note**: Some aggregator sites list $10K–$100K, which conflicts with
  VentureWell's own $25,000 maximum. Check the official page for current
  guidelines

### Pitch Dingman Competition (UMD Dingman Center)

- **Website**:
  <https://www.rhsmith.umd.edu/centers/dingman-center/initiatives-programs/pitch-dingman-competition>
- **Submission**: Same page; shortcut <https://go.umd.edu/pitchdingman>
- **Award size**: 2026 top prize $25,000. Over $170,000 in total prizes in 2025
  and 2026. Prizes change each year
- **Eligibility**: UMD ventures. A University System of Maryland (USM) schools
  track was added in 2023, with a $10,000 prize that year
- **Deadline**: Not stated. Check the program page
- **Note**: The Dingman Center also runs UMD Dingman Center Angels (see the
  `FriendlyAngels` tab)

### Mokhtarzada Hatchery (UMD Computer Science)

- **Website**: <https://hatchery.umd.edu/>
- **Submission**: <https://www.hatchery.umd.edu/apply>
- **Award size**: Seed funding of up to $10,000 per team per year. Teams can
  have other funding
- **Eligibility**: Teams of 2 to 4 members. At least one member is a current
  degree-seeking UMD student. Other members are current UMD students or UMD
  graduates from the prior spring. Technology companies get preference
- **Deadline**: Applications usually open about two weeks before the fall
  semester starts. No fixed date is stated. The 2025-2026 round is closed. The
  site has a 2026-2027 cohort page, but it gives no application details
- **Note**: Also gives mentoring from an experienced entrepreneur, a dedicated
  24/7 workspace in the Brendan Iribe Center (Room 2237), and access to
  speakers and advisors from the Washington, D.C. startup ecosystem. Up to 4
  teams are admitted each year. Launched in 2021, funded by the alumni
  Haroon, Idris, and Zeki Mokhtarzada. Mtech Ventures
  (<https://mtechventures.umd.edu/>) is a separate UMD incubator: it gives lab
  and office space, but no funding

### NFTE World Series of Innovation

- **Website**: <https://innovation.nfte.com>
- **Submission**: <https://innovation.nfte.com>
- **Award size**: $300-$1,500 cash prizes
- **Eligibility**: Ages 5-24 (high school and younger); online innovation
  challenges
- **Deadline**: Dec 13, 2026
- **Source**: The Scouting Report #012

### Tulane Business Model Competition

- **Website**:
  <https://freeman.tulane.edu/lepage/tulane-business-model-competition>
- **Submission**:
  <https://freeman.tulane.edu/lepage/tulane-business-model-competition>
- **Award size**: $125K split among three finalists
- **Eligibility**: National; teams of 1-5 degree-seeking students (one student
  leads and owns at least 20%)
- **Deadline**: Jan 10, 2027
- **Note**: Finals in New Orleans on Mar 9, 2027
- **Source**: The Scouting Report #012, #018

### Diamond Challenge (high school)

- **Website**: <https://diamondchallenge.org/>
- **Submission**: <https://diamondchallenge.org/>
- **Award size**: $100K+ total awards; $12K per first-place winner
- **Eligibility**: High school students; Business and Social Innovation tracks
- **Deadline**: Jan 14, 2027
- **Note**: Run by the University of Delaware Horn program
- **Source**: The Scouting Report #012

### Rice Business Plan Competition

- **Website**: <https://rbpc.rice.edu/eligibility>
- **Submission**: <https://rbpc.rice.edu/eligibility>
- **Award size**: More than $1M in cash and investment
- **Eligibility**: Student-led startups from any university with 2+ student
  founders and less than $300K raised
- **Deadline**: Jan 29, 2027
- **Note**: About 40 teams compete over three days in Houston
- **Source**: The Scouting Report #012, #018

### e-Fest (Schulze Family Foundation Challenge)

- **Website**:
  <https://business.stthomas.edu/schulze-school/student-engagement/competitions/efest/>
- **Submission**:
  <https://business.stthomas.edu/schulze-school/student-engagement/competitions/efest/>
- **Award size**: More than $200K in cash across pitch and business-plan tracks
- **Eligibility**: Undergraduate teams nationwide
- **Deadline**: Jan 31, 2027
- **Note**: Run by the University of St. Thomas and the Schulze Family
  Foundation
- **Source**: The Scouting Report #018

### Blue Ocean Student Entrepreneur Competition (high school)

- **Website**: <https://www.blueoceancompetition.org/>
- **Submission**: <https://www.blueoceancompetition.org/>
- **Award size**: Cash prizes (amount not listed)
- **Eligibility**: High school students worldwide; fully virtual pitch
- **Deadline**: Feb 21, 2027
- **Source**: The Scouting Report #012

### Microsoft Imagine Cup 2027

- **Website**: <https://imaginecup.microsoft.com>
- **Submission**: <https://imaginecup.microsoft.com>
- **Award size**: Azure credits plus Student, Finalist, and Category prizes
  (values not stated)
- **Eligibility**: Early-stage software or AI startups: registered entity
  founded on or after Nov 2024, not yet Series C, founders 18+. Separate Student
  Prize for student teams; open to non-students too
- **Deadline**: Feb 28, 2027
- **Source**: The Scouting Report #016, #018

### TCU Richards Barrentine Values and Ventures Competition

- **Website**: <https://www.neeley.tcu.edu/VandV>
- **Submission**: <https://www.neeley.tcu.edu/VandV>
- **Award size**: Described as "significant non-dilutive capital" (no figure
  given)
- **Eligibility**: Undergraduate teams of 2-5 (one team per school) with
  values-driven for-profit ventures
- **Deadline**: Early spring semester 2027 (exact date TBA)
- **Source**: The Scouting Report #012

### Hult Prize

- **Website**: <https://www.hultprize.org/>
- **Submission**: <https://www.hultprize.org/>
- **Award size**: $1M grand prize
- **Eligibility**: Any college student, anywhere
- **Deadline**: Campus qualifiers Sep 2026 - Feb 2027 (no single date)
- **Note**: Confirm dates on the site
- **Source**: The Scouting Report #011, #012

### Global Student Entrepreneur Awards (GSEA)

- **Website**: <https://eonetwork.org/gsea/>
- **Submission**: <https://eonetwork.org/gsea/>
- **Award size**: $100K+ prize pool
- **Eligibility**: Students already running a real venture (Entrepreneurs'
  Organization)
- **Deadline**: Local rounds run fall and winter (no single date)
- **Note**: Confirm dates on the site
- **Source**: The Scouting Report #011, #012

### Y Combinator Early Decision (students)

- **Website**: <https://www.ycombinator.com/early-decision>
- **Submission**: <https://www.ycombinator.com/early-decision>
- **Award size**: $125K for 7% plus $375K uncapped SAFE
- **Eligibility**: Students who apply now and join a future batch after
  graduating; no need to drop out
- **Deadline**: Rolling (apply anytime)
- **Source**: The Scouting Report #011, #012, #018

### Thiel Fellowship

- **Website**: <https://www.thielfellowship.org/apply>
- **Submission**: <https://www.thielfellowship.org/apply>
- **Award size**: $250K over two years
- **Eligibility**: Age 22 or under; degree not required
- **Deadline**: Rolling (apply anytime)
- **Source**: The Scouting Report #012

### Z Fellows

- **Website**: <https://www.zfellows.com/>
- **Submission**: <https://www.zfellows.com/>
- **Award size**: $10K, or investment at a $1B cap
- **Eligibility**: All ages; one intensive week with technical builders
- **Deadline**: Rolling (multiple cohorts a year)
- **Source**: The Scouting Report #012

### Dorm Room Fund

- **Website**: <https://dormroomfund.com/opportunities>
- **Submission**: <https://dormroomfund.com/opportunities>
- **Award size**: Pre-seed investment for equity (check size not listed)
- **Eligibility**: Student founders across the US
- **Deadline**: Rolling
- **Source**: The Scouting Report #012

### 1517 Fund Medici Grant

- **Website**: <https://www.1517fund.com/medici>
- **Submission**: <https://www.1517fund.com/medici>
- **Award size**: $1K
- **Eligibility**: Young builders and students working on projects outside the
  classroom
- **Deadline**: Rolling (apply anytime)
- **Source**: The Scouting Report #011, #012

### Awesome Foundation

- **Website**: <https://www.awesomefoundation.org/en>
- **Submission**: <https://www.awesomefoundation.org/en>
- **Award size**: $1K micro-grant, no equity
- **Eligibility**: Idea-stage projects accepted; nationwide, via local chapters
- **Deadline**: Rolling (monthly)
- **Source**: The Scouting Report #011, #012

### GoDaddy Empower (students)

- **Website**: <https://www.godaddy.com/godaddy-for-good/empower>
- **Submission**: <https://www.godaddy.com/godaddy-for-good/empower>
- **Award size**: In-kind: free .com domain and AI website builder for three
  years
- **Eligibility**: Enrolled students 18+
- **Deadline**: Not stated
- **Note**: Not a cash grant
- **Source**: The Scouting Report #015, #016

### TigerLaunch 2027 (Princeton)

- **Website**: <https://tigerlaunch.com/>
- **Submission**: <https://tigerlaunch.com/>
- **Award size**: $60K non-dilutive prizes ($30K / $20K / $10K)
- **Eligibility**: Students from any college
- **Deadline**: Closed (was Nov 1, 2026, extended)
- **Note**: Student-led. Regional rounds this winter; global finals at Princeton
  in April
- **Source**: The Scouting Report #015

### SXSW Pitch 2027 (Student Startups)

- **Website**: <https://cart.sxsw.com/products/acceleratorapp>
- **Submission**: <https://cart.sxsw.com/products/acceleratorapp>
- **Award size**: Not stated
- **Eligibility**: Student startups launched before Jan 2024 with under $10M
  raised
- **Deadline**: Closed (was Nov 13, 2026)
- **Note**: Newsletter author discloses serving on the SXSW Pitch Advisory Board
- **Source**: The Scouting Report #015, #016, #017

### VentureWell Ecosystem Futures Fellowship

- **Website**: <https://venturewell.org/ecosystem-futures-fellowship/>
- **Submission**: <https://venturewell.org/ecosystem-futures-fellowship/>
- **Award size**: $10K base + $5K travel per fellow
- **Eligibility**: Two-person campus teams with at least one faculty member,
  building STEM entrepreneurship ecosystems. HBCUs, HSIs, TCUs, and community
  colleges especially welcome
- **Deadline**: Closed (was Oct 30, 2026)
- **Source**: The Scouting Report #013, #015

### Ember Fellowship

- **Website**: <https://emberfellows.org>
- **Submission**: <https://emberfellows.org>
- **Award size**: Not stated. Two-year program ending in a full-time startup
  role in Baltimore, Birmingham, Atlanta, or Detroit
- **Eligibility**: Seniors and recent graduates who want a startup career;
  competitive selection
- **Deadline**: Closed (was Nov 6, 2026)
- **Note**: Training alongside staff from IDEO, McKinsey, and OpenAI. Baltimore
  placement is relevant for UMD
- **Source**: The Scouting Report #013, #015, #016

### State of Black Entrepreneurship Fellows 2027 (Center for Black Entrepreneurship)

- **Website**: <https://cbecenter.org/program/research-program/>
- **Submission**: <https://cbecenter.org/program/research-program/>
- **Award size**: $10K research fellowship
- **Eligibility**: Doctoral students, postdocs, and faculty researchers; HBCU
  affiliates especially encouraged
- **Deadline**: Closed (was Nov 16, 2026)
- **Note**: Year-long contribution to the annual report. Research-focused, not
  founder-focused
- **Source**: The Scouting Report #017

### Other regional student pitch programs

- **Website**: N/A, not yet researched
- **Award size**: Varies
- **Eligibility**: Students in Maryland and the mid-Atlantic region
- **Deadline**: Varies
- **Note**: Listed from the advice above. The Pitch Dingman Competition has a USM
  track. The USM programs are in the USM section below. List more programs here
  once found

## Small Business & Founder Grants

Micro-grants and targeted grants for founders and small businesses, from
The Scouting Report newsletter. Mostly not applicable to university
applicants.

### Shophand Small Business Boost

- **Website**: <https://www.shophand.com/small-business-boost-grant>
- **Submission**: <https://www.shophand.com/small-business-boost-grant>
- **Award size**: $2,500 cash plus $2,500 in services
- **Eligibility**: US for-profits with under 50 employees and under $5M revenue;
  no demographic restriction
- **Deadline**: Nov 30, 2026
- **Source**: The Scouting Report #012, #013, #015, #016, #017, #018

### Threadneedle Alley Foundation Grant

- **Website**: <https://www.threadneedlealley.org/>
- **Submission**: <https://www.threadneedlealley.org/>
- **Award size**: $5,000 plus mentorship
- **Eligibility**: US midlife founders from underserved communities facing real
  financial barriers
- **Deadline**: Nov 30, 2026
- **Source**: The Scouting Report #017, #018

### EDDIE Awards 2027 (Synergies Work)

- **Website**: <https://synergieswork.org/eddie-awards-application/>
- **Submission**: <https://synergieswork.org/eddie-awards-application/>
- **Award size**: $10K top prize plus finalist travel
- **Eligibility**: US founders with disabilities
- **Deadline**: Nov 30, 2026
- **Source**: The Scouting Report #018

### Hiring Our Heroes Small Business Grant

- **Website**: <https://www.hiringourheroes.org/?p=6518>
- **Submission**: <https://www.hiringourheroes.org/?p=6518>
- **Award size**: Not stated
- **Eligibility**: Veteran- and military-spouse-owned small businesses
- **Deadline**: Dec 15, 2026
- **Note**: Run by the US Chamber of Commerce Foundation
- **Source**: The Scouting Report #018

### The No Excuses Fund (.TECH Domains)

- **Website**: <https://noexcuses.get.tech/>
- **Submission**: <https://noexcuses.get.tech/>
- **Award size**: Not stated
- **Eligibility**: Founders who have been turned down
- **Deadline**: Dec 15, 2026
- **Note**: Unconditional fund. Decisions by mid-January
- **Source**: The Scouting Report #018

### HerRise MicroGrant

- **Website**: <https://www.hersuitespot.com/herrise/>
- **Submission**: <https://www.hersuitespot.com/herrise/>
- **Award size**: $1,000 monthly
- **Eligibility**: Women-owned businesses (at least 51% women-owned) with under
  $1M revenue
- **Deadline**: Monthly, last day of each month (next: Nov 30, 2026)
- **Note**: Sponsor not named in the issue (domain: hersuitespot.com)
- **Source**: The Scouting Report #011, #015, #016, #017, #018

### Freed Fellowship

- **Website**: <https://freedfellowship.com/>
- **Submission**: <https://freedfellowship.com/>
- **Award size**: $500 monthly to one winner; $2,500 year-end grant to one
  fellow
- **Eligibility**: Not stated
- **Deadline**: Rolling (monthly)
- **Note**: Applications reopen monthly
- **Source**: The Scouting Report #013, #015, #016, #017, #018

### Amber Grant (WomensNet)

- **Website**: <https://ambergrantsforwomen.com/get-an-amber-grant/>
- **Submission**: <https://ambergrantsforwomen.com/get-an-amber-grant/>
- **Award size**: $10K monthly; $25K year-end grant for one monthly winner
- **Eligibility**: Woman-owned businesses
- **Deadline**: Rolling (monthly)
- **Source**: The Scouting Report #011, #012

### SoGal Black Founder Startup Grant

- **Website**: <https://www.iamsogal.com/black-founder-startup-grant/>
- **Submission**: <https://www.iamsogal.com/black-founder-startup-grant/>
- **Award size**: $5K-$10K cash plus lifetime advisor access
- **Eligibility**: Black women and nonbinary founders
- **Deadline**: Rolling (winners chosen monthly)
- **Source**: The Scouting Report #011, #012

### Galaxy Grant

- **Website**: <https://galaxyofstars.org/galaxy-grants/>
- **Submission**: <https://galaxyofstars.org/galaxy-grants/>
- **Award size**: $2,500 base award (past winners $750-$25K)
- **Eligibility**: Women and minority business owners; free to apply
- **Deadline**: Multiple cycles per year (last known cycle closed Oct 31, 2026)
- **Note**: Sponsor not named in the issue (domain: galaxyofstars.org)
- **Source**: The Scouting Report #013, #015

### AI Impact Grant

- **Website**: <https://aiimpactgrant.com/>
- **Submission**: <https://aiimpactgrant.com/>
- **Award size**: $2,500 quarterly
- **Eligibility**: Entrepreneurs building with AI, including those just starting
- **Deadline**: Quarterly (last known cycle closed Nov 15, 2026)
- **Note**: Sponsor not named in the issue. Relevant to AI work
- **Source**: The Scouting Report #017

### Darcy's Business Freedom Grant

- **Website**: <https://businessfreedomgrant.com/>
- **Submission**: <https://businessfreedomgrant.com/>
- **Award size**: $1,000 unrestricted
- **Eligibility**: Any US small business owner; no entry fee
- **Deadline**: Closed (was Nov 15, 2026)
- **Source**: The Scouting Report #017

### Women Founders Grant

- **Website**: <https://womenfoundersgrant.com/apply-now-women-grants>
- **Submission**: <https://womenfoundersgrant.com/apply-now-women-grants>
- **Award size**: $5,000
- **Eligibility**: Businesses majority-founded by women, any stage, nationwide
- **Deadline**: Closed (was Oct 31, 2026)
- **Note**: Sponsor not named in the issue
- **Source**: The Scouting Report #013, #015

### Breva Thrive Grant (Q4)

- **Website**: <https://www.breva.ai/thrive-grant>
- **Submission**: <https://www.breva.ai/thrive-grant>
- **Award size**: $5,000
- **Eligibility**: US for-profit companies with a revenue history and a
  community-benefit angle
- **Deadline**: Closed (Q4 cycle was Oct 31, 2026)
- **Note**: Quarterly program
- **Source**: The Scouting Report #013, #015

### Visa Foundation: US Small Business Growth & Innovation

- **Website**: <https://rpaforvisafoundation.smapply.us/>
- **Submission**: <https://rpaforvisafoundation.smapply.us/>
- **Award size**: $100K-$250K
- **Eligibility**: US 501(c)(3) entrepreneurial-support organizations with a
  $1M-$10M budget and two years of audited financials, serving small and micro
  businesses
- **Deadline**: Closed (was Oct 21, 2026)
- **Note**: Organization-level capacity grant: the organization applies and
  receives the funds
- **Source**: The Scouting Report #013

## Maryland State Programs

### TEDCO Maryland Innovation Initiative (MII)

- **Website**: <https://www.tedcomd.com/mii>
- **Submission**: <https://www.tedcomd.com/mii>. The page has no direct
  application link or RFA link: check the Request for Applications (RFA) of each
  track
- **Award size**: Non-dilutive, two phases, 9 months each:
  - Technology Assessment: up to $130,000 for a sole application, up to
    $180,000 for a joint application (two Qualifying Universities)
  - Company Formation: up to $300,000
  - Up to $430,000 total for a sole applicant through both phases
- **Eligibility**:
  - Technology Assessment: full-time faculty at a Qualifying University
  - Company Formation: Maryland-based startups that licensed technology from a
    Qualifying University within 12 months
  - Qualifying Universities: Johns Hopkins, Morgan State, UMB, UMBC, and UMCP
    (UMD College Park)
  - The technology must be owned by the university, disclosed to its tech
    transfer office, and have IP protection
- **Deadline**: Reviewed quarterly. Complete applications are due 5:00pm on
  Jan 15, Apr 15, Jul 15, and Oct 15 (next business day if the 15th is not a
  business day). Next: Jan 15, 2027
- **Note**: Students are not listed as applicants on the MII page. A student
  project needs a faculty member to apply, or a licensed startup for Company
  Formation
- **Watch**: Per advice, MII funding will open to students at UMD soon, as an
  expansion of BII beyond Baltimore (see BII below). Not confirmed on the TEDCO
  site. It is not clear if it will get a new web page or be listed on the main
  MII page. Alla McCoy at UMD probably knows the most about this

### TEDCO Baltimore Innovation Initiative (BII)

- **Website**: <https://www.tedcomd.com/baltimore-innovation-initiative-bii>
- **Submission**: Submit an Indication of Interest (IOI) first:
  <https://clear-chiller-c1b.notion.site/193dfc3fd28c80878d1af6010e25ae36>. The
  full application portal link comes after the IOI is accepted. Requirements:
  <https://clear-chiller-c1b.notion.site/Baltimore-Innovation-Initiative-314dfc3fd28c807fb4cdcad1ce81235d>
- **Award size**: Non-dilutive
  - Technology Advancement: up to $50,000, or up to $75,000 for a joint
    application from two or more Qualifying Universities. Up to 9 months
  - Entrepreneurship Commercialization, Programming, and Infrastructure: up to
    $50,000. Up to 12 months
- **Eligibility**:
  - Institutions in the Baltimore-Columbia-Towson MSA that opt in as
    collaborators. Only collaborating organizations can submit proposals
  - Students and student principal investigators can apply for Technology
    Advancement
  - Full-time faculty and staff at eligible institutions can apply
  - Technology-based startups can apply for Technology Advancement if they work
    with faculty, students, or staff from a participating institution
  - Entrepreneurship track: staff or faculty who run entrepreneurship or
    commercialization programs
  - Participating institutions include JHU, Morgan State, Towson, UMB, UMBC,
    University of Baltimore, Loyola, MICA, and Howard CC. UMD College Park is
    not listed
- **Deadline**: Rolling window Aug 19, 2026 to Nov 5, 2026, 5:00pm. Closed as of
  Nov 19, 2026. Late submissions are rejected
- **Note**: Described as a pilot limited to the Baltimore area. The page says
  nothing about expansion. The expansion to UMD comes from advice, not from the
  page. Questions go to the onsite Project Lead listed on the BII page
- **Contact**: Alla McCoy at UMD, for the expansion

## University System of Maryland (USM) Innovation Collective

- **Resources page**: <https://www.usmd.edu/launch/resources/>. Questions to
  <launch@usmd.edu>
- **Advice**: The Launch Fund is in a holding pattern right now. Start with the
  resources on the page above
- **Not funding, so not listed below**: Maryland SBDCs, Maryland Business
  Opportunity Center, Startup UMD workshops, I-Corps, Maryland New Venture
  Program (12-week program, UMBC), UMB IP and legal assistance

### USM Launch Fund

- **Website**: <https://www.usmd.edu/launch/>
- **Submission**: Not stated
- **Award size**: Grant funding. Amount not stated on the resources page
- **Eligibility**: USM-related ventures, including those led by alumni and
  located in USM spaces
- **Status**: In a holding pattern, per advice. Not confirmed on the site
- **Deadline**: Not stated

### USM Momentum Fund

- **Website**: <https://momentum.usmd.edu/>
- **Award size**: $16M fund. It invests in startups (equity-style, not a grant)
- **Eligibility**: USM-related startups, including those led by alumni and
  located in USM spaces
- **Deadline**: Not stated

### Salisbury University: Shore Hatchery

- **Website**:
  <https://www.salisbury.edu/academic-offices/business/entrepreneurship/shore-hatchery/>
- **Award size**: Up to $200,000 per year in prize money, through fall and
  spring business plan competitions
- **Eligibility**: Not stated on the resources page. Check the program page
- **Deadline**: Fall and spring competitions. Dates not stated

### Towson University: StarTUp

- **Website**: <https://www.towson.edu/startup/>
- **Award size**: Free co-working space. The accelerator program gives
  mentoring, space, and $10K
- **Eligibility**: Not stated on the resources page. Check the program page
- **Deadline**: Not stated

### Maryland Industrial Partnerships (MIPS)

- **Website**: <https://www.mips.umd.edu/>
- **Award size**: Funding for university-based research projects that help
  companies develop new products. Amount not stated on the resources page
- **Eligibility**: Implied: companies that partner with university researchers.
  Check the program page
- **Deadline**: Not stated

### Maryland Technical Internship Program (MTIP)

- **Website**: <https://mtip.umbc.edu/>
- **Award size**: Reimbursement for interns. Amount not stated on the resources
  page
- **Eligibility**: Not stated on the resources page. Check the program page
- **Deadline**: Not stated

## Directories, Search Tools, and Other Resources

### Grantmatch (Grants.gov "Search Grants")

- **Website**: <https://www.grants.gov/search-grants>
- **Submission**: N/A. Search/discovery tool only; each result links out to the
  awarding agency's own application process
- **What it is**: The official US government search interface for federal grant
  opportunities, filterable by keyword, agency, eligibility, status, and date
- **Eligibility**: Depends on the program found. Grants.gov itself requires
  SAM.gov/UEI registration to view full opportunity details and apply
- **Deadline**: N/A, search tool only; deadlines vary per listing found
- **Difficulty**: Easy to search; difficulty of any individual award varies
  widely

### Seedfund: NSF AI Topic Page

- **Website**: <https://seedfund.nsf.gov/topics/artificial-intelligence/>
- **Submission**: <https://seedfund.nsf.gov/apply/get-started/>
- **Award size**: Up to $2M in seed funding, zero equity taken
- **Eligibility**: US small businesses (<500 employees), majority US-citizen
  owned, not majority VC/PE/hedge-fund owned; PI must work ≥20 hrs/week at the
  company
- **Focus areas**: Cognitive-science-based tech, computer vision, conversational
  AI, language-based AI, novel hardware, sustainable/low-resource AI,
  trustworthy AI
- **Note**: Same underlying NSF SBIR/STTR program as above; this is NSF's
  AI-specific topic landing page
- **Deadline**: Same as NSF SBIR/STTR above: next Phase I proposal deadline
  Nov 4, 2026
- **Difficulty**: Hard

### Aigrant

- **Website**: <https://aigrant.org/>
- **Submission**: N/A. No application process listed on the site
- **Award size**: $5,000–$50,000, "no strings attached," as cash or compute
  credits
- **Eligibility**: Developers/maintainers of open-source AI/ML software projects
- **Note**: Founded in 2017 by Nat Friedman and Daniel Gross. Distinct from AI
  Grant (aigrant.com) above despite the similar name
- **Deadline**: N/A, no published application process
- **Difficulty**: Moderate. No public intake process found; recipients appear to
  be sourced directly by the funders

### USgrants

- **Website**: <https://www.usgrants.org/business/artificial-intelligence-researchers>
  ("Artificial Intelligence Researchers Grants 2024/2025")
- **Submission**: <https://www.usgrants.org/proposals>
- **What it is**: Commercial grant-directory listing federal, state, and private
  funding opportunities for AI researchers and businesses
- **Note**: Site returned HTTP 403 to automated fetch; content not independently
  verified beyond the URL/title
- **Deadline**: N/A, directory only; deadlines vary per listing
- **Difficulty**: N/A, directory only

### ICTworks: Generative AI Solutions for Social Impact

- **Website**: <https://www.ictworks.org/generative-ai-solutions-for-social-impact/>
- **Submission**: N/A. Guide/roundup article, not a funder itself
- **Deadline**: N/A, guide/roundup article, not a funder
- **Difficulty**: N/A, directory/guide only

### NEA FY 2025 Grant Opportunities

- **Website**:
  <https://www.arts.gov/news/press-releases/2023/nea-fy-2025-grant-opportunities-now-available>
- **Submission**: <https://www.arts.gov/grants/first-time-applicant-guide-for-organizations>
- **What it is**: National Endowment for the Arts general grant cycle; not
  AI-specific, included for completeness for AI+art crossover projects
- **Eligibility**: Arts organizations (nonprofits, government entities); not
  individuals for most tracks
- **Deadline**: FY2027 cycles have both closed (GAP1 Feb 12, 2026; GAP2 Jul 9,
  2026). FY2028 guidelines not yet published; NEA typically posts them around
  December
- **Difficulty**: Moderate

### USAID Program

- **Website**: <https://www.ictworks.org/how-find-usaid-contracts-and-grants-funding/>
- **Submission**: N/A. Guide article; actual submissions go through Grants.gov,
  SAM.gov, or a USAID Mission/prime partner
- **What it is**: Guide to finding USAID funding via two tracks: contracts
  (acquisitions, for-profit) and grants/cooperative agreements (assistance,
  non-profit)
- **Eligibility**: Local NGOs, local businesses, and new/nontraditional partners
  (small businesses, faith-based orgs, diaspora groups)
- **Process**: Register on Grants.gov/SAM.gov, obtain a UEI/CAGE code, monitor
  the USAID Business Forecast, and build relationships with USAID Missions or
  prime contractors
- **Deadline**: N/A, guide only; deadlines vary per USAID solicitation/Mission
- **Difficulty**: Hard. Requires federal registration and relationship-building

### Maryland Department of Commerce: Build Our Future Grant Program

- **Website**: <https://commerce.maryland.gov/fund/build-our-future-grant-program>
- **Submission**: N/A. Page does not list a direct application portal
- **Award size**: ~$7M distributed in FY26 across innovation-infrastructure
  projects statewide
- **Eligibility**: Innovation infrastructure projects in Maryland's eligible
  strategic technology sectors
- **Note**: As of July 1, 2026, administered by the Maryland Economic
  Development Corporation (MEDCO) under the DECADE Act of 2026, not Commerce
  directly
- **Deadline**: Rolling via MEDCO; no fixed deadline. Program authorized through
  June 30, 2030
- **Difficulty**: Moderate

## Ethereum Foundation: Academic Grants Round

- **Website**: <https://blog.ethereum.org/2025/01/21/academic-round-25>
- **Submission**: <https://esp.ethereum.foundation/academic-grants>
- **Award size**: $2M total pool distributed across selected proposals
- **Eligibility**: Academic researchers at universities and research institutions
  globally
- **Disciplines**: Economics & game theory, theoretical and applied cryptography,
  consensus and protocol design, networking & P2P, client engineering, security,
  formal verification, humanities
- **Project length**: Typically 6–12 months, with some flexibility
- **Cadence**: Annual, spring deadline
- **Deadline**: Annual, spring window; exact 2027 date not yet announced
- **Difficulty**: Moderate. Roughly 300 applications from 25 countries in the 2024
  round

## Ethereum Foundation: PhD Fellowship

- **Website**: <https://blog.ethereum.org/2026/02/05/phd-fellowship>
- **Submission**: <https://esp.ethereum.foundation/rounds/phdfp26>
- **Award size**: $24,000 over one year, as a supplement to an existing stipend
  9–10 fellowships total
- **Eligibility**: Actively enrolled PhD students
- **Requirements**: Must produce formal academic research published in conferences
  or journals. Output must be open-access under a free, permissive license
  Written PDF proposal in English following their template
- **Timeline**: Decisions emailed six weeks after the deadline. The 2026 deadline
  was April 22
- **Deadline**: Closed. 2026 deadline was April 22, 2026; annual, next cycle
  expected ~early 2027
- **Difficulty**: Hard, given fewer than a dozen slots

## Ethereum Foundation: Ecosystem Support Program (ESP)

- **Website**: <https://esp.ethereum.foundation/applicants>
- **Submission**: <https://esp.ethereum.foundation/applicants/wishlist>
- **Deadline**: Rolling; always open
- The general, always-open channel. If you miss a dedicated round, book ESP
  Office Hours
- Recent Q1 2026 allocations went heavily to ZK research, cryptographic
  primitives, formal verification of zkVMs, quantum-resistant schemes, and core
  client work

## Ripple UBRI (University Blockchain Research Initiative)

- **Website**: <https://ripple.com/impact/ubri/>
- **Submission**: N/A. Ripple selects and invites partner universities directly;
  no public application portal
- **Scale**: Over $80 million committed since 2018 across 60+ university partners
- **Model**: Institutional partnership, not an open application. Universities set
  their own research priorities
- **Recent**: $1.5M+ in renewed grants disbursed in RLUSD stablecoin. Advisory
  Council of cryptographers from partner universities
- **Deadline**: N/A, invitation-only, no public application
- **Action**: Check whether your institution is already a partner before building
  plans around this

## NLnet Foundation / NGI Zero

- **Website**: <https://nlnet.nl/NGI0/>
- **Submission**: <https://nlnet.nl/propose/>
- **Award size**: Typically up to ~$58K (€50,000), paid in milestones as
  deliverables are completed
- **Eligibility**: Individuals, researchers, nonprofits, small teams, companies,
  institutions: worldwide. They fund people, not just organizations
- **Requirement**: All funded work must be released under free/libre/open-source
  licenses, with open standards where possible
- **Status**: **Currently closed.** New calls open September 3, 2026, deadline
  November 3, 2026 at 12:00 CEST. Historically ran back-to-back calls with
  deadlines on the 1st of every even month
- **Deadline**: November 3, 2026, 12:00 CEST (call opens September 3, 2026)
- **Process**: Short initial application, with detail requested only for promising
  proposals

## Filecoin Grants

- **Website**: <https://fil.org/grants>
- **Submission**: <https://github.com/filecoin-project/devgrants/issues/new/choose>
- **Tiers**: Open Grants, Builder Next Step Grants ($5K–$10K), and RFPs with
  defined deliverables and milestones
- **Focus**: Developer and data tooling, integrations, research and protocols,
  storage and retrieval, Filecoin Virtual Machine
- **Deadline**: Rolling; no fixed deadline, accepted year-round via GitHub issues

## Solana Foundation Grants / Superteam

- **Website**: <https://solana.org/grants-funding>
- **Submission**: <https://share.hsforms.com/1GE1hYdApQGaDiCgaiWMXHA5lohw>
- **Award size**: Main Grants Program is milestone-based, no fixed cap. Superteam
  microgrants up to $10,000 for early-stage builders, emerging-markets focus
- **Eligibility**: Any builder shipping open-source, non-custodial projects on or
  migrating to Solana
- **Deadline**: Rolling; no fixed deadline
- **Difficulty**: Easy–moderate

## Optimism Retro Funding (RetroPGF)

- **Website**: <https://www.optimism.io/blog/retro-funding-2025>
- **Submission**: <https://atlas.optimism.io/missions>
- **Model**: Retroactive. Build and ship first, then apply for a share of OP
  tokens based on demonstrated value. 850M OP (20% of total supply, ~$85M at
  ~$0.10/OP, highly volatile) permanently earmarked for public goods
- **Structure**: Restructured into themed "Retro Funding Missions" (e.g., Dev
  Tooling) with monthly application checkpoints instead of one annual round
- **Deadline**: Paused. Per Optimism's Jan 8, 2026 "Season 9" announcement, Retro
  Funding will not run for at least 12 months from that date; no resumption
  date published
- **Difficulty**: Moderate. Requires an existing track record on Optimism/OP Stack

## Polygon Community Grants

- **Website**:
  <https://polygon.technology/blog/polygon-community-grants-program-announces-season-2-with-35m-pol-and-ai-leading-the-pack>
- **Submission**: <https://polygon.questbook.xyz/>
- Focus for 2026: Explicitly prioritizes projects at the intersection of AI
  agents, verifiable compute, and on-chain infrastructure
- **Deadline**: Closed. Season 2 direct-grant deadline was April 15, 2025;
  questbook.xyz now shows "Program Closed." A Season 3 (~80M POL) has been
  referenced but no confirmed open date is published
- **Difficulty**: Moderate

## L1/L2 Ecosystem Grants

### Uniswap Foundation Grants

- **Website**: <https://www.uniswapfoundation.org/grants>
- **Submission**: <https://share.hsforms.com/1fxQjPQTgTYmPwlYxxKlSGQsdca9>
- **Award size**: Not fixed ("variable based on project scope"); $250K+
  publicly announced for larger grants. Sub-programs: Unichain Retro Grants up
  to $7,500/mo; Hook Incubator up to $40,000 in prizes + tuition; audit
  subsidies up to 100% of audit costs. ~$26M committed in 2025 against an
  $87.5M+ pipeline
- **Eligibility**: Teams building across the Uniswap ecosystem at any stage —
  protocols, tooling, research, governance, community
- **Focus areas**: v4 hooks and Unichain growth are current priorities
- **Deadline**: Rolling; no fixed cycle published
- **Difficulty**: Moderate–Hard. Milestone-based contracts and KYC for larger
  asks; smaller sub-program grants are easier

### Base Ecosystem Grants

- **Website**: <https://docs.base.org/get-started/get-funded>
- **Submission**: Split by program — Builder Grants nomination form:
  <https://docs.google.com/forms/d/e/1FAIpQLSfXuEzmiAzRhie_z9raFCF1BXweXgVt18o-DvBuRRgyTygL2A/viewform>;
  Ecosystem Fund (equity, not a grant): <https://base.org/ecosystem-fund/apply>;
  Base Batches: <https://basebatches.xyz> (2026 cohort closed)
- **Award size**: Weekly Rewards 2 ETH/week pool; Builder Grants 1–5 ETH; Base
  Batches 2026 gave $10K grants to top 15 teams, some also got $50K investment
- **Eligibility**: Anyone building on Base for Weekly Rewards/Builder Grants
  (mostly team-sourced, not open call); Base Batches targets pre-seed teams
  (<~$250K raised) and students
- **Focus areas**: General Base ecosystem growth — DeFi, AI agents, prediction
  markets, tooling, robotics (Batches Virtuals track)
- **Deadline**: Builder Grants/Weekly Rewards rolling; Base Batches 2026
  cohort closed (windows ran Feb–Apr 2026), next round not announced
- **Difficulty**: Moderate–Hard. Builder Grants are largely discovery-based,
  not open application; Batches had 1,100+ applicants for ~15–20 spots

### Avalanche Foundation Grants

- **Website**: <https://docs.avax.network/grants>; flagship program:
  <https://retro9000.avax.network>
- **Submission**: <https://retro9000.avax.network> (connect wallet to apply);
  Research Grants via
  <https://docs.avax.network/grants/avalanche-research-proposals>
- **Award size**: Retro9000 — $40M pool, retroactive, ranked leaderboard, no
  project gets >25% of a monthly pool. Research Grants — up to $50,000, paid
  in 3 milestones over 12 months
- **Eligibility**: Retro9000 is retroactive-only: already-shipped, live
  Avalanche L1s/infra tooling, not pre-launch ideas. Research Grants: academic
  researchers/independent scholars
- **Focus areas**: Retro9000 rewards live Avalanche L1s, core infra/dev
  tooling, C-Chain usage; Research Grants fund cryptoasset-pricing and
  validator-economics research. (infraBUIDL programs are winding down; the
  Blizzard Fund is equity, not a grant)
- **Deadline**: Rolling monthly/quarterly rounds; most recently found round
  deadlines (Jul 2026) have passed, next round not confirmed
- **Difficulty**: Moderate–Hard. Retroactive and usage-ranked, favors projects
  with real already-live traction

### Starknet Seed Grants

- **Website**: <https://www.starknet.io/grants/seed-grants/>
- **Submission**: <https://tinyurl.com/starknet-seed-grants> (Airtable form)
- **Award size**: Up to $25,000 in STRK, non-dilutive
- **Eligibility**: Early-stage teams with an MVP/proof of concept; active in
  the Starknet community or a hackathon participant; must build on/integrate
  existing Starknet tools
- **Focus areas**: No industry restriction — gaming, DeFi, social, payments,
  spatial computing
- **Deadline**: Rolling, no fixed deadline; decision in ~4 weeks
- **Difficulty**: Easy. Broad eligibility, low award floor, rolling
  submissions

### Starknet Growth Grants

- **Website**: <https://www.starknet.io/grants/growth-grants/>
- **Submission**: <https://tinyurl.com/starknet-growth-grants> (Airtable, with
  category-specific sub-forms)
- **Award size**: $25,000 up to $1,000,000 in STRK, non-dilutive
- **Eligibility**: Later-stage teams with a live product already in market and
  significant onchain usage
- **Focus areas**: Innovation, Ecosystem Integration, Content & Creator, "The
  Next Billion" (onboarding), Bitcoin Builder Grants
- **Deadline**: Rolling, no fixed deadline
- **Difficulty**: Hard. Requires proven product and onchain traction; rigorous
  internal-committee interview

### Arbitrum Foundation / DAO Grants

- **Website**: <https://arbitrum.foundation/grants>
- **Submission**: No single portal. Most tracks (STIP, Questbook Domain
  Allocator Season 3, Trailblazer, general Grant Program) are marked
  Inactive/Complete. Of the 4 marked "Active" on the hub: Audit Program's form
  is confirmed closed despite the label; ArbiFuel's Airtable deadline
  (extended to Jan 31, 2026) has lapsed; Gaming Ventures apply page
  (<https://arbitrumgamingventures.com/apply>) appears open; Alchemy-Arbitrum's
  Typeform link is dead
- **Award size**: Audit Program $10M ARB pool over 12 months; ArbiFuel up to 1M
  free transactions/$10K gas credits (window closed); Alchemy program $10M
  pool, up to $500K infra credits/team
- **Eligibility**: Narrow per surviving track — early-stage projects needing
  audits, simplified-UX apps/wallets, game/consumer startups, or Orbit-rollup
  teams
- **Deadline**: Mostly closed or lapsed; the hub page appears stale/not kept
  current
- **Difficulty**: Hard. The lower-barrier generalist track is gone; remaining
  programs are narrow and several apply links are confirmed dead

### Near Foundation Grants

- **Website**: <https://www.near.org/funding> (legacy grants.near.org no
  longer resolves)
- **Submission**: N/A for a direct NF grant. Routed instead to House of Stake
  governance (<https://houseofstake.org>), Potlock
  (<https://app.potlock.org>), NEAR DevHub bounties
  (<https://nearn.io/devhub>), and NEAR Protocol Rewards
  (<https://nearprotocolrewards.com>)
- **Award size**: No current NF figures published. NEAR Protocol Rewards: up
  to $10K/month during the program, $200K+ distributed total
- **Eligibility**: Protocol Rewards targets AI-powered DeFi/agents/AI-ML infra
  builders; DevHub bounties and Potlock are open to community contributors
- **Deadline**: N/A for direct NF grants — NEAR's governance forum states NF
  is "pausing" temporary grants in favor of the House of Stake/NDC treasury
  (transfer vote slated Q1 2026)
- **Difficulty**: N/A direct NF grant (paused, not competitive). Protocol
  Rewards: Moderate; DevHub/Potlock: Easy–Moderate

## DeFi Protocol Grants

### Compound Grants

- **Website**: <https://www.comp.xyz/c/grants/10>; dashboard
  <https://compoundgrants.questbook.app/>
- **Submission**: <https://compoundgrants.questbook.app/>
- **Award size**: Rapid Grants (<$125K, ~2-week turnaround) and Community
  Grants ($125K+); prior 12-month budget was $1.375M
- **Eligibility**: Builders working on dApps, security tooling, DAO
  operations, or multichain expansion for Compound
- **Deadline**: Uncertain. The live Questbook dashboard shows $0 allocated for
  the current round; a third-party tracker lists the program as "deadpooled."
  Treat as unverified/likely dormant, confirm directly on comp.xyz
- **Difficulty**: Moderate when active (~25% historical acceptance rate)

### Curve DAO Grants

- **Website**: <https://gov.curve.finance/>
- **Submission**: N/A, no application form — create a proposal on the
  governance forum or discuss with a team member on Discord/Telegram
- **Award size**: No fixed range; ad hoc DAO-voted CRV grants. Community fund
  holds ~24M unallocated CRV, vesting linearly over 1+ year
- **Eligibility**: Individuals/teams actively contributing to the Curve
  ecosystem
- **Focus areas**: Core protocol development, community support, education,
  ecosystem growth
- **Deadline**: Rolling; no fixed cycle, governed case-by-case
- **Difficulty**: Hard. Requires a full governance forum proposal and DAO
  vote; no lightweight small-grant track

### Lido LEGO (Lido Ecosystem Grants Organization)

- **Website**: <https://lido.fi/lego>; forum
  <https://research.lido.fi/c/community-grants/13>
- **Submission**: N/A form — direct outreach to a LEGO committee member via
  Telegram, then forum review
- **Award size**: Tiered — Sandgrain (up to $1,000), Pebble (up to $10,000),
  Boulder (up to $100,000), Mountain ($100,000+)
- **Eligibility**: Developers/contributors working on initiatives benefiting
  Lido or the broader Ethereum/PoS/DeFi ecosystem
- **Deadline**: Rolling, quarterly budget cycles; reports published at
  lego.lido.fi
- **Difficulty**: Easy–Moderate for small (Sandgrain/Pebble) asks; Hard for
  Mountain-tier

### 1inch Foundation Grants

- **Website**: <https://1inch.network/foundation>
- **Submission**: N/A standing form — apply by joining the conversation on the
  governance forum, <https://gov.1inch.network/>
- **Award size**: Original 2021 program was $3M in 1INCH over 12 months;
  current model is discrete DAO-voted grants (e.g. 2026 events and
  public-policy grant threads), not a fixed pooled fund
- **Eligibility**: Developers/teams/individuals building tools, products, or
  content for the 1inch ecosystem; must comply with regional restrictions
- **Focus areas**: Network development (integrations, AMMs, wallets, bridges,
  security), community development, research, events, policy/regulatory
  engagement
- **Deadline**: Rolling, proposal-driven via governance forum
- **Difficulty**: Moderate. Requires a governance forum proposal/pitch

### Sky Ecosystem Grants (formerly MakerDAO Development Grants)

- **Website**:
  <https://community-development.makerdao.com/en/programs/development-grants>
- **Submission**: N/A. Page states: "not accepting new applications at the
  moment"
- **Award size**: 5,000–50,000 DAI historically, milestone-based
- **Eligibility**: Projects that increase Dai/USDS adoption, support emerging
  economies, or open new markets (excludes pure research, marketing spend,
  token distributions)
- **Deadline**: Closed/paused, awaiting "next wave," no reopening date
  published. (Separately, Sky governance authorized up to $2.5B in USDS for
  Obex-incubated yield projects — a treasury-investment vehicle, not this
  grants program)
- **Difficulty**: N/A while closed; historically Moderate

### Balancer Grants

- **Website**:
  <https://balancer.gitbook.io/balancer-v2/ecosystem/community/grants>
- **Submission**: Grants Application form linked from the grants page;
  contact grants@balancer.community
- **Award size**: $1K–$100K historically (some sources cite up to $75K), paid
  in BAL via milestone-based emissions
- **Eligibility**: Individuals/groups building on or supporting the Balancer
  Protocol
- **Deadline**: Uncertain. Balancer Labs (the corporate entity) announced
  shutdown in March 2026 following a ~$110–128M Nov 2025 exploit; the protocol
  now runs a leaner DAO/Foundation model with zero BAL emissions. Whether
  Grants survived this restructuring is unconfirmed — verify directly before
  relying on it
- **Difficulty**: Moderate historically; Hard/uncertain now given the treasury
  disruption

### Aave Grants DAO

- **Deadline / Status**: **Discontinued.** Per the Aave governance forum
  (confirmed July 2026), AGD is "no longer accepting or reviewing new grant
  applications." Listed for awareness only

## Web3 Infrastructure & Tooling Grants

### Safe Ecosystem Grants

- **Website**: <https://safefoundation.org/grants>
- **Submission**: No web form — apply via the "Apply now" email link on the
  grants page (project description, milestones, budget, team background)
- **Award size**: Not fixed; milestone-based, negotiated per proposal. Wave 1
  (2023–24) distributed ~$500K across 21 teams from 120 applicants. A separate
  bug-bounty track pays up to $1M for critical smart-contract vulnerabilities
- **Eligibility**: Any team/individual with relevant experience and a clear,
  well-scoped contribution to the Safe smart-account ecosystem
- **Focus areas**: Protocol & infrastructure, security, developer
  tooling/SDKs, research, ecosystem growth, applications
- **Deadline**: Rolling ("Spontaneous Grants"), no cohort deadlines
- **Difficulty**: Moderate. ~1 in 6 applicants funded historically; ~4-week
  reply time

### ENS DAO Grants (Ecosystem Working Group)

- **Website**: <https://basics.ensdao.org/ens-ecosystem-wg>
- **Submission**: N/A form — post in the "ENS Ecosystem" category on
  <https://discuss.ens.domains>, attend the weekly open WG call, or contact a
  steward directly
- **Award size**: Most recent published cycle (Term 6, Mar–Oct 2025): standard
  grants 10,000 USDC + 200 ENS; larger grants 25,000 USDC + 500 ENS; mini
  grants 0.5–1 ETH
- **Eligibility**: Developers/projects that are ENS-specific or ENS-centric,
  particularly ones unlikely to get funded by other protocol DAOs
- **Deadline**: In transition. On Aug 11, 2026 tokenholders voted to hand the
  $65M endowment and grants oversight to a new ENS Foundation; Term 7 stewards
  are active but no Term 7 grants summary published yet
- **Difficulty**: Moderate–Hard. No self-serve portal, discretionary steward
  review, retroactive/proof-of-work bias

### Gitcoin Grants (Quadratic Funding)

- **Website**: <https://grants.gitcoin.co/>
- **Submission**: N/A currently — no round open as of Aug 2026; GG25 is still
  "exploratory thinking" per an Aug 4, 2026 governance post, explicitly not a
  promise of a round
- **Award size**: GG24 (Oct 2025, most recent completed round): ~$1.8M total
  matching across 6 thematic domains
- **Eligibility**: Open-source/public-goods projects tied to the Ethereum
  ecosystem, sorted into thematic domains, vetted per-domain by domain
  operators
- **Focus areas**: Ethereum public-goods/infra funding via quadratic funding
  and other "plural funding" mechanisms ("Gitcoin 3.0")
- **Deadline**: Between rounds; GG25 not yet confirmed
- **Difficulty**: Hard right now (no active round); when rounds run,
  Moderate–Hard due to domain vetting and thin matching pools

### Protocol Labs: Founders Forge / PL_Genesis

- **Website**: <https://www.founders-forge.xyz/> and
  <https://www.plgenesis.com/>
- **Submission**:
  <https://docs.google.com/forms/d/e/1FAIpQLSfPkuHoM2a6XkZYvtoZnWJD7lXRNHCWNkOh8Ki9otJncd0mug/viewform>
  (Founders Forge interest form). PL_Genesis hackathon has no standing apply
  link; its last season closed Mar 16, 2026, no next season announced
- **Award size**: Founders Forge up to $30K non-dilutive per team plus
  potential follow-on funding (12-week accelerator); PL_Genesis "Frontiers of
  Collaboration" (closed) had a $150K+ prize pool
- **Eligibility**: Early-stage Web3/AI teams or individual builders, no
  geographic restriction; Founders Forge draws mainly from PL_Genesis
  finishers plus interest-form applicants
- **Focus areas**: AI, decentralized compute, cryptography, DePIN, data/dev
  infrastructure, programmable finance — general frontier-tech, not
  Filecoin-specific
- **Deadline**: Founders Forge cohort program active/rolling interest form;
  PL_Genesis between seasons
- **Difficulty**: Moderate–Hard. Hackathon entry is easy but winning is
  competitive; Founders Forge cohort selection is selective

### The Graph Foundation Grants

- **Website**: <https://thegraph.com/grants>
- **Submission**: N/A currently — a Jul 6, 2026 Foundation forum post says new
  applications are paused to reassess strategic priorities; the site notes "no
  open RFPs at this time" though the page's Apply button hasn't been removed
- **Award size**: Not published; the Foundation states "millions of dollars in
  funding" provided cumulatively
- **Eligibility**: Individuals/projects that are open source and contribute to
  The Graph ecosystem; non-technical applicants accepted
- **Deadline**: Paused, no resumption date given
- **Difficulty**: N/A currently (not accepting applications); historically
  Moderate

### Chainlink Grants / BUILD

- **Deadline / Status**: **Dormant/winding down.** The Chainlink Community
  Grants page (chain.link/community/grants) 404s as of Aug 2026 with no
  working apply link; the Chainlink BUILD program's page also 404s and a Jun
  26, 2026 blog post says BUILD is "evolving" to case-by-case commercial
  agreements rather than open applications, with its associated
  token-rewards program winding down (final claims closed Jul 7, 2026).
  Listed for awareness only

### ENS Public Goods Working Group Grants

- **Deadline / Status**: **Discontinued.** Separate from the ENS DAO Ecosystem
  WG entry above (formerly builder.ensgrants.xyz); the portal states
  applications are closed and the Working Group is sunsetting (~July 2026,
  after 4.5 years). Listed for awareness only

### Protocol Labs Research Grants (classic program)

- **Deadline / Status**: **Discontinued.** Indefinitely paused since Jan 18,
  2023 per official blog post, still confirmed paused as of Aug 2026; parent
  org rebranded to PL R&D with no public grants offering. See Founders
  Forge/PL_Genesis above for current PL funding

## Other L1 Ecosystem Grants

### Algorand Foundation xGov Program (Beta)

- **Website**: <https://xgov.algorand.co/>
- **Submission**: <https://xgov.algorand.co/> — connect wallet, complete KYC,
  pay a one-time 100 ALGO fee plus an Open Proposal Fee/escrow
- **Award size**: Minimum request 10,000 ALGO (~$790 at ~$0.079/ALGO); no
  public maximum. Treasury funded via ~110–120M ALGO/year from node staking
  rewards
- **Eligibility**: Open to anyone, but as of the Oct 2025 Beta relaunch, only
  retroactive proposals are accepted (completed open-source contributions,
  not future plans)
- **Deadline**: Open, rolling/continuous; xGov Council election period opens
  mid/late Aug 2026
- **Difficulty**: Moderate. Retroactive-only rule, mandatory KYC/fees,
  competitive council + block-producer voting

### Stellar Community Fund: Build Award (SCF)

- **Website**: <https://communityfund.stellar.org/awards>
- **Submission**: <https://communityfund.stellar.org/awards/reccaFUJmN4HNQxvo>,
  starting with an interest form
- **Award size**: Up to $150,000 in XLM per Build Award, milestone tranches
  over ~3–6 months; smaller "Instawards" up to $15K; possible follow-on to
  $300K total lifecycle funding
- **Eligibility**: Open Track (new projects), Integration Track (existing
  tools/traction), or RFP Track; all applicants require KYC/KYB
- **Focus areas**: DeFi/payment protocols, developer tooling/infrastructure,
  asset tokenization, novel Stellar/Soroban use cases
- **Deadline**: Round #45 open, deadline Aug 16, 2026 (rolling rounds continue
  after)
- **Difficulty**: Moderate–Hard. SCF 7.0 added referral-based intake favoring
  alumni/partner-referred teams plus AI prescreening for others

### Aptos Foundation Grants

- **Website**: <https://aptosnetwork.com/grants> (formerly
  aptosfoundation.org/grants)
- **Submission**: No standalone form URL — apply flow is embedded on each
  track page; contact grants@aptosfoundation.org
- **Award size**: Ecosystem Grants ~$5,000–$50,000; Payment Grants up to
  $150,000; Aptos Assembly Program up to $50,000 plus accelerator access —
  all milestone-based
- **Eligibility**: Teams with a live product and evidence of real
  adoption/growth, must add substantial new ecosystem value, open-source or
  commit to open-sourcing
- **Focus areas**: Ecosystem Grants (apps/infra/tools); Payment Grants
  (wallets, APIs, stablecoin payments); Assembly Program (5-week sprint + Demo
  Day)
- **Deadline**: Rolling, no fixed deadline
- **Difficulty**: Moderate. Requires a live product with demonstrated traction

### Sui Foundation Grants (RFP + Academic Research Awards)

- **Website**: <https://www.sui.io/programs-funding>
- **Submission**: <https://sui.io/request-for-proposals> (Sui Grants Hub,
  per-RFP windows); Academic Research Awards at
  <https://suifoundation.notion.site/Sui-Ecosystem-Academic-Research-Awards>;
  Hydropower accelerator at <https://hydropower.sui.io/>
- **Award size**: Academic Research Awards: $25K. RFP/Direct Strategic
  Investment: not fixed, case-by-case. Hydropower: non-dilutive, no stated
  cash figure
- **Eligibility**: Builders using Move aligned with Foundation-published RFP
  priorities; researchers advancing blockchain science/security (Academic
  track); early-stage founders (Hydropower)
- **Deadline**: RFP cohorts rolling/per-topic; no fixed 2026 cohort dates
  found
- **Difficulty**: Hard. Proposals must match specific published RFP topics;
  committee review, shortlisting, vote, identity verification; historical
  rounds selected ~14 of "hundreds" of submissions

### Web3 Foundation Grants (Polkadot)

- **Deadline / Status**: **Discontinued** (Dec 30, 2025 official notice). W3F
  stopped its general open grants program, shifting to internal targeted
  funding and technical review of Polkadot OpenGov Treasury proposals; no
  public apply portal remains. Listed for awareness only

### Interchain Foundation Grants (Cosmos)

- **Deadline / Status**: **Closed/paused.** The apply form
  (apply.interchain.io) states "no longer accepting responses... temporarily
  closed due to application volume"; the program-info GitHub repo was
  archived Apr 20, 2026. A separate, live "Investments" track (equity-style,
  with Cosmos Labs) continues at <https://interchain.io/investments>. Listed
  for awareness only

# Common Rejection Reasons

**Federal AI grants:** unclear research objectives, insufficient preliminary
data, budgets misaligned with proposed activities, failure to address
responsible AI considerations. NSF AI programs run roughly 15–25% success rates

**Web3 grants:**

- _Generic project descriptions_: be specific about what it does, who it serves,
  what's novel
- _Closed-source code_: most Web3 grants require open source
- _Mismatched ecosystem commitment_: applying to one foundation while building
  on another chain
- _No track record_: zero GitHub activity or ecosystem presence gets
  deprioritized
- _Asking too much_: grants under $100K are meaningfully easier than those over
  $250K
- _Vague milestones_: "build the protocol" fails; "ship v0.1 with feature X by
  date Y" works

**UAE grants:**

- _No local host_: AARE/AYIA and Dubai RDI both require a UAE-based PI or
  institution. A US-only team without a Dubai/Abu Dhabi partner is disqualified
  before review. Sandooq Al Watan's small Grants programme and MBZUAI's student
  fellowships are the main exceptions open directly to individuals
- _Budget mismatch_: Dubai RDI requires at least 50% of funds spent inside
  Dubai, so a proposal built around foreign-only spending fails eligibility
  screening
- _Cold outreach_: ASPIRE has no public rolling portal. Applications that skip a
  warm introduction (existing UAE partner institution or direct outreach to
  ATRC/ASPIRE) rarely get traction

# Suggested Sequence

1. **This month:** NAIRR Start-Up resource request: fastest turnaround, gets
   compute in hand
2. **Near term:** DOE Genesis Mission AI if the research fits; start PAMS
   registration now regardless
3. **September:** NLnet call reopens
4. **Fall prep:** Draft the Ethereum Foundation Academic Grants proposal for the
   spring round
5. **Watch:** NSF AI Institutes solicitation, AI Forge nonprofit launch
6. **If pursuing UAE:** identify a Dubai- or Abu Dhabi-based host institution
   first (e.g., Khalifa University, MBZUAI, NYU Abu Dhabi): Dubai RDI and
   ASPIRE both gate eligibility on having one: then approach Dubai RDI's next
   Research Projects Funding round or ASPIRE via a warm introduction
