#!/usr/bin/env python

"""
Process PitchBook CSV data.
"""

import json
import logging
import re
from typing import Callable, List, Set, Union

import dev_scripts_helpers.dockerize.lib_prettier as dshdlipr
import pandas as pd
from tqdm.auto import tqdm

import helpers.hcache_simple as hcacsimp  # noqa: F401
import helpers.hdbg as hdbg
import helpers.hllm_cli as hllmcli
import helpers.hprint as hprint

# TODO(gp): Convert this into an import.
from ck_marketing.workflows.contact_df_utils import is_valid_email

_LOG = logging.getLogger(__name__)


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize a dataframe.

    :param df: input dataframe
    :return: dataframe with sanitized column names
    """
    df = df.reset_index(drop=True)
    df.insert(0, "idx", df.index)
    return df


def load_49_companies_data() -> pd.DataFrame:
    """
    Load data from CSV files.

    :return: dataframe with loaded data
    """
    dfs = []
    df = pd.read_csv("../../data_49_companies/pitchbook_table_0.csv")
    dfs.append(df)
    df = pd.concat(dfs)
    #
    df = _sanitize_df(df)
    #
    _LOG.info("Loaded %d rows", df.shape[0])
    return df


def load_200_companies_data() -> pd.DataFrame:
    """
    Load data from CSV files.

    :return: dataframe with loaded data
    """
    dfs = []
    for i in range(0, 4):
        df = pd.read_csv(f"../../data_200_companies/pitchbook_table_{i}.csv")
        dfs.append(df)
    df = pd.concat(dfs)
    #
    df = _sanitize_df(df)
    #
    _LOG.info("Loaded %d rows", df.shape[0])
    return df


# def get_pitchbook_data_from_gsheet(url: str, tab_name: str, *,
#     remove_spaces: bool = True,
# ) -> pd.DataFrame:
#     df = ckmktwf.get_cached_gsheet_to_df(url, tab_name)
#     pb_columns = [
#         "People",
#         "LinkedInURL",
#         "LastName",
#         "FirstName",
#         "PrimaryCompany",
#         "PrimaryPosition",
#         "Biography",
#         "BoardSeats",
#         "Roles",
#         "DealRoles",
#         "Location",
#         "AddressLine1",
#         "AddressLine2",
#         "City",
#         "State/Province",
#         "PostCode",
#         "Country/Territory/Region",
#         "Phone",
#         "Fax",
#         "Email",
#     ]
#     if remove_spaces:
#         df.columns = [col.replace(" ", "") for col in df.columns]
#     hdbg.dassert_eq(pb_columns, df.columns.to_list())
#     # Sanitize column names.
#     df = _sanitize_df(df)
#     return df


# #############################################################################


# Common titles and suffixes to remove when extracting first names.
_NAME_SUFFIXES = [
    "Ph.D",
    "PhD",
    "M.D.",
    "MD",
    "Jr.",
    "Jr",
    "Sr.",
    "Sr",
    "II",
    "III",
    "IV",
    "Esq.",
    "Esq",
]

# Role patterns to identify in position strings.
# Maps role keywords to column names.
_ROLE_PATTERNS = {
    "is_cofounder": [
        r"\bfounder\b",
        r"\bco-founder\b",
        r"\bcofounder\b",
        r"\bco founder\b",
    ],
    # "is_founder": [r"\bfounder\b"],
    "is_ceo": [r"\bceo\b", r"\bchief executive officer\b"],
    "is_coo": [r"\bcoo\b", r"\bchief operating officer\b"],
    "is_cto": [r"\bcto\b", r"\bchief technology officer\b"],
    "is_cso": [r"\bcso\b", r"\bchief strategy officer\b"],
    # "is_cfo": [r"\bcfo\b", r"\bchief financial officer\b"],
    # "is_cmo": [r"\bcmo\b", r"\bchief marketing officer\b"],
    # "is_cpo": [r"\bcpo\b", r"\bchief product officer\b"],
    # "is_cio": [r"\bcio\b", r"\bchief information officer\b"],
    # "is_chief": [r"\bchie(?:f)?\b"],  # Handles truncated "Chie..."
    # "is_chairman": [r"\bchairman\b", r"\bchairperson\b", r"\bchair\b"],
    # "is_president": [r"\bpresident\b"],
    # "is_board_member": [r"\bboard member\b", r"\bboard director\b"],
    # "is_director": [r"\bdirector\b"],
    # "is_vp": [r"\bvp\b", r"\bvice president\b"],
}


def _parse_position_roles(position: str) -> Set[str]:
    """
    Parse a position string and identify all role types present.

    Handles truncated text (e.g., "Chie..." for "Chief").

    :param position: the position string (e.g., "Co-Founder & Chief...")
    :return: set of role column names that matched (e.g., {"is_cofounder", "is_chief"})
    """
    if pd.isna(position):
        return set()
    hdbg.dassert_isinstance(position, str)
    position_lower = position.lower()
    matched_roles = set()
    # Check each role pattern.
    for role_name, patterns in _ROLE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, position_lower):
                matched_roles.add(role_name)
                _LOG.debug(
                    "Matched role='%s' with pattern='%s' in position='%s'",
                    role_name,
                    pattern,
                    position,
                )
                break
    return matched_roles


# #############################################################################


def _extract_first_name(full_name: str) -> str:
    """
    Extract the first name from a full name string.

    Removes common titles and suffixes before extracting the first word.

    :param full_name: the full name string (e.g., "George Church Ph.D")
    :return: the first name (e.g., "George")
    """
    hdbg.dassert_isinstance(full_name, str)
    # Remove titles/suffixes.
    cleaned_name = full_name
    for suffix in _NAME_SUFFIXES:
        # Use word boundary to avoid partial matches.
        pattern = r"\b" + re.escape(suffix) + r"\b"
        cleaned_name = re.sub(pattern, "", cleaned_name, flags=re.IGNORECASE)
    # Remove extra whitespace and split.
    cleaned_name = cleaned_name.strip()
    # Split on whitespace and take the first token.
    parts = cleaned_name.split()
    hdbg.dassert_lt(
        0,
        len(parts),
        "Name has no parts after cleaning:",
        full_name,
    )
    first_name = parts[0]
    _LOG.debug(
        "Extracted first_name='%s' from full_name='%s'", first_name, full_name
    )
    return first_name


def remove_invalid_emails(
    df: pd.DataFrame,
    *,
    email_col_name: str = "Email",
    reverse: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Remove invalid emails from a dataframe.

    :param df: input dataframe
    :param email_col_name: name of the email column
    :param reverse: if True, remove valid emails instead of invalid ones
    :return: dataframe with invalid emails removed
    """
    df[email_col_name] = df[email_col_name].astype(str)
    mask = df[email_col_name].apply(is_valid_email)
    if reverse:
        mask = ~mask
    _LOG.info(
        "Removed emails: %s", hprint.perc(mask.sum(), len(mask), invert=True)
    )
    if verbose:
        _LOG.info("Removed emails:")
        display(df[~mask])
    df = df[mask]
    df = df.reset_index(drop=True)
    return df


# #############################################################################


def process_pitchbook_dataframe(
    df: pd.DataFrame, *, add_role_columns: bool = True
) -> pd.DataFrame:
    """
    Process the dataframe to add first name and role columns.

    :param df: input dataframe with 'People' and 'Primary Position'
        columns
    :param add_role_columns: if True, add role columns
    :return: processed dataframe with additional columns
    """
    # Create a copy to avoid modifying the original.
    df_out = df.copy()
    #
    for col_name in df_out.columns:
        df_out[col_name] = df_out[col_name].astype(str)
    # 1) Extract first names.
    hdbg.dassert_in("People", df.columns, "Missing 'People' column in input CSV")
    _LOG.info("Extracting first names from 'People' column")
    df_out["FirstName"] = df_out["People"].apply(_extract_first_name)
    # 2) Parse positions and create boolean columns.
    _LOG.info("Parsing 'Primary Position' into role columns")
    if add_role_columns:
        # Get all possible role columns.
        role_columns = sorted(_ROLE_PATTERNS.keys())
        # Initialize all role columns to False.
        for role_col in role_columns:
            df_out[role_col] = False
        # Parse each position and set the appropriate role columns.
        hdbg.dassert_in(
            "PrimaryPosition",
            df.columns,
            "Missing 'Primary Position' column in input CSV",
        )
        for idx, position in enumerate(df_out["PrimaryPosition"]):
            matched_roles = _parse_position_roles(position)
            for role in matched_roles:
                df_out.at[idx, role] = True
        _LOG.info("Created %d role columns: %s", len(role_columns), role_columns)
    # 3) Drop unnecessary columns.
    df_out.drop(
        ["Roles", "BoardSeats", "DealRoles", "Phone"], axis=1, inplace=True
    )
    df_out.drop(
        [
            "Biography",
            "Location",
            "AddressLine1",
            "AddressLine2",
            "City",
            "PostCode",
            "Fax",
        ],
        axis=1,
        inplace=True,
    )
    # 4) Clean up primary company name.
    df_out["PrimaryCompany"] = (
        df_out["PrimaryCompany"]
        .astype(str)
        .str.replace(r"\s*\([^)]*\)", "", regex=True)
        .str.strip()
    )
    return df_out


# People                                                         Scott Belsky
# PrimaryPosition           Chief Strategy Officer & Executive Vice Presid...
# PrimaryCompany                       Adobe (Multimedia and Design Software)
# PrimaryCompany Type                                          Public Company
# Location                                                       San Jose, CA
# Biography                 Mr. Scott Belsky serves as Advisor at Expa. Mr...
# PrimaryCompany Website                                        www.adobe.com
# Email                                                                   NaN
# FirstName                                                             Scott
# is_ceo                                                                False
# is_cofounder                                                          False
# is_coo                                                                False
# is_cso                                                                 True
# is_cto                                                                False


def pick_people(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pick people from a dataframe.

    :param df: input dataframe
    :return: dataframe with selected people
    """
    # 1) Filter out people without an email address.
    df["Email"] = df["Email"].astype(str)
    mask = df["Email"] != "nan"
    _LOG.info(
        "Picked %d people out of %d with email addresses", mask.sum(), len(mask)
    )
    df = df[mask]
    # 2) Pick the people with the highest priority.
    df["priority"] = 0
    df["priority"] += (
        df["is_ceo"].replace({True: 5})
        + df["is_cofounder"].replace({True: 4})
        + df["is_coo"].replace({True: 3})
        + df["is_cso"].replace({True: 2})
        + df["is_cto"].replace({True: 1})
    )
    df["priority"] = df["priority"].astype("Int64").fillna(10)
    df_selected = (
        df.sort_values(["PrimaryCompany Website", "priority"])
        .groupby("PrimaryCompany Website", as_index=False)
        .last()
    )
    return df_selected


def compute_pitchbook_stats(df: pd.DataFrame, *, top_n: int = 10) -> None:
    # Compute the total number of people.
    total_people = len(df)
    _LOG.info("Total number of people: %d", total_people)
    # Compute the total number of companies.
    total_companies = df["PrimaryCompany"].nunique()
    _LOG.info("Total number of companies: %d", total_companies)
    # Count the number of people per company.
    df_count = df.groupby("PrimaryCompany").size().reset_index(name="count")
    df_count.sort_values("count", ascending=False, inplace=True)
    _LOG.info("Top companies by number of people:")
    _LOG.info("%s", df_count.head(top_n).to_string())


# #############################################################################
# Emails
# #############################################################################


# TODO(ai_gp3): Move to message_utils.py


def get_email(row: pd.Series) -> str:
    hdbg.dassert_isinstance(row, pd.Series)
    company_website = row["PrimaryCompany Website"]
    company_name = row["PrimaryCompany"]
    #
    txt = rf"""
    Customize the message <start_msg> <end_msg> for {company_name} at
    {company_website} by specifying <BUSINESS_OUTCOME>. You must make a reference
    to {company_name} in the message.
    Make it less than 30 word, impactful, intriguing without using bold.

    <start_msg>
    My team and I are working on giving LLMs something they've never had before:
    the ability to reason about probability and time. It's a shift that
    materially impact how {company_name} approaches AI in <BUSINESS_OUTCOME>
    <end_msg>

    Print only the text of the completed text starting from "It's a shift that"
    """
    # Can I send you some white paper about our tech? Or would you like to jump on a 15 min call?
    txt = txt.format(company_name=company_name, company_website=company_website)
    txt = hprint.dedent(txt)
    # Process with LLM.
    model = "gpt-4o-mini"
    use_llm_executable = False
    response = hllmcli.apply_llm(
        txt,
        model=model,
        use_llm_executable=use_llm_executable,
    )
    return response


def get_email2(row: pd.Series) -> str:
    hdbg.dassert_isinstance(row, pd.Series)
    first_name = row["FirstName"]
    company_name = row["PrimaryCompany"]
    #
    txt = rf"""
    Customize the email below for {company_name} to
    reference applications of the first LLMs that reason about probability and time.

    Make it less than 70 words, impactful, intriguing without using bold.
    Follow the format of the email below as much as possible, make reference to
    the products of {company_name}.

    Print only the text without comments.

    <example1>
    Given Bosch's commitment to innovation and efficiency, I see clear applications for LLMs that reason about probability and time, such as:
    - optimizing predictive maintenance for smart manufacturing solutions
    - enhancing traffic management systems with real-time data analysis
    - improving energy conservation by forecasting usage patterns in IoT devices

    <example2>
    Where this may be relevant for Bosch is moving from correlation-based prediction to intervention-aware systems, particularly in areas like:
    - predictive maintenance where “what if” reasoning matters more than point forecasts
    - mobility and logistics systems that must adapt in real time to changing conditions
    - energy systems where policy, behavior, and exogenous shocks drive outcomes
    """
    # We're exploring partnerships with product teams who could benefit from more
    # predictive, context-aware AI capabilities. For **Figma**, we see a few
    # high-value possibilities, such as:
    #   - automatically adjusting project timelines based on historical team velocity
    #   - forecasting design trends that can inform component libraries or templates
    #   - predicting user engagement to help teams prioritize design decisions
    # Can I send you some white paper about our tech? Or would you like to jump on a 15 min call?
    txt = txt.format(company_name=company_name)
    txt = hprint.dedent(txt)
    # Process with LLM.
    model = "gpt-4o-mini"
    use_llm_executable = False
    response = hllmcli.apply_llm(
        txt,
        model=model,
        use_llm_executable=use_llm_executable,
    )
    #
    txt = rf"""
Hi {first_name},

I'm GP Saggese, CTO of Causify.AI (and an AI instructor at the University of
Maryland, USA). We build causal AI that improves LLMs for forecasting and automation.

We're currently in confidential discussions following unsolicited inbound
acquisition interest from a strategic industrial player, and are selectively
speaking with companies where our technology could materially accelerate an
existing roadmap.

In brief, Causify has:
- ~$1M ARR with enterprise customers in <18 months
- a team of 5 AI PhDs + 6 senior engineers with deep AI expertise from NVIDIA, Google, Synopsys
- 3 patents focused on causal forecasting and decision automation

{response}

I'd value a brief 10-15 minute conversation to determine whether this capability
is strategically relevant enough to warrant a deeper discussion — or whether
it's not a fit.

If helpful, we can keep this initial conversation informal and confidential.

If you are interested please pick a time on my calendar:
https://cal.com/giacinto-paolo-saggese-c9ysgf/30-min-meeting-anytime

Thank you,
    """
    txt = hprint.dedent(txt)

    txt = dshdlipr.prettier_on_str(txt, "txt")
    return txt


def get_email3(first_name: str, company_website: str, company_name: str) -> str:
    txt = rf"""
    Customize the email below for {company_name} at {company_website} to
    reference applications of the first LLMs that reason about probability and time.

    Make it less than 70 words, impactful, intriguing without using bold.
    Follow the format of the email below as much as possible, make reference to
    the products of {company_name}.

    Print only the text without comments.

    We're exploring partnerships with product teams who could benefit from more
    predictive, context-aware AI capabilities.
    Given Adobe's work in generative creativity and experience intelligence, I see a few areas where these capabilities could be directly relevant, such as:
    - forecasting asset usage and content needs across Creative Cloud and Firefly
    - predicting project timelines and bottlenecks inside collaborative workflows
    - modeling user behavior to improve personalization in Experience Cloud
    """
    txt = txt.format(company_name=company_name, company_website=company_website)
    txt = hprint.dedent(txt)
    # Process with LLM.
    model = "gpt-4o-mini"
    use_llm_executable = False
    response = hllmcli.apply_llm(
        txt,
        model=model,
        use_llm_executable=use_llm_executable,
    )
    #
    txt = rf"""
Hi {first_name},

I'm GP Saggese, and I teach machine learning at the University of Maryland, where my team is developing new methods that let LLMs reason explicitly about probability and time.

{response}

We have deployed this to early partners, and we have already $1M ARR in less than 12 months.

Would someone on your team be open to a brief 10-minute conversation to see if any of this aligns with {company_name}'s roadmap?

Best,
GP
    """
    txt = hprint.dedent(txt)
    return txt


# #############################################################################


def get_email4(row: pd.Series) -> str:
    hdbg.dassert_isinstance(row, pd.Series)
    first_name = row["FirstName"]
    company_name = row["PrimaryCompany"]
    txt = rf"""
    Customize the email below for {company_name} to reference applications of
    the first LLMs that reason about probability and time.

    Make it less than 70 words, impactful, intriguing without using bold.
    Follow the format of the email below as much as possible, make reference to
    the products of {company_name}.

    Print only the text without comments.

    <example1>
    We see clear alignment with Salesforce initiatives across Marketing, Sales,
    and Service Cloud, particularly around predictive analytics and
    uncertainty-aware forecasting.

    <example2>
    We see strong alignment with OpenAI's work on advanced reasoning,
    long-horizon agents, and uncertainty-aware decision-making, particularly
    where models must act over time rather than predict tokens.
    """
    txt = txt.format(company_name=company_name)
    txt = hprint.dedent(txt)
    # Process with LLM.
    model = "gpt-4o-mini"
    use_llm_executable = False
    response = hllmcli.apply_llm(
        txt,
        model=model,
        use_llm_executable=use_llm_executable,
    )
    #
    txt = rf"""
Hi {first_name},

I'm GP Saggese, CTO of Causify.AI and professor in AI at the University of
Maryland, USA.

We build causal AI that enables LLMs to reason about probability and time,
significantly improving forecasting and decision automation.

We have been working with companies in your space.

{response}

I'd appreciate a brief 10-15 minute conversation to determine whether this capability
is strategically relevant enough to warrant a deeper discussion - or whether
it's not a fit.

If helpful, we can keep this initial conversation informal and confidential.

If you are interested please pick a time on my calendar:
https://cal.com/giacinto-paolo-saggese-c9ysgf/30-min-meeting-anytime

Thank you,
    """
    txt = hprint.dedent(txt)
    txt = dshdlipr.prettier_on_str(txt, "txt", print_width=240)
    return txt


# #############################################################################


@hcacsimp.simple_cache(cache_type="json")
def get_email5(row_or_rows: List[pd.Series]) -> List[str]:
    """
    Generate personalized emails for one or more companies.

    Supports batching multiple companies in a single LLM call for
    efficiency.

    :param row_or_rows: list of pd.Series rows (can be size 1 for single
        row)
    :return: list of email strings
    """
    hdbg.dassert_isinstance(row_or_rows, list)
    rows = row_or_rows
    hdbg.dassert_lt(0, len(rows), "Empty list of rows")
    _LOG.debug("Processing %d companies in batch", len(rows))
    # Extract company info from all rows.
    companies = []
    for row in rows:
        hdbg.dassert_isinstance(row, pd.Series)
        companies.append(
            {
                "first_name": row["FirstName"],
                "company_name": row["PrimaryCompany"],
            }
        )
    # Create system prompt with instructions.
    system_prompt = """
    You are customizing email content for multiple companies. For each company:
    - Reference applications of the first LLMs that reason about probability and time
    - Make it less than 50 words, impactful, intriguing without using bold
    - Make reference to the products of the company

    Use these examples as guidance:

    Example 1: We see clear alignment with Salesforce initiatives across Marketing, Sales,
    and Service Cloud, particularly around predictive analytics and
    uncertainty-aware forecasting.

    Example 2: We see strong alignment with OpenAI's work on advanced reasoning,
    long-horizon agents, and uncertainty-aware decision-making, particularly
    where models must act over time rather than predict tokens.

    Return a JSON object with this structure:
    {"results": [{"company_name": "CompanyX", "text": "customized text here"}, ...]}

    Print ONLY valid JSON, no other text or comments.
    """
    system_prompt = hprint.dedent(system_prompt)
    # Create user prompt with company list.
    user_prompt = (
        f"Customize text for these companies:\n{json.dumps(companies, indent=2)}"
    )
    # Call LLM.
    model = "gpt-4o-mini"
    use_llm_executable = False
    _LOG.debug("Calling LLM with %d companies", len(companies))
    response = hllmcli.apply_llm(
        user_prompt,
        system_prompt=system_prompt,
        model=model,
        use_llm_executable=use_llm_executable,
    )
    # Parse JSON response.
    _LOG.debug("Parsing LLM response")
    response_data = json.loads(response)
    hdbg.dassert_in(
        "results", response_data, "LLM response missing 'results' key"
    )
    results = response_data["results"]
    hdbg.dassert_eq(
        len(results),
        len(rows),
        "Number of results doesn't match number of rows:",
        len(results),
        len(rows),
    )
    # Format emails for each company.
    emails = []
    for i, (row, result) in enumerate(zip(rows, results)):
        first_name = row["FirstName"]
        company_name = row["PrimaryCompany"]
        custom_text = result["text"]
        # Verify the result matches the expected company.
        hdbg.dassert_eq(
            result["company_name"],
            company_name,
            f"Company mismatch at index {i}:",
            result["company_name"],
            company_name,
        )
        # Format the email.
        txt = rf"""
Hi {first_name},

I'm GP Saggese, professor in AI at the University of Maryland (USA) and CTO of Causify.AI.

We build causal AI that enables LLMs to reason about probability and time,
significantly improving forecasting and decision automation.

We have validated our tech with companies similar to {company_name}, and we are not coming
out of stealth with $1M ARR in less than 12 months.
{custom_text}

Can I get 10 mins of your time to show a demo of what our new AI can do?
This is not a sales pitch, I just want to show something new and get your
advice.

If you are interested please pick a time on my calendar:
https://cal.com/giacinto-paolo-saggese-c9ysgf/30-min-meeting-anytime

Thank you,
        """
        txt = txt.format(company_name=company_name)
        txt = hprint.dedent(txt)
        txt = dshdlipr.prettier_on_str(txt, "txt", print_width=240)
        emails.append(txt)
    _LOG.debug("Generated %d emails", len(emails))
    return emails


# #############################################################################
# LinkedIn messages
# #############################################################################


def _get_only_element(
    elems: List[Union[str, pd.Series]],
) -> Union[str, pd.Series]:
    hdbg.dassert_isinstance(elems, list)
    hdbg.dassert_eq(len(elems), 1)
    elem = elems[0]
    hdbg.dassert_isinstance(elem, (str, pd.Series))
    return elem


# TODO(ai_gp): Generalize to accept a list of jobs and compute batch like
# get_email5.
@hcacsimp.simple_cache(cache_type="json")
def get_job_word(job: List[Union[str, pd.Series]]) -> List[str]:
    job = _get_only_element(job)
    if isinstance(job, pd.Series):
        job = job["PrimaryPosition"]
    hdbg.dassert_isinstance(job, str)
    txt = rf"""
    For the profession '{job}', emit less than 3 words that can replace XYZ in the
    phrase "I noticed your work around XYZ". Only emit the words, no other text.

    <example1>
    Vice President of Business Development and Corporate Strategy, Americas
    -> business development
    <example2>
    Chief Technology Officer
    -> technology
    <example4>
    Vice President of Strategy and Business Development
    -> strategy
    """
    txt = txt.format(job=job)
    txt = hprint.dedent(txt)
    # Process with LLM.
    model = "gpt-4o-mini"
    use_llm_executable = False
    response = hllmcli.apply_llm(
        txt,
        model=model,
        use_llm_executable=use_llm_executable,
    )
    response = [response]
    return response


def get_LIn_connect_message(rows: List[pd.Series]) -> List[str]:
    row = _get_only_element(rows)
    hdbg.dassert_isinstance(row, pd.Series)
    first_name = row["FirstName"]
    company = row["PrimaryCompany"]
    job_word = row["job_word"]
    txt = rf"""
    Hi {first_name},

    I'm GP Saggese, and I teach AI at the Univ of Maryland, doing research on
    causal AI and LLMs.

    Given your work at {company} in {job_word}, I thought to connect with you.

    Thanks,
    GP
    """
    txt = txt.format(first_name=first_name, company=company, job_word=job_word)
    txt = hprint.dedent(txt)
    hdbg.dassert_lte(len(txt), 270, "LinkedIn message is too long")
    return [txt]


def get_email_connect_message(rows: List[pd.Series]) -> List[str]:
    row = _get_only_element(rows)
    hdbg.dassert_isinstance(row, pd.Series)
    first_name = row["FirstName"]
    company = row["PrimaryCompany"]
    job_word = row["job_word"]
    txt = rf"""
    Hi {first_name},

    I'm GP Saggese, and I teach AI at the Univ of Maryland, doing research on
    causal AI and LLMs.

    Given your work at {company} in {job_word}, I thought to connect with you.

    Here is my LinkedIn profile: https://www.linkedin.com/in/gpsaggese/
    Please feel free to connect with me.

    Thanks,
    GP
    """
    txt = txt.format(first_name=first_name, company=company, job_word=job_word)
    txt = hprint.dedent(txt)
    return [txt]


# #############################################################################


#


# TODO(gp): This overlaps with hllmcli.apply_llm_prompt_to_df.
def apply_func_to_df(
    df: pd.DataFrame,
    func: Callable[[List[pd.Series]], List[str]],
    target_col: str,
    *,
    batch_size: int = 1,
) -> pd.DataFrame:
    """
    Apply a function to each row or batch of rows in a dataframe.

    Supports batching multiple rows together for efficient processing
    (e.g., when using LLMs).

    :param df: input dataframe
    :param func: function to apply (accepts list of rows, returns list
        of results)
    :param target_col: name of column to store results
    :param batch_size: number of rows to process in each batch (default
        1)
    :return: dataframe with new target column
    """
    hdbg.dassert_lt(0, batch_size, "batch_size must be positive:", batch_size)
    results = []
    if batch_size == 1:
        # Single row processing (backwards compatible).
        _LOG.debug("Processing %d rows individually", len(df))
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            batch_results = func([row])
            hdbg.dassert_isinstance(batch_results, list)
            hdbg.dassert_eq(len(batch_results), 1)
            results.append(batch_results[0])
    else:
        # Batch processing.
        num_batches = (len(df) + batch_size - 1) // batch_size
        _LOG.info(
            "Processing %d rows in %d batches of size %d",
            len(df),
            num_batches,
            batch_size,
        )
        for i in tqdm(range(0, len(df), batch_size)):
            # Get batch of rows.
            batch_end = min(i + batch_size, len(df))
            batch_rows = [df.iloc[j] for j in range(i, batch_end)]
            _LOG.debug(
                "Processing batch %d-%d (%d rows)",
                i,
                batch_end - 1,
                len(batch_rows),
            )
            # Call function with batch.
            batch_results = func(batch_rows)
            # Validate results.
            hdbg.dassert_isinstance(
                batch_results, list, "Batched function must return list"
            )
            hdbg.dassert_eq(
                len(batch_results),
                len(batch_rows),
                "Number of results doesn't match batch size",
            )
            # Add results.
            results.extend(batch_results)
    # Verify we got all results.
    hdbg.dassert_eq(
        len(results),
        len(df),
        "Number of results doesn't match dataframe length:",
    )
    df[target_col] = results
    return df
