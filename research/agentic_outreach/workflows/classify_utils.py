"""
Import as:

import ck_marketing.workflows.classify_utils as cmwoclut
"""

import logging
from typing import Optional, Union

import pandas as pd

import helpers.hdbg as hdbg
import helpers.hllm_cli as hllmcli
import helpers.hpandas as hpandas
import helpers.hprint as hprint

_LOG = logging.getLogger(__name__)

# #############################################################################
# Person industry
# #############################################################################

Industries = """
- Agriculture
- Automotive
- Construction
- Consumer Goods
- Education
- Energy & Utilities
- Engineering Services
- Event Management
- Financial Services
- Government & Nonprofits
- Healthcare
- Human Resources & Staffing
- IT - Hardware
- IT - Software
- IT - Cybersecurity
- IT - Cloud Services
- IT - Managed Services
- IT - Consulting & Integration
- IT - Support Services
- IT - Data & Analytics
- IT - DevOps & Automation
- Legal Services
- Logistics & Transportation
- Manufacturing
- Marketing & Advertising Agencies
- Media & Entertainment
- Pharmaceutical & Biotechnology
- Real Estate
- Retail & eCommerce
- Sports & Recreation
- Telecommunications
- Travel & Hospitality
"""


def get_person_industry_prompt() -> str:
    prompt = f"""
    Given the following list of industries with examples, classify the text into the
    corresponding industry:
    {Industries}

    You MUST report the industry exactly as one of the options above. Do not
    include any other text.
    If you are not sure about the industry, return "unknown".
    """
    prompt = hprint.dedent(prompt)
    return prompt


def extract_person_industry_from_df(obj: Union[str, pd.Series]) -> str:
    """
    Extract the industry fields from the object using the company name, company
    domain, and job title.
    """
    if isinstance(obj, pd.Series):
        txt = ""
        # Check that required columns exist.
        hdbg.dassert_in("company_name", obj.index)
        company_name = str(obj["company_name"])
        if company_name != "":
            txt += "company_name=" + company_name
        #
        hdbg.dassert_in("company_domain", obj.index)
        company_domain = str(obj["company_domain"])
        if company_domain != "":
            txt += " company_domain=" + company_domain
        #
        hdbg.dassert_in("job_title", obj.index)
        job_title = str(obj["job_title"])
        if job_title != "":
            txt += " job_title=" + job_title
    else:
        hdbg.dassert_isinstance(obj, str)
        txt = obj
    return txt


# #############################################################################
# Person type
# #############################################################################


def get_person_type_prompt() -> str:
    # From ck_marketing/plugins/pitchbook/position_departments.grouped.txt
    prompt = """
    Given the following list of departments with examples, classify the text
    into the corresponding department:

    - Student
       - E.g., PhD Student, Master's Student, Bachelor's Student, High School Student

    - Professor
       - E.g., Professor, Associate Professor, Assistant Professor, Professor of
       Practice

    - VC
        - E.g., Venture Capital, VC, Corporate VC, Accelerator, Incubator

    - Angel Investor
        - E.g., Angel, Angel Investor

    - Family office
        - E.g., Family Office, Family Office Investor, Family Office Partner

    - Private Equity
        - E.g., Private Equity, PE, Private Equity Investor, Private Equity Partner

    - Corporate Development
        - E.g., Acquisitions, Alliances, Business Development, Corporate
        Development, International, Investments, Mergers & Acquisitions,
        Planning, Strategy

    - Finance, Legal & Risk
        - E.g., Accounting, Finance, Investor Relations, Legal, Risk Management

    - People, Culture & Administration
        - E.g., Administration, Diversity & Inclusion, Human Resources

    - Technology
        - E.g., Engineering, Information Technology, Research, Research &
        Development, Technology

    - Product
        - E.g., Development, Merchandising, Product Development, Product
        Management, Products / Services

    - Operations, Supply Chain & Production
        - E.g., Logistics, Manufacturing, Operations, Production, Purchasing,
        Quality Assurance

    - Sales, Marketing & Communications
        - E.g., Communications, Marketing, Sales, Sales & Marketing

    - Client & Customer Services
        - E.g., Client Services

    You MUST report the department exactly as one of the options above. Do not
    include any other text.
    If you are not sure about the department, return "unknown".
    """
    prompt = hprint.dedent(prompt)
    return prompt


def extract_person_department_from_df(obj: Union[str, pd.Series]) -> str:
    if isinstance(obj, pd.Series):
        txt = ""
        # Check that required columns exist.
        hdbg.dassert_in("job_title", obj.index)
        job_title = str(obj["job_title"])
        if job_title != "":
            txt += "job_title=" + job_title
        #
        hdbg.dassert_in("biography", obj.index)
        biography = str(obj["biography"])
        if biography != "":
            txt += " biography=" + biography
    else:
        hdbg.dassert_isinstance(obj, str)
        txt = obj
    return txt


# #############################################################################
# Person executive type
# #############################################################################


def get_is_executive_prompt() -> str:
    prompt = """
    Given the following list of job titles with examples, classify the text
    into the corresponding job title:

    - CEO
    - COO
    - CTO
    - CSO
    - CFO
    - CMO
    - CPO
    - CIO
    - CHRO
    - Owner
    - Founder
    - Partner
    - Principal

    You MUST report the job title exactly as one of the options above. Do not
    include any other text.
    If you are not sure about the job title, return "unknown".
    """
    prompt = hprint.dedent(prompt)
    return prompt


def extract_is_executive_from_df(obj: Union[str, pd.Series]) -> str:
    if isinstance(obj, pd.Series):
        txt = ""
        # Check that required columns exist.
        hdbg.dassert_in("job_title", obj.index)
        job_title = str(obj["job_title"])
        if job_title != "":
            txt += "job_title=" + job_title
    else:
        hdbg.dassert_isinstance(obj, str)
        txt = obj
    return txt


# #############################################################################


def classify_industry_type_executive(
    df: pd.DataFrame,
    *,
    batch_size: int = 50,
    model: Optional[str] = "gpt-5-nano",
    display_stats: bool = True,
) -> pd.DataFrame:
    """
    Classify the industry, department, and executive type of a dataframe of
    LinkedIn connections.

    :param df: dataframe to classify
    :param batch_size: number of items to process in each batch
    :param model: model to use for classification
    :param display_stats: whether to display statistics of the classification
    :return: dataframe with the classified industry, department, and executive type
    """
    df = df.copy()
    _LOG.info(
        "Classifying industry, department, and executive type of %d items",
        len(df),
    )
    _LOG.info(hprint.to_str("batch_size model"))
    # 1) Classify the industry.
    prompt = get_person_industry_prompt()
    tag = "Classifying industry"
    batch_mode = "combined"
    target_col = "industry"
    df, stats = hllmcli.apply_llm_prompt_to_df(
        prompt,
        df,
        extract_person_industry_from_df,
        target_col,
        batch_mode,
        batch_size=batch_size,
        model=model,
        tag=tag,
    )
    _LOG.info("stats=%s", stats)
    # 2) Classify the department.
    prompt = get_person_type_prompt()
    tag = "Classifying department"
    batch_mode = "combined"
    target_col = "type"
    df, stats = hllmcli.apply_llm_prompt_to_df(
        prompt,
        df,
        extract_person_department_from_df,
        target_col,
        batch_mode,
        batch_size=batch_size,
        model=model,
        tag=tag,
    )
    _LOG.info("stats=%s", stats)
    # 3) Classify the executive type.
    prompt = get_is_executive_prompt()
    tag = "Classifying executive type"
    batch_mode = "combined"
    target_col = "executive"
    df, stats = hllmcli.apply_llm_prompt_to_df(
        prompt,
        df,
        extract_is_executive_from_df,
        target_col,
        batch_mode,
        batch_size=batch_size,
        model=model,
        tag=tag,
    )
    _LOG.info("stats=%s", stats)
    # Display statistics if requested.
    if display_stats:
        hpandas.display_value_counts_stats_df(
            df,
            col_names=["type", "industry", "executive"],
            num_rows=10,
        )
    return df
