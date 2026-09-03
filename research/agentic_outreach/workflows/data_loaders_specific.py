"""
All the functions to load data from Google Sheets into a Contact_df.

Import as:

import ck_marketing.workflows.data_loaders_specific as cmwdlosp
"""

import logging

import numpy as np
import pandas as pd
from IPython.display import display
from tqdm.autonotebook import tqdm

import helpers.hdbg as hdbg
import helpers.hgoogle_drive_api as hgoogdr
import helpers.hprint as hprint

import ck_marketing.workflows.data_loaders_utils as cmwdlout

_LOG = logging.getLogger(__name__)


# #############################################################################
# LinkedIn Sales Navigator.
# #############################################################################


# error
# baseUrl                              https://www.linkedin.com/sales/lead/ACwAAATw5f...
# timestamp                            2023-11-09T20:34:20.691Z
# linkedinProfileUrl                   https://www.linkedin.com/in/adam-alfi-52891823/
# email                                aalfi@iconiqcapital.com
# linkedinProfile                      https://www.linkedin.com/in/adam-alfi-52891823/
# description                          Partner at ICONIQ / Growth Stage Technology In...
# headline                             Partner at ICONIQ
# location                             San Francisco, California, United States
# imgUrl
# firstName                            Adam
# lastName                             Alfi
# fullName                             Adam Alfi
# subscribers                          6445
# connectionDegree                     2nd
# vmid                                 ACoAAATw5fABJ5ZF-SwciO6SM_wl4NHzQsymiys
# userId                               82896368
# linkedinSalesNavigatorUrl            https://www.linkedin.com/sales/people/ACoAAATw...
# connectionsCount                     500
# connectionsUrl                       https://www.linkedin.com/search/results/people...
# mutualConnectionsUrl                 https://www.linkedin.com/search/results/people...
# mutualConnectionsText                Greg Kotchick is a mutual connection
# mailFromDropcontact                  aalfi@iconiqcapital.com
# company                              ICONIQ Capital
# companyUrl                           https://www.linkedin.com/company/6376200/
# jobTitle                             Partner
# jobDescription
# jobLocation
# jobDateRange                         Dec 2022 - Present
# jobDuration                          1 yr
# company2                             ICONIQ Capital
# companyUrl2                          https://www.linkedin.com/company/6376200/
# jobTitle2                            Principal, ICONIQ Growth
# jobLocation2                         San Francisco Bay Area
# jobDateRange2                        2021 - Dec 2022
# jobDuration2                         2 yrs
# school                               Georgetown University
# schoolUrl                            https://www.linkedin.com/company/4794/
# schoolDegree
# schoolDateRange
# school2                              ESCI-UPF
# schoolDegree2
# schoolDateRange2
# qualificationFromDropContact         nominative@pro
# civilityFromDropContact              Mr
# phoneNumberFromDropContact           +1 415-967-7763
# websiteFromDropContact               www.iconiqcapital.com
# twitter
# twitterProfileUrl
# website
# birthday
# companyWebsite                       http://www.iconiqcapital.com
# allSkills                            Spanish, Operations Management, Data Analysis,...
# skill1                               Spanish
# endorsement1
# skill2                               Operations Management
# endorsement2
# skill3                               Data Analysis
# endorsement3
# skill4                               Financial Analysis
# endorsement4
# skill5                               Logistics Management
# endorsement5
# skill6                               Manufacturing
# endorsement6
# profileId                            adam-alfi-52891823
# schoolUrl2                           https://www.linkedin.com/company/701578/
# jobDescription2                      ICONIQ Growth is a tech focused direct investm...
# schoolDescription
# schoolDescription2
# mail
# phoneNumber
# facebookUrl


def get_data_from_PB_SalesNavigator_Connections_Export(
    normalize: bool = True, verbose: bool = False
) -> pd.DataFrame:
    """
    Scrape data from LinkedIn Sales Navigator.

    E.g., Search4.FinTech_VC_in_US.SalesNavigator.
    """
    # Read two tabs from a gsheet.
    url = (
        "https://docs.google.com/spreadsheets/d/1Lbnyvbb28Cv-y0k"
        "-nrG1NSES9F6rxesGoHZV2LOJ6wA/"
    )
    tab_names = ("ScrapeProfile", "ScrapeProfile2")
    dfs = []
    for tab_name in tab_names:
        df_tmp = hgoogdr.get_cached_gsheet_to_df(url, tab_name)
        if verbose:
            display(df_tmp.head(1))
        dfs.append(df_tmp)
    df = pd.concat(dfs).drop_duplicates()
    df = df.loc[:, df.columns.str.strip() != ""]
    #
    if normalize:
        df["origin"] = "Search4.FinTech_VC_in_US"
        cols_map = {
            "timestamp": "origin_timestamp",
            "origin": None,
            "linkedinProfileUrl": "linkedin_url",
            "firstName": "first_name",
            "lastName": "last_name",
            "email": None,
            "jobTitle": "job_title",
            "description": "biography",
            "jobLocation": "city",
            "company": "company_name",
            # "companyWebsite": "company_domain",
        }
        df_out = cmwdlout.normalize_contact_schema(df, cols_map)
    else:
        df_out = df
    return df_out


# #############################################################################
# LinkedIn Sales Navigator 2.
# #############################################################################


# profileUrl                  https://www.linkedin.com/sales/lead/ACwAAABGC0...
# fullName                    Zhenya Loginov
# firstName                   Zhenya
# lastName                    Loginov
# companyName                 Accel
# title                       Partner
# companyId                   17412
# companyUrl                  https://www.linkedin.com/sales/company/17412
# regularCompanyUrl           https://www.linkedin.com/company/17412
# summary
# titleDescription            I invest and help European and Israeli founder...
# industry                    Venture Capital and Private Equity Principals
# companyLocation             Palo Alto, California, United States
# location                    Palo Alto, California, United States
# durationInRole              10 months in role
# durationInCompany           10 months in company
# pastExperienceCompanyName
# pastExperienceCompanyUrl
# pastExperienceCompanyTitle
# pastExperienceDate
# pastExperienceDuration
# connectionDegree            Out of Network
# profileImageUrl             https://media.licdn.com/dms/image/C5103AQHLVkA...
# sharedConnectionsCount      0
# name                        Zhenya Loginov
# vmid                        ACwAAABGC0cBTaYzMWKXyySFD4zZyoPI59OadWk
# linkedInProfileUrl          https://www.linkedin.com/in/ACwAAABGC0cBTaYzMW...
# isPremium                   TRUE
# isOpenLink                  FALSE
# query                       https://www.linkedin.com/sales/search/people?q...
# timestamp                   2024-07-11T18:03:01.973Z
# defaultProfileUrl           https://linkedin.com/in/zhenyaloginov
# hunter_extracted_email      zloginov@accel.com
# hunter_verification         valid
# dropcontact_mail            NaN
# all_emails                  NaN


def _extract_and_validate_email(df: pd.DataFrame) -> str:
    """
    Extract and validate the email from a transposed DataFrame.
    """
    # List of possible email sources.
    email_fields = [
        "hunter_extracted_email",
        "dropcontact_mail",
        "all_emails",
    ]
    # Extract email values from the relevant fields.
    email_values = df[email_fields]
    num_values = [
        set([str(v) for v in row if (str(v) != "" and str(v) != "nan")])
        for _, row in email_values.iterrows()
    ]
    num_values_out = []
    for val in num_values:
        if len(val) > 1:
            raise ValueError(
                "Multiple emails found: {%s}" % ", ".join(sorted(val))
            )
        elif len(val) == 1:
            val = list(val)[0]
        elif len(val) == 0:
            val = "nan"
        num_values_out.append(val)
    return num_values_out


def get_data_from_LinkedIn2(
    normalize: bool = True, verbose: bool = False
) -> pd.DataFrame:
    """
    E.g., Accel_search_export.gsheet.
    """
    # Accel_search_export.gsheet
    # Andreessen Horowitz (a16z)_search_export.gsheet
    # Benchmark Capital_search_export.gsheet
    # Bessemer Venture Partners_search_export.gsheet
    # General Catalyst_search_export.gsheet
    # Greylock Partners_search_export.gsheet
    # Index Ventures_search_export.gsheet
    # Insight Partners_search_export.gsheet
    # Kleiner Perkins_search_export.gsheet
    # Sequoia Capital_search_export.gsheet
    #
    # > find /Users/saggese/Library/CloudStorage/GoogleDrive-gp@kaizen-tech.io/Shared\ drives/Cold\ outreach/\!All_VC_lists -name "*search_export*" -print0 | sort -z  | xargs -0 -n 1 cat
    # {"":"WARNING! DO NOT EDIT THIS FILE! ANY CHANGES MADE WILL BE LOST!","doc_id":"1tgKVlDMVJJkPPyulTU1ibkBcGLKXeVjLTYN8jhQ9c3M","resource_key":"","email":"gp@kaizen-tech.io"}
    # https://docs.google.com/spreadsheets/d/1tgKVlDMVJJkPPyulTU1ibkBcGLKXeVjLTYN8jhQ9c3M/edit?gid=1151734371#gid=1151734371
    # {"":"WARNING! DO NOT EDIT THIS FILE! ANY CHANGES MADE WILL BE LOST!","doc_id":"1p7mKeeUuUS4a2OsHnTbWpe5Fscp8mvjocsWRL8ya6PA","resource_key":"","email":"gp@kaizen-tech.io"}
    urls = """
1tgKVlDMVJJkPPyulTU1ibkBcGLKXeVjLTYN8jhQ9c3M
1p7mKeeUuUS4a2OsHnTbWpe5Fscp8mvjocsWRL8ya6PA
1Iz9ypwENHwSU-meGknkrH_q7Gbg-CK6pArMQcxNvF3s
1y3bVUkC2qaZWFwkY9xvOcoUnqXEIDEC06mjUdiwXNNc
1RQMhAOpiu8BTyiNUl9O6DrmovEYAFqPPyBhiPD9uEZA
197W6s8K4tOzSdoT11rk3huGTxlttzXxpkWruHHzyeDQ
1xSYf8Hzg7vPmP_pSBe4NMkANX013FuTQR2SJgOr8AqU
1Gf6dVplfK-ufHoGdY3RchlcapQY2Ig5gopM2YMOTuz4
1EzsnB-a0cmiWpl2A9-McNrTR4D_jXJY9UB0byy8PuIA
1Ric5JLQOwkj9m4iwZtSo46VzOI0XDEMdeJZBEO4fPQ8
1rmImy9VByGf1cNKbYmUVh7ktojtRpXL4QLdvPLAB8C4
"""
    urls = urls.split()
    urls = [
        "https://docs.google.com/spreadsheets/d/" + url
        for url in urls
        if url != ""
    ]
    _LOG.debug("urls=\n%s", urls)
    dfs = []
    # urls = urls[:2]
    for url in tqdm(urls):
        # _LOG.debug("Reading %s", url)
        df = hgoogdr.get_cached_gsheet_to_df(url, "hunter_verification")
        if verbose:
            display(df.head(1))
        dfs.append(df)
        # time.sleep(20)
    # Concat.
    df2 = pd.concat(dfs, axis=0)
    if verbose:
        display(df2.head(2))
    #
    df2["email"] = _extract_and_validate_email(df2)
    # Convert to contact schema.
    if normalize:
        df2["origin"] = "VC_search_export"
        cols_map = {
            "timestamp": "origin_timestamp",
            "origin": None,
            "linkedInProfileUrl": "linkedin_url",
            "firstName": "first_name",
            "lastName": "last_name",
            "email": None,
            "hunter_verification": "email_verification",
            "title": "job_title",
            "titleDescription": "biography",
            "companyName": "company_name",
            "companyLocation": "city",
        }
        df_out = cmwdlout.normalize_contact_schema(df2, cols_map)
    else:
        df_out = df2
    return df_out


# #############################################################################
# LinkedIn Sales Navigator 3.
# #############################################################################


# linkedinProfileUrl                   https://www.linkedin.com/in/robtoews/
# email
# firstName                            Rob
# lastName                             Toews
# company                              Radical Ventures
# jobTitle                             Partner
# Email                                rob@radical.vc
# Score                                97
# Verification status                  valid
# Position
# Twitter
# Linkedin
# Phone number
# Company                              Radical Ventures
# Source 1                             http://weeklyvoice.com/canadian-companies-prio...
# Source 2                             http://salt.org/speakers/rob-toews
# Source 3                             http://radical.vc/how-accurate-were-our-2023-a...
# Source 4                             http://radical.vc/neurips-2022-and-whats-next-...
# Source 5                             http://nationalposttoday.com/ai-skills-in-dema...


def get_data_from_LinkedIn3(
    normalize: bool = True, verbose: bool = False
) -> pd.DataFrame:
    """
    Search7.AI_VC_in_US.gsheet.
    """
    url = "https://docs.google.com/spreadsheets/d/12qmHUo6sTuFpLQcdw22EaA3xsJ9ERuiCJdhDyF-2elM"
    tab_name = "test-590999-valid"
    df = hgoogdr.get_cached_gsheet_to_df(url, tab_name)
    # TODO(gp): Eisenbug. It's unclear why there is this column due to caching.
    if "email" in df.columns:
        del df["email"]
    if verbose:
        display(df.head(1))
    if normalize:
        df["origin"] = "Search7.AI_VC_in_US"
        cols_map = {
            "origin": None,
            "linkedinProfileUrl": "linkedin_url",
            "firstName": "first_name",
            "lastName": "last_name",
            "Email": "email",
            "Verification status": "email_verification",
            "Position": "biography",
            "company": "company_name",
        }
        df_out = cmwdlout.normalize_contact_schema(df, cols_map)
    else:
        df_out = df
    return df_out


# #############################################################################
# LinkedIn Sales Navigator 4.
# #############################################################################


# email_first           aagarwal@insightpartners.com
# first_name                            Anika
# last_name                           Agarwal
# job_title                 Managing Director
# company_name               Insight Partners
# company_domain          insightpartners.com
# city            New York, New York, United States
# linkedin_id                             NaN
# created_date                            NaN
# list_name                               NaN
# YAMM                                    NaN
# email_second                            NaN
# phone                                   NaN
# company_phone                           NaN
# middle_name                             NaN
# url                                     NaN
# company_id                              NaN
# hunter_verification              accept_all


def get_data_from_LinkedIn4(
    normalize: bool = True, verbose: bool = False
) -> pd.DataFrame:
    """
    VC Tier 1 - Partners
    """
    url = "https://docs.google.com/spreadsheets/d/1WU_u-4gKDb5NE-u1xwMrkFWDT77PoNQuku0pcj3fbNY/edit?gid=1063998981#gid=1063998981"
    #
    dfs = []
    for tab_name in ("Sheet1", "Sheet2", "Sheet3"):
        df_tmp = hgoogdr.get_cached_gsheet_to_df(url, tab_name)
        if verbose:
            display(df_tmp.head(1))
        dfs.append(df_tmp)
    df = pd.concat(dfs)
    #
    df_tmp = hgoogdr.get_cached_gsheet_to_df(url, "validity_merged_df")
    df = df.merge(
        df_tmp[["email_first", "hunter_verification"]],
        how="outer",
        on="email_first",
    )
    #
    if normalize:
        df["origin"] = "VC Tier 1"
        cols_map = {
            "origin": None,
            "url": "linkedin_url",
            "first_name": None,
            "last_name": None,
            "email_first": "email",
            "hunter_verification": "email_verification",
            "job_title": None,
            "company_name": None,
            "company_domain": None,
            "city": None,
        }
        df_out = cmwdlout.normalize_contact_schema(df, cols_map)
    else:
        df_out = df
    return df_out


# #############################################################################
# LinkedIn Sales Navigator 5.
# #############################################################################


# email_first               rodriguez@foundersfund.com
# First name                                 Rodriguez
# last_name                                       Keig
# job_title                                    Founder
# company_name                           Founders Fund
# company_domain                      foundersfund.com
# city               New York, New York, United States
# linkedin_id                               1107202338
# created_date                              2024-05-13
# list_name                             VC List Tier 2
# docsend_link  https://docsend.com/view/v9itej52tumaupih?emai...
# link                               Kaizen pitch deck
# Merge status                                 BOUNCED
# email_verification                           invalid


def get_data_from_LinkedIn5(
    normalize: bool = True, verbose: bool = False
) -> pd.DataFrame:
    """
    VC_Tier_2_Partners.
    """
    url = "https://docs.google.com/spreadsheets/d/17gxl8o_lS9zOsuJMmX1CSZJfZUBx0UZ66eaAlYutfaA/edit?gid=1525085692#gid=1525085692"
    #
    dfs = []
    target_tab_names = "VC-20240525-1 VC-20240520-4 VC-20240520-3 VC-20240520-2 VC-20240520".split()
    for tab_name in target_tab_names:
        df_tmp = hgoogdr.get_cached_gsheet_to_df(url, tab_name)
        if verbose:
            display(df_tmp.head(1))
        dfs.append(df_tmp)
    df = pd.concat(dfs)
    #
    df["email_verification"] = np.where(
        df["Merge status"] != "BOUNCED", "valid", "invalid"
    )
    #
    if normalize:
        df["origin"] = "VC Tier 2"
        cols_map = {
            "origin": None,
            "First name": "first_name",
            "last_name": "last_name",
            "email_first": "email",
            "email_verification": None,
            "job_title": None,
            "company_name": None,
            "company_domain": None,
            "city": None,
        }
        df_out = cmwdlout.normalize_contact_schema(df, cols_map)
    else:
        df_out = df
    return df_out


# #############################################################################
# VCSheet.
# #############################################################################

# Name                                             Michael Gilroy
# First name                                              Michael
# Title         Co-COO of Growth, Co-Head of Fintech, General ...
# Email                                        mgilroy@coatue.com
# Connect                                         Connect20240530


def _get_last_name(x: str) -> str:
    x2 = x.split()
    if len(x2) > 1:
        return " ".join(x2[1:])
    else:
        return x


def get_data_from_VCSheet(
    normalize: bool = True, verbose: bool = False
) -> pd.DataFrame:
    """
    VCSheet_Query1.
    """
    url = (
        "https://docs.google.com/spreadsheets/d"
        "/1U8jJYZbC1oyZpsSWhCCe6SDpAH8e6R2yRDeFauHVcj0/edit?gid=1769984900"
        "#gid=1769984900"
    )
    #
    tab_name = "Sheet1"
    df = hgoogdr.get_cached_gsheet_to_df(url, tab_name)
    if verbose:
        display(df.head(1))
    #
    df["last_name"] = df["Name"].apply(lambda x: _get_last_name(x))
    df["company_name"] = df["Title"].apply(lambda x: x.split("@")[1])
    #
    if normalize:
        df["origin"] = "VCSheet_Query1"
        cols_map = {
            "origin": None,
            "LinkedIn": "linkedin_url",
            "First name": "first_name",
            "last_name": "last_name",
            "Email": "email",
            "Title": "job_title",
            "company_name": None,
        }
        df_out = cmwdlout.normalize_contact_schema(df, cols_map)
    else:
        df_out = df
    return df_out


# #############################################################################
# Euro-VCs.
# #############################################################################

# Investor                             Angels FTW
# Domain                               https://www.joinodin.com/
# Main Country                         Worldwide
# Investor Category                    Angel Investment Group,Angel Investor,Venture ...
# Overview                             Keep your cap table clean with Odin. Learn mor...
# Main City
# Industries                           Administrative Services, Agriculture and Farm...
# Stages                               Seed,Convertible Note,Series A,Pre-Seed
# Fund Restrictions
# Fund Restriction Notes
# Contact 1 First Name
# Contact 1 Last Name
# Contact 1 Email
# Contact 1 Linkedin
# Contact 1 Title
# Contact 2 First Name
# Contact 2 Last Name
# Contact 2 Email
# Contact 2 Linkedin
# Contact 2 Title
# Portfolio
# Categories
# Second Country
# Second City
# Approved                             true
# Created                              2022-11-28T16:51:37.000Z
# origin                               Euro-VC-LinkedIn


def get_data_from_EuroVC(
    normalize: bool = True, verbose: bool = False
) -> pd.DataFrame:
    """
    Euro-VCs.
    """
    url = "https://docs.google.com/spreadsheets/d/1r3_drVggRB61KvNPxlf58sEwNoUvU_aquLwGLDyUSKI"
    #
    tab_name = "Sheet1"
    df = hgoogdr.get_cached_gsheet_to_df(url, tab_name)
    if verbose:
        display(df.head(1))
    #
    _LOG.debug("%s", df.shape)
    for col in df.columns:
        df[col] = df[col].str.replace(r"[^\x00-\x7F]+", "", regex=True)
    if verbose:
        display(df.head(1))
    df["stages"] = df["Stages"] + "/" + df["Investor Category"]
    df["restrictions"] = (
        df["Fund Restrictions"] + "." + df["Fund Restriction Notes"]
    )
    # Reshape the DataFrame by pivoting Contact 1 and Contact 2 into rows.
    cols = [x for x in df.columns if not x.startswith("Contact ")]
    _LOG.debug("%s", cols)
    df_contact1 = df[
        cols
        + [
            "Contact 1 First Name",
            "Contact 1 Last Name",
            "Contact 1 Email",
            "Contact 1 Linkedin",
            "Contact 1 Title",
        ]
    ].rename(columns=lambda x: x.replace("Contact 1 ", ""))
    df_contact2 = df[
        cols
        + [
            "Contact 2 First Name",
            "Contact 2 Last Name",
            "Contact 2 Email",
            "Contact 2 Linkedin",
            "Contact 2 Title",
        ]
    ].rename(columns=lambda x: x.replace("Contact 2 ", ""))
    # Concatenate the two DataFrames to stack them as rows
    df = pd.concat([df_contact1, df_contact2], ignore_index=True)
    valid_mask = df["First Name"] != ""
    _LOG.debug(
        "Removed %s rows with empty first name",
        hprint.perc((~valid_mask).sum(), df.shape[0]),
    )
    df = df[valid_mask]
    if verbose:
        display(df.head(1))
    #
    if normalize:
        df["origin"] = "Euro-VC-LinkedIn"
        cols_map = {
            "origin": None,
            "Linkedin": "linkedin_url",
            "First Name": "first_name",
            "Last Name": "last_name",
            "Email": "email",
            "Title": "job_title",
            "Domain": "company_name",
            # "stages": None,
            # "restrictions": "restrictions",
            "Created": "origin_timestamp",
            "Main City": "city",
            # "Overview": "notes",
        }
        df_out = cmwdlout.normalize_contact_schema(df, cols_map)
    else:
        df_out = df
    return df_out


# #############################################################################
# Folkapp.
# #############################################################################


# Person                               Vasudev Bailey
# First name                           Vasudev
# Email                                vb@av.co
# Urls                                 https://www.linkedin.com/in/baileyv
# Companies                            Artis Ventures (AV)
# Portfolio companies                  YouTube, Lemonaid Health, Activ Surgical, Tast...
# Fund type                            Venture Fund
# Fund stage                           Seed;Pre-Seed;Series A;Series B;Series C;Series D
# Fund focus                           Health;Entertainment & Media;AI & Machine Lear...
# Location                             San Francisco;California
# Twitter Link                         http://twitter.com/artisventures
# LinkedIn Link                        http://www.linkedin.com/company/artis-ventures
# Facebook Link                        http://www.facebook.com/pages/ARTIS-Ventures/3...
# Number of Investments                101
# Number of Exits                      27
# Fund Description                     ARTIS Ventures is a financial services firm th...
# Founding Year                        2001
# Description
# origin                               Folkapp


def _remove_duplicates(input_string: str) -> str:
    # Split the string into a list of words.
    words = input_string.split()
    # Use a set to track words we've seen.
    seen = set()
    result = []
    # Iterate over each word in the list.
    for word in words:
        # If the word hasn't been seen, add it to the result.
        if word.lower() not in seen:
            result.append(word)
            # Add the lowercase version to handle case-insensitive duplicates.
            seen.add(word.lower())
    return " ".join(result)


def get_data_from_FolkApp(
    normalize: bool = True, verbose: bool = False
) -> pd.DataFrame:
    """
    Folkapp.
    """
    # Read data.
    url = "https://docs.google.com/spreadsheets/d/1j6_mI5r05-P6smXMGOq3f0bpmyc7p2W0n7mY8gGbpY0/edit?gid=0#gid=0"
    tab_name = "Sheet1"
    df = hgoogdr.get_cached_gsheet_to_df(url, tab_name)
    if verbose:
        display(df.head(1))
    #
    if normalize:
        df["origin"] = "Folkapp"
        df["Person_modified"] = df["Person"].apply(_remove_duplicates)
        df["is_person_modified"] = df["Person"] != df["Person_modified"]

        def _extract_last_name(x):
            vals = x.split()
            if len(vals) > 1:
                return vals[-1]
            else:
                return ""

        df["last_name"] = [_extract_last_name(x) for x in df["Person"]]
        #
        cols_map = {
            "origin": None,
            "Urls": "linkedin_url",
            "First name": "first_name",
            "last_name": "last_name",
            "Email": "email",
            "Description": "job_title",
            "Companies": "company_name",
            # "Fund type": "category",
            # "Fund stage": "stages",
            # "Fund focus": "industry",
            "Location": "city",
        }
        df_out = cmwdlout.normalize_contact_schema(df, cols_map)
        # Some LinkedIn urls are websites and not LinkedIn urls.
        df_out["is_linkedin"] = [
            "linkedin.com" in x for x in df_out["linkedin_url"]
        ]
        srs_tmp = df_out["linkedin_url"].copy()
        df_out["linkedin_url"] = np.where(df_out["is_linkedin"], srs_tmp, "")
        df_out["company_domain"] = np.where(df_out["is_linkedin"], "", srs_tmp)
        del df_out["is_linkedin"]
    else:
        df_out = df
    return df_out


# #############################################################################
# Hedge fund list.
# #############################################################################


# Company       21st Century Digital Industries Fund
# Full Name                       Richard B. Steward
# First Name                              Richard B.
# Last Name                                  Steward
# Address 1                    960 Pines Lake Dr. W.
# Address2
# City                                         Wayne
# State                                           NJ
# Zip                                          07470
# Country                                         US
# Phone                                 973-839-8776
# Fax                                   973-839-2185
# Email                           rstew10446@aol.com


def get_data_from_hedge_fund_list(
    normalize: bool = True, verbose: bool = False
) -> pd.DataFrame:
    """
    Hedge fund list.
    """
    # Read data.
    url = "https://docs.google.com/spreadsheets/d/10h8NtGDx4GQL8JWWJHEkD1GS685nYIZOIJMXiiz7pqA"
    tab_name = "Hedge Fund Contacts"
    df = hgoogdr.get_cached_gsheet_to_df(url, tab_name)
    if verbose:
        display(df.head(1))
    #
    print("Input data...", df[:1].to_csv())
    if normalize:
        df["origin"] = "hedge_fund_list"
        print("Input data...", df[:1].to_csv())
        df["City"] = df["City"] + ", " + df["State"] + ", " + df["Country"]
        print("Input data...", df[:1].to_csv())
        # df["category"] = "hedge_fund"
        print("Input data...", df[:1].to_csv())
        #
        cols_map = {
            "origin": None,
            "First Name": "first_name",
            "Last Name": "last_name",
            "Email": "email",
            "Company": "company_name",
            # "category": None,
            "City": "city",
        }
        df_out = cmwdlout.normalize_contact_schema(df, cols_map)
        print("Input data...", df_out[:1].to_csv())
    else:
        df_out = df.copy()
    return df_out


# #############################################################################
# GP LinkedIn connections (after LinkedIn enrichment)
# #############################################################################

# companyIndustry                                                                 Design
# companyName                                                                   FYC Labs
# firstName                                                                       Justin
# lastName                                                                       Fortier
# linkedinCompanyUrl                               https://linkedin.com/company/fyc-labs
# linkedinCompanySlug                                                           fyc-labs
# linkedinFollowersCount                                                            3063
# linkedinHeadline                                          Founder + CEO/CTO @ FYC Labs
# linkedinIsHiringBadge                                                            FALSE
# linkedinIsOpenToWorkBadge                                                        FALSE
# linkedinJobDateRange                                                Oct 2012 - Present
# linkedinJobLocation                                          California, United States
# linkedinJobTitle                     Chief Executive Officer / Chief Technical Officer
# linkedinPreviousCompanySlug                                               opengrantsio
# linkedinPreviousJobDateRange                                        Nov 2022 - Present
# linkedinPreviousJobDescription
# linkedinPreviousJobTitle                                      Chief Technology Officer
# linkedinPreviousSchoolDegree                                           Master's degree
# linkedinProfileId                                                            244286678
# linkedinProfileSlug                                                     justinffortier
# linkedinProfileUrl                              https://linkedin.com/in/justinffortier
# linkedinProfileUrn                             ACoAAA6PhNYBqKRF_PWQKs6zWnxk3-WzKXiXN6k
# linkedinSchoolUrl                           https://linkedin.com/school/ucsantabarbara
# linkedinSchoolCompanySlug                                               ucsantabarbara
# linkedinSchoolDegree                                           Bachelor of Arts (B.A.)
# linkedinSchoolName                                                    UC Santa Barbara
# linkedinSkillsLabel                  Marketing, Leadership, Competitive Analysis, M...
# location                                             Folsom, California, United States
# previousCompanyName                                                         OpenGrants
# connectionDegree                                                                   1st
# refreshedAt                                                   2024-12-31T02:08:03.997Z
# mutualConnectionsUrl                 https://www.linkedin.com/search/results/people...
# connectionsUrl                       https://www.linkedin.com/search/results/people...
# linkedinConnectionsCount                                                           500
# profileUrl                                 https://www.linkedin.com/in/justinffortier/
# linkedinDescription                  I love creating and working with amazing compa...
# linkedinJobDescription               FYC is a web development and graphic design ag...
# linkedinPreviousJobLocation                          Folsom, California, United States
# linkedinPreviousSchoolUrl            https://linkedin.com/school/san-diego-state-un...
# linkedinPreviousSchoolCompanySlug                           san-diego-state-university
# linkedinPreviousSchoolDescription    The focus of my Masters Degree was on persuasi...
# linkedinPreviousSchoolName                                  San Diego State University
# linkedinSchoolDescription            UCSB Club Hockey, Law and Society Study Groups...
# linkedinPreviousSchoolDateRange
# linkedinSchoolDateRange


def get_data_from_GP_LIn_connections(
    normalize: bool = True, verbose: bool = False
) -> pd.DataFrame:
    """
    GP LinkedIn connections.
    """
    # Read data.
    url = "https://docs.google.com/spreadsheets/d/19ziUmqbPaUO73cqlJB1F9y-j1Oq98nMzo6wTmzyVnwg"
    tab_name = "Sheet1"
    df = hgoogdr.get_cached_gsheet_to_df(url, tab_name)
    if verbose:
        display(df.head(1))
    #
    if normalize:
        df["origin"] = "PB.LIN_Connections_Exports.GP_Lin_Connections_2024_12_31"
        df["biography"] = (
            df["linkedinHeadline"]
            + "; "
            + df["linkedinDescription"]
            + "; "
            + df["linkedinJobDescription"]
        )
        #
        cols_map = {
            "origin": None,
            # When it was loaded from the source.
            "refreshedAt": "origin_timestamp",
            "firstName": "first_name",
            "lastName": "last_name",
            # Email.
            # "email": "",
            # "email_verification": "",
            "linkedinProfileUrl": "linkedin_url",
            "linkedinJobTitle": "job_title",
            "biography": None,
            "companyName": "company_name",
            # "company_domain": "",
            "location": "city",
            # Seed,Convertible Note,Series A,Pre-Seed
            # "stages": "",
            # "restrictions": "",
            # Angel, VC, PE, Family Office, Corporate VC, Accelerator, Incubator
            # "companyIndustry": "category",
            # "notes": "",
        }
        df_out = cmwdlout.normalize_contact_schema(df, cols_map)
    else:
        df_out = df
    return df_out


# #############################################################################
# Super Networking.
# #############################################################################

# Name                                        Priyaluk (Neuy)
# Email                                       priyaluk_wij@tk-partners.net
# LinkedIn                                    https://www.linkedin.com
# Check sizes                                 100K USD min/ 2.5M USD max
# Open for more deals from other investors                        TRUE
# Ready to share my deal flow with others                         TRUE
# Can advise startups                                             TRUE
# Pre-seed                                                       FALSE
# Seed                                                            TRUE
# Series A                                                        TRUE
# Series B                                                       FALSE
# Series C+                                                      FALSE
# Other options
# Angel                                                           TRUE
# Angel Syndicate Lead                                           FALSE
# VC Fund                                                         TRUE
# Accelerator                                                     TRUE
# Family Office                                                   TRUE
# Private Equity Fund                                            FALSE
# Venture Studio                                                 FALSE
# Fund of Funds                                                  FALSE
# CVC                                                            FALSE
# Limited Partner                                                FALSE
# Other types
# Globally – everywhere                                          FALSE
# US                                                              TRUE
# Canada                                                         FALSE
# UK                                                              TRUE
# Europe                                                         FALSE
# Israel                                                         FALSE
# Latin America                                                  FALSE
# Middle East                                                    FALSE
# Africa                                                         FALSE
# Asia Pacific                                                    TRUE
# Other regions
# Agnostic – all industries                                      FALSE
# AI                                                             FALSE
# B2B                                                            FALSE
# B2C                                                            FALSE
# SaaS                                                           FALSE
# Fintech                                                        FALSE
# Healthcare                                                      TRUE
# Biotech                                                        FALSE
# Energy                                                          TRUE
# ClimateTech                                                    FALSE
# E-com & Retail                                                 FALSE
# Future of Work / HRtech                                        FALSE
# Mobility & Transportation                                      FALSE
# Marketing / Adtech                                             FALSE
# PropTech                                                        TRUE
# AgriTech                                                        TRUE
# SpaceTech                                                      FALSE
# Cybersecurity                                                  FALSE
# Blockchain / Crypto                                            FALSE
# Education                                                      FALSE
# Other sectors


def get_data_from_super_networking_gsheet(
    normalize: bool = True, verbose: bool = False
) -> pd.DataFrame:
    url = "https://docs.google.com/spreadsheets/d/1kxB15NcHcuhEVtD982qnvbx276J7fmDKvJx63rRpD0E/edit?gid=381241246#gid=381241246"
    df = hgoogdr.get_cached_gsheet_to_df(url, "List of investors")
    if verbose:
        display(df.head(1))
    # Add column.
    columns = df.iloc[0, :].tolist()
    columns = [v.split("\n")[0] if "\n" in v else v for v in columns]
    columns = [v.strip() for v in columns]
    df.columns = columns
    # Remove the first row.
    df = df.iloc[1:, :]
    if normalize:
        for col_name in df.columns:
            df[col_name] = df[col_name].str.strip()
        # Convert to true/false.
        for col_name in df.columns:
            # print("'%s': %s" % (col_name, df[col_name].unique()))
            if col_name in (
                "Name",
                "Email",
                "LinkedIn",
                "Check sizes",
                "Other options",
                "Other types",
                "Other regions",
                "Other sectors",
            ):
                continue
            # #col_name = "Pre-seed"
            hdbg.dassert_is_subset(df[col_name].unique(), ("TRUE", "FALSE"))
            df[col_name] = [(v == "TRUE") for v in df[col_name]]
        # Reindex.
        df.index = range(0, len(df))
        # Split names.
        df = cmwdlout.split_first_last_name(df, "Name")
        # Add
        df["origin"] = "super_networking"
        cols_map = {
            "origin": None,
            "LinkedIn": "linkedin_url",
            "first_name": None,
            "last_name": None,
            "Email": "email",
        }
        df = cmwdlout._rename_columns_to_contact_schema(df, cols_map)
        # df_out = normalize_contact_schema(df, cols_map)
    if verbose:
        display(df.head(1))
    return df


def filter_super_networking_gsheet(df: pd.DataFrame) -> pd.DataFrame:
    # ['Email', 'LinkedIn',
    #
    # 'Check sizes',
    # 'Open for more deals from other investors', 'Ready to share my deal flow with others', 'Can advise startups',
    # 'Pre-seed', 'Seed', 'Series A', 'Series B', 'Series C+', 'Other options',
    #
    # 'Angel', 'Angel Syndicate Lead', 'VC Fund', 'Accelerator', 'Family Office',
    # 'Private Equity Fund', 'Venture Studio', 'Fund of Funds', 'CVC', 'Limited Partner', 'Other types',
    #
    # 'Globally – everywhere', 'US', 'Canada', 'UK',
    # 'Europe', 'Israel', 'Latin America', 'Middle East', 'Africa', 'Asia Pacific',
    # 'Other regions',
    #
    # 'Agnostic – all industries', 'AI', 'B2B', 'B2C', 'SaaS',
    # 'Fintech', 'Healthcare', 'Biotech', 'Energy', 'ClimateTech', 'E-com & Retail',
    # 'Future of Work / HRtech', 'Mobility & Transportation', 'Marketing / Adtech',
    # 'PropTech', 'AgriTech', 'SpaceTech', 'Cybersecurity', 'Blockchain / Crypto',
    # 'Education', 'Other sectors']
    col_names = [
        "Name",
        "first_name",
        "last_name",
        "email",
        "linkedin_url",
        "Check sizes",
        "Seed",
        "Series A",
        "Globally – everywhere",
        "US",
    ]
    #
    mask1 = df["Seed"] | df["Series A"]
    print("mask1=", mask1.sum())
    #
    mask2 = df["Globally – everywhere"] | df["US"]
    print("mask2=", mask2.sum())
    #
    mask3 = None
    col_names_tmp = [
        "Agnostic – all industries",
        "AI",
        "B2B",
        "SaaS",
        "Fintech",
        "Energy",
        "ClimateTech",
        "E-com & Retail",
        "Future of Work / HRtech",
        "Mobility & Transportation",
        "Marketing / Adtech",
        "PropTech",
        "AgriTech",
    ]
    col_names.extend(col_names_tmp)
    for col_name in col_names_tmp:
        mask_tmp = df[col_name]
        # print(col_name, mask_tmp.sum())
        if mask3 is None:
            mask3 = mask_tmp
        else:
            mask3 |= mask_tmp
    print("mask3=", mask3.sum())
    #
    mask = mask1 & mask2 & mask3
    print("mask=", mask.sum())
    #
    df2 = df[col_names][mask]
    return df2
