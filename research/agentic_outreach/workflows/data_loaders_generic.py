"""
All the functions to load data from Google Sheets into a Contact_df.

Import as:

import ck_marketing.workflows.data_loaders_generic as cmwdloge
"""

import logging
from typing import List, Union

import pandas as pd
from IPython.display import display

import ck_marketing.workflows.data_loaders_utils as cmwdlout
import helpers.hdbg as hdbg
import helpers.hpandas as hpandas
import helpers.hgoogle_drive_api as hgodrapi

_LOG = logging.getLogger(__name__)


# #############################################################################
# Pitchbook.
# #############################################################################

# E.g., https://docs.google.com/spreadsheets/d/1aKzWUw9mwP-2_vzz27ggeLe1sgF9OrWRWmGMo9Dk9bU

# idx                                                                         0
# People                                                         Oliver Steinig
# LinkedIn URL                   http://linkedin.com/in/oliver-steinig-68603aa/
# Last Name                                                             Steinig
# First Name                                                             Oliver
# Primary Company                                            Bosch (Automotive)
# Primary Position            Vice President of Business Development and Cor...
# Biography                   Ms. Oliver Steinig serves as Vice President of...
# Board Seats
# Roles                                                                       1
# Deal Roles
# Location                                                   Gerlingen, Germany
# Address Line 1                                           Robert-Bosch-Platz 1
# Address Line 2                                                  Schillerh��he
# City                                                                Gerlingen
# State / Province
# Post Code                                                               70839
# Country / Territory / Region                                          Germany
# Phone                                                                 #ERROR!
# Fax                                                                   #ERROR!
# Email                                                oliver.steinig@bosch.com


def get_data_from_pitchbook(
    url: str,
    tab_name: str,
    timestamp: pd.Timestamp,
    tag: str,
    normalize: bool,
    *,
    allow_subset: bool = False,
    remove_spaces_in_cols: bool = True,
    verbose: bool = False,
    force_no_cache: bool = False,
) -> pd.DataFrame:
    """
    Load and optionally normalize data from Pitchbook Google Sheets.

    :param url: The URL of the Pitchbook sheet.
    :param tab_name: The name of the sheet in the Pitchbook.
    :param timestamp: The timestamp to associate with the data.
    :param tag: The tag of the data (e.g., "200AICompanies.MA").
    :param normalize: Whether to normalize the data.
    :param allow_subset: Whether to allow the subset of columns that are in the
        schema.
    :param remove_spaces_in_cols: Whether to remove spaces in the column names.
    :param verbose: Whether to print the data.
    :param force_no_cache: Whether to bypass the cache and fetch fresh data.
    :return: A dataframe with the data.
    """
    _ = tag
    credentials = hgodrapi.get_credentials()
    # TODO(gp): Remove / factor out this.
    # if tab_name == "_all_":
    #     # Get all the sheets from the gsheet.
    #     tabs = hgodrapi.get_tabs_from_gsheet(url, credentials=credentials)
    #     _LOG.info("tabs=%s", tabs)
    #     dfs = []
    #     for tab in tabs:
    #         _LOG.info("tab=%s", tab)
    #         df = ckmktwfutils.get_gsheet_to_df(url, tab, remove_spaces_in_cols=remove_spaces_in_cols, force_no_cache=force_no_cache)
    #         _LOG.info(
    #             "df.shape=%s\ndf.columns=%s", df.shape, df.columns.tolist()
    #         )
    #         df["origin"] = "Pitchbook." + tab
    #         dfs.append(df)
    #     df = pd.concat(dfs)
    # else:
    # Load the raw data from the Google Sheet.
    df = hgodrapi.get_gsheet_to_df(
        url,
        tab_name,
        remove_spaces_in_cols=remove_spaces_in_cols,
        force_no_cache=force_no_cache,
    )
    if verbose:
        display(df.head(1))
    # Normalize the data if requested.
    if normalize:
        hdbg.dassert_isinstance(timestamp, pd.Timestamp)
        df["origin_timestamp"] = str(timestamp)
        df["email_timestamp"] = str(timestamp)
        # Define the column mapping from Pitchbook schema to our standard
        # schema.
        cols_map = {
            "origin": None,
            "origin_timestamp": None,
            "email_timestamp": None,
            "First Name": "first_name",
            "Last Name": "last_name",
            "LinkedIn URL": "linkedin_url",
            "Primary Position": "job_title",
            "Primary Company": "company_name",
            # "company_domain": "",
            "City": "city",
            "Email": "email",
            "Country/Territory/Region": "country",
        }
        df_out = cmwdlout.normalize_contact_schema(
            df, cols_map, allow_subset=allow_subset
        )
    else:
        df_out = df
    return df_out


# #############################################################################
# LinkedIn Connections Export
# #############################################################################

# profileUrl                    https://www.linkedin.com/in/grahamcpeck/
# firstName                                                    Graham C.
# lastName                                                          Peck
# fullName                                                Graham C. Peck
# title                Co-Founder @ DealSend & Attaq Vector | Partner...
# connectionSince                               2024-12-26T21:09:08.000Z
# profileImageUrl      https://media.licdn.com/dms/image/v2/C5603AQHT...
# timestamp                                     2024-12-29T01:14:30.420Z
# connectedProfileUrl                  https://linkedin.com/in/gpsaggese
# connectedUsername                               Giacinto Paolo Saggese


def get_data_from_LinkedIn_Connections_Exports(
    url: str,
    tab_name: str,
    tag: str,
    normalize: bool,
    *,
    allow_subset: bool = False,
    remove_spaces_in_cols: bool = True,
    verbose: bool = False,
    force_no_cache: bool = False,
) -> pd.DataFrame:
    """
    Load LinkedIn connections data from PhantomBuster LinkedIn Outreach export.

    This loads connection data before LinkedIn enrichment is applied.

    :param url: The URL of the Google sheet containing the data.
    :param tab_name: The name of the Google sheet tab.
    :param tag: The tag of the data (e.g., "200AICompanies.MA").
    :param normalize: Whether to normalize the data.
    :param allow_subset: Whether to allow the subset of columns that are in the
        schema.
    :param remove_spaces_in_cols: Whether to remove spaces in the column names.
    :param verbose: Whether to print the data.
    :param force_no_cache: Whether to bypass the cache and fetch fresh data.
    :return: A dataframe with the data.
    """
    # Load the raw data from the Google Sheet.
    df = hgodrapi.get_gsheet_to_df(
        url,
        tab_name,
        remove_spaces_in_cols=remove_spaces_in_cols,
        force_no_cache=force_no_cache,
    )
    if verbose:
        display(df.head(1))
    # Normalize the data if requested.
    if normalize:
        # Set the origin and timestamp columns.
        df["origin"] = "PB.LIN_Connections_Exports." + tag
        df["origin_timestamp"] = df["timestamp"]
        df["email_timestamp"] = df["timestamp"]
        # Define the column mapping from LinkedIn Connections schema to our
        # standard schema.
        cols_map = {
            "origin": None,
            "origin_timestamp": None,
            "email_timestamp": None,
            "profileUrl": "linkedin_url",
            "firstName": "first_name",
            "lastName": "last_name",
            "title": "job_title",
        }
        df_out = cmwdlout.normalize_contact_schema(
            df, cols_map, allow_subset=allow_subset
        )
    else:
        df_out = df
    return df_out


# #############################################################################
# Sales Navigator Search Results.
# #############################################################################

# Scrape and export the results of a Sales Navigator search into a spreadsheet


# profileUrl	        https://www.linkedin.com/sales/lead/ACwAACIIGwQBarOTcv9cBkPkS4qb143voXG2rdA,NAME_SEARCH,je9Y
# fullName	            Fawaz Ahmar
# firstName	            Fawaz
# lastName	            Ahmar
# companyName	        PwC
# title	                Senior Director - FS / M&A Advisory
# companyId	            1044
# companyUrl	        https://www.linkedin.com/sales/company/1044
# regularCompanyUrl	    https://www.linkedin.com/company/1044
# summary               "Fawaz is an experienced Senior Director within the Financial Services and M&A Advisory consulting group at PwC with a focus on Deals and Technology Transformation and Corporate Development. His primary areas of expertise are integrations and separations, due diligence, technology strategy, digital transformation, project delivery and sales across multiple industries with significant experience in Banking, Capital Markets and Insurance.
#                       He has led teams globally across the Americas, Europe and Asia and has extensive experience in mergers and acquisitions, divestitures, carve outs, data analytics and business transformation.
#                       His client list as a Trusted Advisor includes global Financial Services institutions for leading Deals Transformation efforts in addition to top-tier Investment Banks / FinTechs where he has been involved in driving execution of their largest Regulatory Reform Programs on Wall Street.
# titleDescription
# industry	            Professional Services
# companyLocation	    London, England, United Kingdom
# location	            New York, New York, United States
# durationInRole	    8 years in role
# durationInCompany	    8 years in company
# pastExperienceCompanyName
# pastExperienceCompanyUrl
# pastExperienceCompanyTitle
# pastExperienceDate
# pastExperienceDuration
# connectionDegree	    2nd
# profileImageUrl	    https://media.licdn.com/dms/image/v2/C4D03AQFeYWENtiwcNA/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1517629277330?e=1767830400&v=beta&t=OAOIPKlaUW267IPdByk4UlF5XYbjinYYFpLcISR647w
# sharedConnectionsCount	2
# name	                Fawaz Ahmar
# vmid	                ACwAACIIGwQBarOTcv9cBkPkS4qb143voXG2rdA
# linkedInProfileUrl	https://www.linkedin.com/in/ACwAACIIGwQBarOTcv9cBkPkS4qb143voXG2rdA/
# isPremium	            TRUE
# isOpenLink	        FALSE
# query	https://www.linkedin.com/sales/search/people?query=(recentSearchParam%3A(id%3A5116736402%2CdoLogHistory%3Atrue)%2Cfilters%3AList((type%3ACOMPANY_HEADCOUNT%2Cvalues%3AList((id%3AI%2Ctext%3A10%252C000%252B%2CselectionType%3AINCLUDED)))%2C(type%3ACURRENT_TITLE%2Cvalues%3AList((text%3AMergers%2520and%2520Acquisitions%2CselectionType%3AINCLUDED)%2C(text%3AM%2526A%2CselectionType%3AINCLUDED)%2C(text%3ACorporate%2520Development%2CselectionType%3AINCLUDED)%2C(text%3ACorp%2520Dev%2CselectionType%3AINCLUDED)))%2C(type%3ASENIORITY_LEVEL%2Cvalues%3AList((id%3A310%2Ctext%3ACXO%2CselectionType%3AINCLUDED)%2C(id%3A300%2Ctext%3AVice%2520President%2CselectionType%3AINCLUDED)%2C(id%3A220%2Ctext%3ADirector%2CselectionType%3AINCLUDED)%2C(id%3A320%2Ctext%3AOwner%2520%252F%2520Partner%2CselectionType%3AINCLUDED)))%2C(type%3AREGION%2Cvalues%3AList((id%3A102221843%2Ctext%3ANorth%2520America%2CselectionType%3AINCLUDED)))%2C(type%3AYEARS_IN_CURRENT_POSITION%2Cvalues%3AList((id%3A2%2Ctext%3A1%2520to%25202%2520years%2CselectionType%3AINCLUDED)%2C(id%3A3%2Ctext%3A3%2520to%25205%2520years%2CselectionType%3AINCLUDED)%2C(id%3A4%2Ctext%3A6%2520to%252010%2520years%2CselectionType%3AINCLUDED)%2C(id%3A5%2Ctext%3AMore%2520than%252010%2520years%2CselectionType%3AINCLUDED)))))&sessionId=CZqz%2BZ75RriV0TlA4%2FHKew%3D%3D&viewAllFilters=true
# timestamp	            2025-12-23T22:22:40.983Z
# defaultProfileUrl	    https://linkedin.com/in/fawazahmar
# searchAccountProfileId	8207060
# searchAccountProfileName	Giacinto Paolo Saggese


def get_data_from_SalesNavigator_Search_Results(
    url: str,
    tab_name: str,
    tag: str,
    normalize: bool,
    *,
    allow_subset: bool = False,
    verbose: bool = False,
    remove_spaces_in_cols: bool = True,
    force_no_cache: bool = False,
) -> pd.DataFrame:
    """
    Load and optionally normalize LinkedIn Sales Navigator Search Results data.

    :param url: The URL of the LinkedIn Sales Navigator Search Results sheet.
    :param tab_name: The name of the Google sheet tab.
    :param tag: The tag of the data (e.g., "200AICompanies.MA").
    :param normalize: Whether to normalize the data.
    :param allow_subset: Whether to allow the subset of columns that are in the
        schema.
    :param verbose: Whether to print the data.
    :param remove_spaces_in_cols: Whether to remove spaces in the column names.
    :param force_no_cache: Whether to bypass the cache and fetch fresh data.
    :return: A dataframe with the data.
    """
    # Load the raw data from the Google Sheet.
    df = hgodrapi.get_gsheet_to_df(
        url,
        tab_name,
        remove_spaces_in_cols=remove_spaces_in_cols,
        force_no_cache=force_no_cache,
    )
    if verbose:
        display(df.head(1))
    # Normalize the data if requested.
    if normalize:
        # Set the origin and timestamp columns.
        df["origin"] = "PB.SalesNavigator_SearchResults." + tag
        df["origin_timestamp"] = df["timestamp"]
        df["email_timestamp"] = None
        # Define the column mapping from Sales Navigator schema to our
        # standard schema.
        cols_map = {
            "origin": None,
            "origin_timestamp": None,
            "email_timestamp": None,
            "profileUrl": "linkedin_url",
            "firstName": "first_name",
            "lastName": "last_name",
            "title": "job_title",
            "companyName": "company_name",
            "companyLocation": "city",
            "companyUrl": "company_domain",
            "summary": "biography",
        }
        df_out = cmwdlout.normalize_contact_schema(
            df, cols_map, allow_subset=allow_subset
        )
    else:
        df_out = df
    return df_out


# #############################################################################
# LinkedIn Profile Scraper
# #############################################################################


# profileUrl	        https://www.linkedin.com/in/fawazahmar/
# refreshedAt	        2025-12-24T00:29:39.324Z
# scraperProfileId	    8207060
# scraperFullName	    Fawaz Ahmar
# companyIndustry	    Information Technology & Services
# companyName	        PwC
# firstName	Fawaz
# lastName	Ahmar
# linkedinCompanyUrl	https://linkedin.com/company/pwc
# linkedinCompanySlug	pwc
# linkedinCompanyId	1044
# linkedinDescription	"Fawaz is an experienced Senior Director within the Financial Services and M&A Advisory consulting group at PwC with a focus on Deals and Technology Transformation and Corporate Development. His primary areas of expertise are integrations and separations, due diligence, technology strategy, digital transformation, project delivery and sales across multiple industries with significant experience in Banking, Capital Markets and Insurance.
#                       He has led teams globally across the Americas, Europe and Asia and has extensive experience in mergers and acquisitions, divestitures, carve outs, data analytics and business transformation.
#                       His client list as a Trusted Advisor includes global Financial Services institutions for leading Deals Transformation efforts in addition to top-tier Investment Banks / FinTechs where he has been involved in driving execution of their largest Regulatory Reform Programs on Wall Street."
# linkedinFollowersCount	2323
# linkedinHeadline	    Senior Director | FS / M&A Advisory | PwC
# linkedinIsHiringBadge	FALSE
# linkedinIsOpenToWorkBadge	FALSE
# linkedinJobDateRange	2018 - Present
# linkedinJobLocation	New York, New York, United States
# linkedinJobTitle	    Senior Director - FS / M&A Advisory
# linkedinPreviousCompanySlug	accenture
# linkedinPreviousJobDateRange	2009 - 2017
# linkedinPreviousJobLocation	Toronto, Canada Area
# linkedinPreviousJobTitle	Consulting Executive
# linkedinProfileId	570956548
# linkedinProfileSlug	fawazahmar
# linkedinProfileUrl	https://linkedin.com/in/fawazahmar
# linkedinProfileUrn	ACoAACIIGwQBFuO8HRLI_dLcOtTH3dkK0RqIKB0
# linkedinProfileImageUrn	urn:li:digitalmediaAsset:C4D03AQFeYWENtiwcNA
# linkedinProfileImageUrl	https://media.licdn.com/dms/image/v2/C4D03AQFeYWENtiwcNA/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1517629277330?e=1768435200&v=beta&t=xcHuPKUcokeLqCF0OjlFtcprEe7_zSCM5I9ESibM-5s
# linkedinSchoolUrl	    https://linkedin.com/school/university-of-toronto
# linkedinSchoolCompanySlug	university-of-toronto
# linkedinSchoolDegree	Bachelors
# linkedinSchoolName	University of Toronto
# linkedinSkillsLabel	Performance Improvement, Technology, FinTech, Product Management, Regulatory Projects, Software Development Life Cycle (SDLC), Digital Transformation, Program Management, Project Delivery, Data Analysis, Mergers & Acquisitions, Data Analytics, Post-Merger Integration, Customer Relationship Management (CRM), Data Conversion, Core Banking, Digital Health, Business Continuity Planning, Liquidity Risk, Recovery Resolution Planning
# location	New York, New York, United States
# previousCompanyName	Accenture
# connectionDegree	2nd
# mutualConnectionsUrl	https://www.linkedin.com/search/results/people/?facetNetwork=%5B%22F%22%5D&facetConnectionOf=%5B%22ACoAACIIGwQBFuO8HRLI_dLcOtTH3dkK0RqIKB0%22%5D&origin=MEMBER_PROFILE_CANNED_SEARCH&RESULT_TYPE=PEOPLE
# connectionsUrl	    https://www.linkedin.com/search/results/people/?facetConnectionOf=%5B%22ACoAACIIGwQBFuO8HRLI_dLcOtTH3dkK0RqIKB0%22%5D&facetNetwork=%5B%22F%22%2C%22S%22%5D&origin=MEMBER_PROFILE_CANNED_SEARCH
# linkedinConnectionsCount	2332
# linkedinJobDescription
# linkedinPreviousJobDescription
# linkedinPreviousSchoolUrl
# linkedinPreviousSchoolCompanySlug
# linkedinPreviousSchoolDateRange
# linkedinPreviousSchoolDegree
# linkedinPreviousSchoolDescription
# linkedinPreviousSchoolName
# linkedinSchoolDateRange
# linkedinSchoolDescription


def get_data_from_LinkedIn_Profile_Scraper(
    url: str,
    tab_name: str,
    tag: str,
    normalize: bool,
    *,
    allow_subset: bool = False,
    remove_spaces_in_cols: bool = True,
    verbose: bool = False,
    force_no_cache: bool = False,
) -> pd.DataFrame:
    """
    Load and optionally normalize LinkedIn Profile Scraper data.

    :param url: The URL of the LinkedIn Profile Scraper sheet.
    :param tab_name: The name of the Google sheet tab.
    :param tag: The tag of the data (e.g., "200AICompanies.MA").
    :param normalize: Whether to normalize the data.
    :param allow_subset: Whether to allow the subset of columns that are in the
        schema.
    :param remove_spaces_in_cols: Whether to remove spaces in the column names.
    :param verbose: Whether to print the data.
    :param force_no_cache: Whether to bypass the cache and fetch fresh data.
    :return: A dataframe with the data.
    """
    # Load the raw data from the Google Sheet.
    df = hgodrapi.get_gsheet_to_df(
        url,
        tab_name,
        remove_spaces_in_cols=remove_spaces_in_cols,
        force_no_cache=force_no_cache,
    )
    if verbose:
        display(df.head(1))
    # Normalize the data if requested.
    if normalize:
        # Set the origin column.
        df["origin"] = "PB.LIN_Profile_Scraper." + tag
        # Construct the biography column by concatenating available LinkedIn
        # fields.
        if all(
            col in df.columns
            for col in [
                "linkedinHeadline",
                "linkedinDescription",
                "linkedinJobDescription",
            ]
        ):
            df["biography"] = (
                df["linkedinHeadline"]
                + "; "
                + df["linkedinDescription"]
                + "; "
                + df["linkedinJobDescription"]
            )
        else:
            df["biography"] = None
        # Define the column mapping from LinkedIn Profile schema to our
        # standard schema.
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
            # Seed,Convertible Note,Series A,Pre-Seed.
            # "stages": "",
            # "restrictions": "",
            # Angel, VC, PE, Family Office, Corporate VC, Accelerator, Incubator.
            # "companyIndustry": "category",
            # "notes": "",
        }
        df_out = cmwdlout.normalize_contact_schema(
            df, cols_map, allow_subset=allow_subset
        )
    else:
        df_out = df
    if verbose:
        display(df_out.head(1))
    return df_out


# #############################################################################
# PhantomBuster All Leads.
# #############################################################################

# linkedinProfileUrl	https://linkedin.com/in/ngiliana
# salesNavigatorProfileUrl
# fullName	Nali Giliana
# firstName	Nali
# lastName	Giliana
# companyName	Alteryx
# linkedinCompanyUrl
# salesNavigatorCompanyUrl
# companyWebsite
# personalWebsite
# linkedinJobTitle	Chief of Strategy & staff
# linkedinJobDateRange	Dec 2024 - Present · 1 yr 1 mo
# linkedinJobLocation	United States · Remote
# companyIndustry
# linkedinHeadline	Strategy | CMO | COO | M&A | Biz Dev | GTM | Global Growth | Executive Advisor | Marketing & Technology Executive | Experience Platform | Former Adobe, HP, UserTesting
# location	Scottsdale, Arizona, United States
# professionalEmail1
# professionalEmail2
# personalEmail1
# phoneNumber1
# phoneNumber2
# civility
# twitterProfileUrl
# website1
# website2
# linkedinFollowersCount	2929
# linkedinOpenProfile	No
# linkedinDescription
# linkedinSkillsLabel
# previousCompanyName	UserTesting
# linkedinPreviousCompanyUrl	https://linkedin.com/company/564709
# linkedinPreviousCompanySlug	564709
# linkedinPreviousJobTitle	UserTesting
# linkedinPreviousJobDateRange	Feb 2024 - Dec 2024 · 11 mos
# linkedinPreviousJobLocation	Phoenix, Arizona, United States · Remote
# linkedinJobDescription
# linkedinPreviousJobDescription
# linkedinSchoolName	Arizona State University
# linkedinSchoolCompanyUrl	https://www.linkedin.com/school/4292
# linkedinSchoolCompanySlug	4292
# linkedinSchoolDegree	Bachelor of Science, Management Information Systems
# linkedinSchoolDateRange	2003 - 2006
# linkedinSchoolDescription
# linkedinPreviousSchoolName
# linkedinPreviousSchoolCompanyUrl
# linkedinPreviousSchoolCompanySlug
# linkedinPreviousSchoolDegree
# linkedinPreviousSchoolDateRange
# linkedinPreviousSchoolDescription
# linkedinMutualConnectionsUrl	https://www.linkedin.com/search/results/people/?facetNetwork=%5B%22F%22%5D&facetConnectionOf=%5B%22ACoAAAM-aEUBmGUDbWUtgtN0KadTwTlpYWMTKqw%22%5D&origin=MEMBER_PROFILE_CANNED_SEARCH&RESULT_TYPE=PEOPLE
# linkedinProfileSlug	ngiliana
# salesNavigatorProfileSlug
# linkedinCompanySlug
# linkedinProfileId	54421573
# linkedinProfileUrn	ACoAAAM-aEUBmGUDbWUtgtN0KadTwTlpYWMTKqw
# linkedinIsHiringBadge	No
# linkedinIsOpenToWorkBadge	No
# id	5607736512734285
# connectionDegree	2nd
# createdAt	12/25/2025
# updatedAt	12/25/2025
# createdBy	LinkedIn Outreach - MA
# updatedBy	LinkedIn Outreach - MA


def get_data_from_PhantomBuster_All_Leads(
    url: str,
    tab_name: str,
    tag: str,
    normalize: bool,
    *,
    allow_subset: bool = False,
    remove_spaces_in_cols: bool = True,
    verbose: bool = False,
    force_no_cache: bool = False,
) -> pd.DataFrame:
    """
    Load and optionally normalize PhantomBuster All Leads data.

    :param url: The URL of the Google sheet containing the data.
    :param tab_name: The name of the Google sheet tab.
    :param tag: The tag of the data (e.g., "200AICompanies.MA").
    :param normalize: Whether to normalize the data.
    :param allow_subset: Whether to allow the subset of columns that are in the
        schema.
    :param remove_spaces_in_cols: Whether to remove spaces in the column names.
    :param verbose: Whether to print the data.
    :param force_no_cache: Whether to bypass the cache and fetch fresh data.
    :return: A dataframe with the data.
    """
    # Load the raw data from the Google Sheet.
    df = hgodrapi.get_gsheet_to_df(
        url,
        tab_name,
        remove_spaces_in_cols=remove_spaces_in_cols,
        force_no_cache=force_no_cache,
    )
    if verbose:
        display(df.head(1))
    # Normalize the data if requested.
    if normalize:
        # Set the origin.
        df["origin"] = "PB.PhantomBuster_All_Leads." + tag
        # Construct the biography column by concatenating available LinkedIn
        # fields.
        if all(
            col in df.columns
            for col in [
                "linkedinHeadline",
                "linkedinDescription",
                "linkedinJobDescription",
            ]
        ):
            df["biography"] = (
                df["linkedinHeadline"]
                + "; "
                + df["linkedinDescription"]
                + "; "
                + df["linkedinJobDescription"]
            )
        else:
            df["biography"] = None
        # Define the column mapping from LinkedIn Profile schema to our
        # standard schema.
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
            # Seed,Convertible Note,Series A,Pre-Seed.
            # "stages": "",
            # "restrictions": "",
            # Angel, VC, PE, Family Office, Corporate VC, Accelerator, Incubator.
            # "companyIndustry": "category",
            # "notes": "",
        }
        df_out = cmwdlout.normalize_contact_schema(
            df, cols_map, allow_subset=allow_subset
        )
    else:
        df_out = df
    if verbose:
        display(df_out.head(1))
    return df_out


# #############################################################################


def _check_data_type(
    cols: List[str],
    df: pd.DataFrame,
    *,
    log_level: int = logging.DEBUG,
) -> bool:
    """
    Check if the specified columns are present in the dataframe.

    :param cols: The list of column names to check for.
    :param df: The dataframe to check against.
    :param log_level: The log level for debug output.
    :return: True if all columns are present, False otherwise.
    """
    _LOG.log(log_level, "\ncols=%s\ncolumns=%s", cols, df.columns.tolist())
    # Check if the columns are a subset of the dataframe columns.
    is_subset = set(cols).issubset(df.columns)
    _LOG.log(log_level, "is_subset=%s", is_subset)
    if not is_subset:
        # Try with spaces removed from column names.
        cols = [col.replace(" ", "") for col in cols]
        is_subset = set(cols).issubset(df.columns)
        _LOG.log(log_level, "is_subset=%s", is_subset)
    return is_subset


def analyze_data_type(
    url: str,
    tab_name: str,
    *,
    force_no_cache: bool = False,
    log_level: int = logging.DEBUG,
) -> None:
    """
    Analyze and identify the data type of a Google Sheet based on its columns.

    This function checks the sheet against known data source patterns and
    attempts to load and normalize the data using the appropriate loader
    function if a match is found.

    :param url: The URL of the GSheet.
    :param tab_name: The name of the sheet in the GSheet, or "_all_" to analyze
        all tabs.
    :param force_no_cache: Whether to bypass the cache and fetch fresh data.
    :param log_level: The log level for debug output.
    """
    # If the tab_name is "_all_", recursively analyze all tabs (worksheets) in
    # the gsheet.
    if tab_name == "_all_":
        credentials = hgodrapi.get_credentials()
        tabs = hgodrapi.get_tabs_from_gsheet(url, credentials=credentials)
        _LOG.log(log_level, "tabs=%s", tabs)
        for tab in tabs:
            analyze_data_type(
                url, tab, force_no_cache=force_no_cache, log_level=log_level
            )
        return
    # Set up default parameters for data loading and normalization.
    tag = "test"
    timestamp = pd.Timestamp.now()
    allow_subset = True
    normalize = True
    verbose = False
    # Load the sheet into a dataframe.
    df = hgodrapi.get_gsheet_to_df(
        url,
        tab_name,
        force_no_cache=force_no_cache,
    )
    results = {}
    # Check the dataframe against each data type pattern and try to
    # load/normalize using the corresponding function if it matches.
    # Check pattern 1: Pitchbook data.
    var_name = "data_from_pitchbook"
    _LOG.log(log_level, "# Checking pattern '%s'", var_name)
    results[var_name] = False
    cols = [
        "Primary Position",
        "Primary Company",
    ]
    if _check_data_type(cols, df):
        # If the required Pitchbook columns are present, extract and normalize.
        _ = get_data_from_pitchbook(
            url,
            tab_name,
            timestamp,
            tag,
            normalize,
            allow_subset=allow_subset,
            verbose=verbose,
            force_no_cache=force_no_cache,
        )
        results[var_name] = True
    _LOG.log(log_level, "%s=%s", var_name, results[var_name])
    # Check pattern 2: LinkedIn Connections Exports data.
    var_name = "data_from_LinkedIn_Connections_Exports"
    _LOG.log(log_level, "# Checking pattern '%s'", var_name)
    results[var_name] = False
    cols = ["profileUrl", "title", "connectionSince"]
    if _check_data_type(cols, df, log_level=log_level):
        # If the required columns are present, extract and normalize.
        _ = get_data_from_LinkedIn_Connections_Exports(
            url,
            tab_name,
            tag,
            normalize,
            allow_subset=allow_subset,
            verbose=verbose,
            force_no_cache=force_no_cache,
        )
        results[var_name] = True
    _LOG.log(log_level, "%s=%s", var_name, results[var_name])
    # Check pattern 3: Sales Navigator Search Results data.
    var_name = "data_from_SalesNavigator_Search_Results"
    _LOG.log(log_level, "# Checking pattern '%s'", var_name)
    results[var_name] = False
    cols = ["profileUrl", "title", "companyId"]
    if _check_data_type(cols, df, log_level=log_level):
        # If the required columns are present, extract and normalize.
        _ = get_data_from_SalesNavigator_Search_Results(
            url,
            tab_name,
            tag,
            normalize,
            allow_subset=allow_subset,
            verbose=verbose,
            force_no_cache=force_no_cache,
        )
        results[var_name] = True
    _LOG.log(log_level, "%s=%s", var_name, results[var_name])
    # Check pattern 4: LinkedIn Profile Scraper data.
    var_name = "data_from_LinkedIn_Profile_Scraper"
    _LOG.log(log_level, "# Checking pattern '%s'", var_name)
    results[var_name] = False
    cols = ["refreshedAt", "linkedinCompanyUrl", "connectionDegree"]
    if _check_data_type(cols, df, log_level=log_level):
        # If the required columns are present, extract and normalize.
        _ = get_data_from_LinkedIn_Profile_Scraper(
            url,
            tab_name,
            tag,
            normalize,
            allow_subset=allow_subset,
            verbose=verbose,
            force_no_cache=force_no_cache,
        )
        results[var_name] = True
    _LOG.log(log_level, "%s=%s", var_name, results[var_name])
    # Check for other column patterns.
    df_cols = [col.lower() for col in df.columns.tolist()]
    # Check for LinkedIn profile columns.
    cols = [
        "LinkedInURL",
        "LinkedIn URL",
        "defaultProfileUrl",
        "linkedinProfileUrl",
    ]
    if any(col.lower() in df_cols for col in cols):
        var_name = "has_linkedin_profile"
        results[var_name] = True
    # Check for email columns.
    cols = ["Email"]
    if any(col.lower() in df_cols for col in cols):
        var_name = "has_email"
        results[var_name] = True
    # Check for LinkedIn text message columns.
    cols = ["LinText", "LIn_msg"]
    if any(col.lower() in df_cols for col in cols):
        var_name = "has_lin_text"
        results[var_name] = True
    # Check for email text columns.
    cols = ["EmailTxt", "EmailText", "Email Text"]
    if any(col.lower() in df_cols for col in cols):
        var_name = "has_email_txt"
        results[var_name] = True
    # Check for merge status columns.
    cols = ["Merge status", "MergeStatus"]
    if any(col.lower() in df_cols for col in cols):
        var_name = "has_merge_status"
        results[var_name] = True
    # Log the summary showing which types matched and were processed.
    results_as_str = ", ".join([k for k, v in results.items() if v])
    _LOG.info("%s=(%s)", tab_name, results_as_str)


# #############################################################################


def get_data_from_fuzzy_column_matching(
    url: str,
    tab_name: Union[str, List[str]],
    normalize: bool,
    *,
    tag: str = "",
) -> pd.DataFrame:
    """
    Load data from a Google Sheet using fuzzy column matching.

    :param url: The URL of the Google sheet containing the data.
    :param tab_name: The name of the Google sheet tab, "_all_" for all tabs,
        or a list of tab names.
    :param normalize: Whether to normalize the data using fuzzy column matching.
    :param tag: The tag of the data (e.g., "200AICompanies.MA").
    :return: A dataframe with the data.
    """
    credentials = hgodrapi.get_credentials()
    if tab_name == "_all_":
        tab_names = hgodrapi.get_tabs_from_gsheet(url, credentials=credentials)
    elif isinstance(tab_name, list):
        tab_names = tab_name
    else:
        tab_names = None
    # Process multiple tabs recursively.
    if tab_names is not None:
        dfs = []
        for tab_name in tab_names:
            df_tmp = get_data_from_fuzzy_column_matching(
                url, tab_name, normalize, tag=tag
            )
            dfs.append(df_tmp)
        df = pd.concat(dfs)
        return df
    # Load the raw data from the Google Sheet.
    remove_spaces_in_cols = False
    force_no_cache = False
    df = hgodrapi.get_gsheet_to_df(
        url,
        tab_name,
        remove_spaces_in_cols=remove_spaces_in_cols,
        force_no_cache=force_no_cache,
    )
    if normalize:
        out_map, unmatched_keys, unmatched_cols = cmwdlout.fuzzy_column_matching(
            df.columns, print_results=False
        )
        _LOG.warning("unmatched_keys=%s", unmatched_keys)
        _LOG.warning("unmatched_cols=%s", unmatched_cols)
        # The mapping is from the original columns to the new columns, so we
        # need to invert it.
        out_map_tmp = {v: k for k, v in out_map.items()}
        hpandas.dassert_valid_remap(df.columns.tolist(), out_map_tmp)
        df = df.rename(columns=out_map_tmp)
        # Keep only the columns that are in the mapping.
        df = df[out_map_tmp.values()]
        # Set the origin column.
        if tag == "":
            file_name = hgodrapi.get_gsheet_name(url, credentials=credentials)
            origin = f"{file_name}.{tab_name}"
        else:
            origin = tag
        df.insert(0, "origin", origin)
    return df
