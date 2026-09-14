"""
Import as:

import ai_outreach.plugins.dropcontact.drop_contact_api as aopddcoap
"""

# https://developer.dropcontact.com/#introduction

import logging
import math
import time
from typing import Any, Dict, List

import pandas as pd
import requests  # type: ignore[import-untyped]
import tqdm

import helpers.hdbg as hdbg

_LOG = logging.getLogger(__name__)


def _preprocess_dropcontact_data(
    first_names: List[str], last_names: List[str], company_names: List[str]
) -> List[Dict[str, str]]:
    """
    Preprocess data for DropContact API.

    :param first_names: first names.
    :param last_names: last names.
    :param company_names: company names.
    :return: dictionaries with first name, last name and company name.
    """
    data: List[Dict[str, str]] = []
    hdbg.dassert_eq(len(first_names), len(last_names))
    hdbg.dassert_eq(len(first_names), len(company_names))
    for first_name, last_name, company in zip(
        first_names, last_names, company_names
    ):
        data.append(
            {
                "first_name": first_name,
                "last_name": last_name,
                "company": company,
            }
        )
    return data


def _request_dropcontact(batch_data: List[Dict[str, str]], api_key: str) -> Any:
    """
    Send request to DropContact API.

    :param batch_data: first name, last name and company name.
    :param api_key: api key of DropContact.
    :return: query result data.
    """
    post_response = requests.post(
        "https://api.dropcontact.io/batch",
        json={
            "data": batch_data,
            "siren": True,
            "language": "en",
        },
        headers={
            "Content-Type": "application/json",
            "X-Access-Token": api_key,
        },
        timeout=30,
    ).json()
    return post_response


def _generate_result_df(query_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Generate dataframe from query result.
    """
    result_list = []
    for result in query_results:
        first_name = result.get("first_name", "")
        last_name = result.get("last_name", "")
        full_name = result.get("full_name", "")
        email = ""
        if "email" in result:
            email = ";".join(map(lambda x: x["email"], result["email"]))
        phone = ""
        if "phone" in result:
            phone += result["phone"]
        phone = str(phone)
        if "mobile_phone" in result:
            if phone:
                phone += ";"
            phone += result["mobile_phone"]
        pronoun = result.get("civility", "")
        job = result.get("job", "")
        result_list.append(
            [first_name, last_name, full_name, email, phone, pronoun, job]
        )
    result_title = [
        "first name",
        "last name",
        "full name",
        "email",
        "phone",
        "pronoun",
        "job title",
    ]
    df = pd.DataFrame(data=result_list, columns=result_title)
    return df


def _send_batch_request(
    data: List[Dict[str, str]], api_key: str, batch_size: int
) -> List[Dict[str, Any]]:
    """
    Send batch request to DropContact API.

    :param data: first name, last name and company name.
    :param api_key: api key of DropContact.
    :param batch_size: batch size.
    :return: query result.
    """
    batches = []
    data_length = len(data)
    for batch_i in range(math.ceil(data_length / batch_size)):
        batches.append(data[batch_i * batch_size : (batch_i + 1) * batch_size])
    query_results = []
    for batch_idx, batch_data in enumerate(
        tqdm.tqdm(batches, desc="Processing batches", ascii=True, ncols=100)
    ):
        _LOG.debug("Starting query batch %s", batch_idx)
        batch_result = []
        post_response = _request_dropcontact(batch_data, api_key)
        query_id = post_response["request_id"]
        print(f"Batch {str(batch_idx)}: Query ID: {str(query_id)}.")
        # Wait for query result, 10 seconds per attempt, 120 seconds timeout.
        for _ in range(12):
            get_response = requests.get(
                f"https://api.dropcontact.io/batch/{query_id}",
                headers={"X-Access-Token": api_key},
                timeout=30,
            ).json()
            query_finished = get_response["success"]
            if query_finished:
                batch_result = get_response["data"]
                credits_left = get_response["credits_left"]
                print(
                    f"Batch {str(batch_idx)}: Query finished. Credits left: {str(credits_left)}."
                )
                break
            reason = get_response["reason"]
            error = get_response["error"]
            if error:
                print(f"Error detected, reason: {str(reason)}.")
                break
            time.sleep(10)
        if not batch_result:
            print(f"Batch {str(batch_idx)}: Query failed, reason: timeout.")
            batch_result = [{}] * len(batch_data)
        query_results += batch_result
    return query_results


def get_email_from_dropcontact(
    first_names: List[str],
    last_names: List[str],
    company_names: List[str],
    api_key: str,
    *,
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Get email from DropContact API using first name, last name and company
    name.

    :param first_names: first names.
    :param last_names: last names.
    :param company_names: company names.
    :param api_key: api key of DropContact.
    :return: DataFrame with first name, last name, full name, email,
        phone, pronoun, job title.
    """
    data = _preprocess_dropcontact_data(first_names, last_names, company_names)
    query_results = _send_batch_request(data, api_key, batch_size)
    result_df = _generate_result_df(query_results)
    return result_df
