"""
Import as:

import ai_outreach.plugins.linkedin.phantombuster_api as aoplphap
"""

import io
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
import requests  # type: ignore[import-untyped]

import helpers.hdbg as hdbg
import helpers.hio as hio

_LOG = logging.getLogger(__name__)


# #############################################################################
# Phantom
# #############################################################################


class Phantom:

    def __init__(self) -> None:
        """
        Initialize the Phantom class with the API key from environment.
        """
        self.api_key = os.getenv("Phantom_API_KEY")
        self.headers = {
            "X-Phantombuster-Key-1": self.api_key,
            "Content-Type": "application/json",
        }

    def create_sales_nav_phantom(
        self,
        agent_name: str,
        sales_nav_query: str,
        linkedin_session_cookie: str,
    ) -> Dict:
        """
        Create a Sales Navigator Phantom with the provided search query and
        LinkedIn session cookie.

        :param sales_nav_query: sales Navigator search query URL.
        :param linkedin_session_cookie: linkedIn session cookie.
        :return: response from the PhantomBuster API.
        """
        url = "https://api.phantombuster.com/api/v2/agents/save"
        payload = {
            "argument": {
                "numberOfProfiles": 2500,
                "numberOfResultsPerSearch": 2500,
                "numberOfLinesPerLaunch": 10,
                "removeDuplicateProfiles": True,
                "searches": sales_nav_query,
                "sessionCookie": linkedin_session_cookie,
            },
            "org": "phantombuster",
            "script": "Sales Navigator Search Export.js",
            "branch": "master",
            "environment": "release",
            "name": agent_name,
            "fileMgmt": "delete",
            "launchType": "manually",
            "nbLaunches": 2,
            "lastEndType": "finished",
        }
        response = requests.post(
            url, json=payload, headers=self.headers, timeout=30
        )
        return cast(Dict[str, Any], response.json())

    def create_linkedIn_info_extractor_phantom(
        self, gsheet_url: str, agent_name: str, linkedin_session_cookie: str
    ) -> Dict[str, Any]:
        """
        Create and configure a LinkedIn Search Export Phantom to extract
        profile info from linkedin urls.

        :param gsheet_url: URL of the Google Sheet containing LinkedIn
            URLs
        :param agent_name: Name of the Phantom to create
        :param linkedin_session_cookie: LinkedIn session cookie
        :return: Details of the created Phantom
        """
        url = "https://api.phantombuster.com/api/v2/agents/save"
        payload = {
            "argument": {
                "spreadsheetUrl": gsheet_url,
                "sessionCookie": linkedin_session_cookie,
                "numberOfProfilesPerSearch": 10000,
            },
            "org": "phantombuster",
            "name": agent_name,
            "script": "LinkedIn Profile Scraper.js",
            "branch": "master",
            "environment": "release",
            "fileMgmt": "delete",
            "launchType": "manually",
            "nbLaunches": 2,
            "lastEndType": "finished",
        }
        response = requests.post(
            url, json=payload, headers=self.headers, timeout=30
        )
        if response.status_code == 200:
            linkedin_resp = cast(Dict[str, Any], response.json())
        else:
            raise RuntimeError(
                f"Error creating Phantom: {response.status_code} {response.text}"
            )
        return linkedin_resp

    def create_linkedIn_url_finder_phantom(
        self, agent_name: str, gsheet_url: str, linkedin_session_cookie: str
    ) -> Dict[str, Any]:
        """
        Create and configure a Phantom to find LinkedIn profile URLs based on a
        Google Sheet.

        :param agent_name: name of the Phantom to create
        :param gsheet_url: url of the Google Sheet containing names
        :param linkedin_session_cookie: linkedIn session cookie (li_at)
        :return: details of the created Phantom
        """
        url = "https://api.phantombuster.com/api/v2/agents/save"
        payload = {
            "argument": {
                "spreadsheetUrl": gsheet_url,
                "nameColumn": "fullName",
                "sessionCookie": linkedin_session_cookie,
            },
            "org": "phantombuster",
            "name": agent_name,
            "script": "LinkedIn URL Finder.js",
            "branch": "master",
            "environment": "release",
            "fileMgmt": "delete",
            "launchType": "manually",
            "nbLaunches": 2,
            "lastEndType": "finished",
        }
        response = requests.post(
            url, json=payload, headers=self.headers, timeout=30
        )
        if response.status_code == 200:
            return cast(Dict[str, Any], response.json())
        raise ValueError(
            f"Error creating Phantom: {response.status_code} {response.text}"
        )

    def delete_phantom(self, agent_id: str) -> None:
        """
        Delete a Phantom by its agent ID.
        """
        url = "https://api.phantombuster.com/api/v2/agents/delete"
        payload = {"id": agent_id}
        response = requests.post(
            url, json=payload, headers=self.headers, timeout=30
        )
        hdbg.dassert_eq(
            response.status_code,
            200,
            f"Deletion failed for Phantom ID {agent_id}. Response: {response.text}",
        )
        _LOG.info(
            "Successfully deleted phantom %s. Response: %s",
            agent_id,
            response.text,
        )

    def get_all_agents(self) -> List:
        """
        Fetch all agents from the PhantomBuster API.
        """
        url = "https://api.phantombuster.com/api/v2/agents/fetch-all"
        response = requests.get(url, headers=self.headers, timeout=30)
        response_data = response.json()
        if isinstance(response_data, list):
            return cast(List[Any], response_data)
        if "status" in response_data and response_data["status"] == "success":
            return cast(List[Any], response_data["data"])
        raise RuntimeError(f"Failed to get agents: {response_data}")

    def get_agent_name(self, agent_id: str) -> str:
        """
        Retrieve the name of a specific agent given its ID.
        """
        agents = self.get_all_agents()
        for agent in agents:
            if agent["id"] == agent_id:
                return cast(str, agent["name"])
        raise ValueError(f"Agent with ID {agent_id} not found")

    def launch_agent(self, agent_id: str) -> Dict:
        """
        Launch a specific agent by its ID.
        """
        url = "https://api.phantombuster.com/api/v2/agents/launch"
        payload = {"id": agent_id}
        response = requests.post(
            url, headers=self.headers, json=payload, timeout=30
        )
        response_json = cast(Dict[Any, Any], response.json())
        hdbg.dassert_in(
            "containerId",
            response_json,
            msg=f"Failed to launch agent: {response_json}",
        )
        return response_json

    def launch_and_get_df(self, agent_id: str) -> pd.DataFrame:
        ress = self.launch_agent(agent_id)
        _LOG.debug(ress)
        time.sleep(10)
        result_response_json = self.fetch_agent_results(agent_id)
        time.sleep(10)
        csv_url = self.get_csv_url(result_response_json.get("output", ""))
        df = self.download_csv(csv_url)
        df = df.replace([np.nan, np.inf, -np.inf], "", inplace=False)
        return df

    def fetch_agent_results(self, agent_id: str) -> Dict:
        """
        Fetch the results of a specific agent after it has been launched.
        """
        url = "https://api.phantombuster.com/api/v2/agents/fetch-output"
        while True:
            response = requests.get(
                url, headers=self.headers, params={"id": agent_id}, timeout=30
            )
            response_json = cast(Dict[Any, Any], response.json())
            if response_json.get("status") == "error":
                raise RuntimeError("Error fetching the agent's results")
            if not response_json.get("isAgentRunning", True):
                print("Agent finished running")
                return response_json
            print("Agent still running, waiting for 30 seconds...")
            time.sleep(30)

    def get_csv_url(self, output_text: str) -> str:
        """
        Extract the CSV URL from the agent's output text.
        """
        csv_url_match = re.search(r"https://[^\s]+/result\.csv", output_text)
        hdbg.dassert(
            csv_url_match is not None,
            msg="No CSV URL found in the output",
        )
        assert csv_url_match is not None
        return csv_url_match.group(0)

    def download_csv(self, csv_url: str) -> pd.DataFrame:
        """
        Download a CSV file from a given URL and load it into a DataFrame.
        """
        response = requests.get(csv_url, timeout=30)
        csv_content = response.content.decode("utf-8")
        return pd.read_csv(io.StringIO(csv_content))

    def get_all_phantoms(self) -> Optional[pd.DataFrame]:
        """
        Retrieve all the names and IDs of the Phantoms from Phantombuster.
        """
        data = self._get_phantom_data()
        if data and "data" in data and "agents" in data["data"]:
            return pd.DataFrame(data["data"]["agents"])
        _LOG.error("The data structure is invalid: Phantoms not found.")
        return None

    def download_result_csv_by_phantom_id(
        self, phantom_id: str, output_path: str
    ) -> None:
        result_url = self._get_result_csv_by_phantom_id(phantom_id)
        if result_url:
            self._download_result_csv_to_local(result_url, output_path)

    # #########################################################################

    @staticmethod
    def _download_result_csv_to_local(
        result_csv_url: str, output_path: str
    ) -> None:
        hio.create_enclosing_dir(output_path, incremental=True)
        try:
            response = requests.get(result_csv_url, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            _LOG.error("%s", e)
            return None
        response.encoding = "UTF-8"
        hio.to_file(output_path, response.text)
        _LOG.info("Result CSV saved to %s", output_path)
        return None

    @staticmethod
    def _extract_result_csv_url(response_text: str) -> str:
        data = json.loads(response_text)
        output = data["output"]
        csv_url_match = re.search("CSV saved at (https://[^\\s]+)", output)
        if csv_url_match:
            return csv_url_match.group(1)
        raise ValueError(
            f"Unable to find the result CSV URL in the response: {output}"
        )

    def _get_api_response(self, url: str) -> Optional[requests.Response]:
        headers = {
            "accept": "application/json",
            "X-Phantombuster-Key": self.api_key,
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response

    def _get_phantom_data(self) -> Optional[Dict[str, Any]]:
        url = "https://api.phantombuster.com/api/v1/user"
        response = self._get_api_response(url)
        if response is None:
            return None
        return cast(Optional[Dict[str, Any]], response.json())

    def _get_all_containers_id_by_phantom_id(
        self, phantom_id: str
    ) -> Optional[List[str]]:
        url = f"https://api.phantombuster.com/api/v2/containers/fetch-all?agentId={phantom_id}"
        response = self._get_api_response(url)
        if response is None:
            return None
        data = response.json().get("containers", [])
        containers_id = [container["id"] for container in data]
        return containers_id

    def _get_result_csv_by_container_id(self, container_id: str) -> Optional[str]:
        url = f"https://api.phantombuster.com/api/v2/containers/fetch-output?id={container_id}"
        response = self._get_api_response(url)
        if response is None:
            return None
        result_csv_url = self._extract_result_csv_url(response.text)
        if result_csv_url:
            _LOG.info("Result CSV URL: %s", result_csv_url)
            return result_csv_url
        return None

    # #########################################################################

    def _get_result_csv_by_phantom_id(self, phantom_id: str) -> str:
        containers_id = self._get_all_containers_id_by_phantom_id(phantom_id)
        hdbg.dassert_is(
            containers_id,
            None,
            "There is no container id available. Have you run the Phantom?",
            only_warning=False,
        )
        container_to_process = containers_id[-1]
        result_csv_url = self._get_result_csv_by_container_id(
            container_to_process
        )
        hdbg.dassert(result_csv_url is not None, "Failed to fetch result CSV.")
        return result_csv_url
