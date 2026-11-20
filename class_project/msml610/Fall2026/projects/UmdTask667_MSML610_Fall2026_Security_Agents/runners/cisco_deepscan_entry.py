#!/usr/bin/env python
"""Run Cisco unchanged with the repo's cached, validated API transport."""

import os
import sys

sys.path.insert(0, os.environ["CISCO_TOOL_DIR"])

import deepscan

from cisco_client_lib import CachedClient


if __name__ == "__main__":
    deepscan.LLMClient = CachedClient
    deepscan.main()
