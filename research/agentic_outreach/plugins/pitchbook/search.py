#!/usr/bin/env python

"""
Automate PitchBook people search scraping and save results as MHTML files.

This script opens a Chrome browser for manual SSO login to PitchBook,
then navigates through search result pages and saves each page as an
MHTML file for later processing with parse_mhtml.py.

Import as:

import ck_marketing.plugins.pitchbook.search as ckplpbse

Examples:
# Scrape all pages to default output directory.
> python search.py --output_dir ./output

# Scrape with custom max pages limit.
> python search.py --output_dir ./output --max_pages 10

# Scrape with verbose logging.
> python search.py --output_dir ./output -v DEBUG
"""

import argparse
import logging
import os
import time
from typing import Optional

from playwright.sync_api import Page, sync_playwright

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)

# PitchBook URLs.
_PITCHBOOK_LOGIN_URL = "https://my.pitchbook.com"
_PITCHBOOK_SEARCH_URL = (
    "https://my.pitchbook.com/as-criteria/PERSON/PEOPLE/search"
)

# #############################################################################
# Login
# #############################################################################


def _wait_for_manual_login(page: Page) -> None:
    """
    Open PitchBook and wait for user to manually complete SSO login.

    Opens the search page in Chrome browser and waits for the user to
    complete single sign-on authentication manually. The function waits
    until the search page is fully loaded or user navigates to it.

    :param page: Playwright page object
    """
    _LOG.info("Opening PitchBook search page in browser")
    _LOG.info("Please complete SSO login manually in the browser window")
    page.goto(_PITCHBOOK_SEARCH_URL)
    # Wait for user to complete login and page to load.
    # The user will be redirected to login if not authenticated.
    # Once logged in and back on search page this will complete.
    _LOG.info("Waiting for login to complete...")
    _LOG.info(
        "Press Enter in this terminal once you have logged in and see the search results page"
    )
    # Wait for user confirmation.
    input("Press Enter to continue after logging in...")
    # Ensure page is loaded.
    page.wait_for_load_state("networkidle", timeout=30000)
    _LOG.info("Login confirmed, proceeding with scraping")


# #############################################################################
# Page scraping
# #############################################################################


def _save_page_as_mhtml(page: Page, *, output_path: str) -> None:
    """
    Save current page as MHTML file using CDP.

    :param page: Playwright page object
    :param output_path: path where MHTML file should be saved
    """
    _LOG.info("Saving page as MHTML: path='%s'", output_path)
    # Use Chrome DevTools Protocol to capture snapshot.
    client = page.context.new_cdp_session(page)
    snapshot = client.send("Page.captureSnapshot", {"format": "mhtml"})
    mhtml_content = snapshot["data"]
    # Write MHTML content to file.
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mhtml_content)
    _LOG.debug("MHTML file saved successfully")


def _get_next_button(page: Page) -> Optional[object]:
    """
    Find and return the Next button if it exists and is enabled.

    :param page: Playwright page object
    :return: Next button element or None if not found or disabled
    """
    # Try common selectors for Next button.
    selectors = [
        'button:has-text("Next")',
        'a:has-text("Next")',
        '[aria-label="Next"]',
        ".pagination button:last-child",
        ".pagination a:last-child",
    ]
    for selector in selectors:
        try:
            button = page.locator(selector).first
            if button.is_visible() and button.is_enabled():
                _LOG.debug("Found Next button with selector: %s", selector)
                return button
        except Exception:
            continue
    _LOG.debug("Next button not found or not enabled")
    return None


def _scrape_search_results(
    page: Page,
    *,
    output_dir: str,
    max_pages: Optional[int] = None,
) -> None:
    """
    Navigate through search result pages and save each as MHTML.

    Assumes page is already on the search results page after login.

    :param page: Playwright page object
    :param output_dir: directory to save MHTML files
    :param max_pages: maximum number of pages to scrape (None = all)
    """
    _LOG.info("Starting search results scraping")
    # Create output directory.
    hio.create_dir(output_dir, incremental=False)
    page_num = 1
    while True:
        # Check if we've reached max pages.
        if max_pages is not None and page_num > max_pages:
            _LOG.info("Reached max pages limit: %s", max_pages)
            break
        # Save current page.
        output_file = os.path.join(output_dir, f"page_{page_num:03d}.mhtml")
        _LOG.info("Processing page %s", page_num)
        _save_page_as_mhtml(page, output_path=output_file)
        # Look for Next button.
        next_button = _get_next_button(page)
        if next_button is None:
            _LOG.info("No more pages to scrape")
            break
        # Click Next and wait for new page to load.
        _LOG.info("Clicking Next button")
        next_button.click()
        page.wait_for_load_state("networkidle", timeout=30000)
        # Small delay to ensure page is fully rendered.
        time.sleep(2)
        page_num += 1
    _LOG.info("Scraping completed: total_pages=%s", page_num)


# #############################################################################
# Main
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        required=True,
        help="Directory to save MHTML files",
    )
    parser.add_argument(
        "--max_pages",
        action="store",
        type=int,
        default=None,
        help="Maximum number of pages to scrape (default: all pages)",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    _LOG.info("Starting PitchBook scraper")
    _LOG.debug("Output directory: %s", args.output_dir)
    _LOG.debug("Max pages: %s", args.max_pages or "unlimited")
    # Launch browser and start scraping.
    with sync_playwright() as p:
        _LOG.info("Launching Chrome browser for manual SSO login")
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        try:
            # Wait for user to complete SSO login manually.
            _wait_for_manual_login(page)
            # Scrape search results.
            _scrape_search_results(
                page,
                output_dir=args.output_dir,
                max_pages=args.max_pages,
            )
        finally:
            _LOG.info("Closing browser")
            browser.close()
    _LOG.info("Script completed successfully")


if __name__ == "__main__":
    _main(_parse())
