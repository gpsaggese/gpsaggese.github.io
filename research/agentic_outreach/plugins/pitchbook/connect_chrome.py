#!/usr/bin/env python
"""
Automate pagination through PitchBook results and save each page as MHTML.

This script connects to a Chrome browser via CDP (Chrome DevTools Protocol),
navigates to page 1, then iterates through all pages clicking Next and saving
each page as MHTML format.

Example usage:
> python connect_chrome.py --output_dir ./output
> python connect_chrome.py --output_dir /tmp/pitchbook_results -v DEBUG

Import as:

import ck_marketing.plugins.pitchbook.connect_chrome as cmplpbcoch
"""

import argparse
import logging
import random
import re
import time
from pathlib import Path

from playwright.sync_api import TimeoutError, sync_playwright
from tqdm import tqdm

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)

CDP_URL = "http://127.0.0.1:9222"

# #############################################################################


def _parse() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        type=str,
        required=True,
        help="Directory where MHTML files will be saved",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _save_page_as_mhtml(
    *,
    page,
    context,
    output_path: Path,
) -> None:
    """
    Save current page as MHTML file.

    :param page: Playwright page object.
    :param context: Playwright browser context.
    :param output_path: Path where MHTML file should be saved.
    """
    # Make sure the page is fully loaded.
    page.wait_for_load_state("networkidle")
    # Wait random delay (4-6 seconds) for dynamic content to mimic human behavior.
    delay = random.uniform(4, 6)
    _LOG.debug("Waiting %.2f seconds for dynamic content", delay)
    time.sleep(delay)
    # Create a raw CDP session.
    cdp = context.new_cdp_session(page)
    # Ask Chrome to snapshot the page as MHTML.
    result = cdp.send("Page.captureSnapshot", {"format": "mhtml"})
    # Save the MHTML content.
    output_path.write_text(result["data"], encoding="utf-8")
    _LOG.info("Saved MHTML: %s", output_path)


def _select_page_size_50(*, page) -> None:
    """
    Select 50 from dropdown menu near 'Show' text.

    :param page: Playwright page object.
    :raises AssertionError: If dropdown cannot be found or clicked.
    """
    _LOG.info("Attempting to find and click dropdown near 'Show' text...")

    # First, try to find and use a standard HTML select element.
    # Based on pb.png: dropdown is at bottom left, near "Show" text, currently showing "25".
    select_selectors = [
        "select:near(:text('Show'))",  # Select element near "Show" text.
        "select",
        "select[name*='pageSize']",
        "select[name*='page']",
        "select[name*='results']",
    ]
    for selector in select_selectors:
        try:
            element = page.query_selector(selector)
            if element:
                _LOG.info("Found HTML select element: %s", selector)
                page.select_option(selector, value="50")
                _LOG.info("✓ Selected 50 from HTML select element: %s", selector)
                page.wait_for_load_state("networkidle", timeout=10000)
                _LOG.info("✓ Page size set to 50")
                return
        except Exception as e:
            _LOG.debug("Failed with selector %s: %s", selector, str(e))
            continue

    # If HTML select didn't work, try custom dropdown approach.
    # Based on pb.png: Look for dropdown showing "25" near "Show" text at bottom of page.
    dropdown_selectors = [
        ":text('Show') >> xpath=following-sibling::select[1]",  # Select after "Show".
        ":text('Show') >> xpath=following-sibling::*[1]//select",  # Select in next sibling.
        "button:has-text('25'):near(:text('Show'))",  # Button showing current value "25".
        "div:has-text('25'):near(:text('Show'))",  # Div showing current value "25".
        ":text('Show') + select",  # Select immediately after "Show".
        ":text('Show') ~ select",  # Select as sibling of "Show".
        "[role='combobox']:near(:text('Show'))",  # ARIA combobox near "Show".
        "button:near(:text('Show'))",
        ":text('Show')",  # Try clicking "Show" text itself.
    ]

    clicked = False
    used_selector = None
    for selector in dropdown_selectors:
        try:
            _LOG.info("Trying dropdown selector: %s", selector)
            page.click(selector, timeout=2000)
            _LOG.info("✓ Clicked dropdown using selector: %s", selector)
            clicked = True
            used_selector = selector
            break
        except (TimeoutError, Exception) as e:
            _LOG.debug("Failed with selector %s: %s", selector, str(e))
            continue
    # Assert that dropdown was clicked.
    hdbg.dassert(
        clicked,
        "FAILED: Could not find or click dropdown near 'Show' text. Tried selectors:",
        dropdown_selectors,
    )

    _LOG.info("✓ Dropdown opened successfully with: %s", used_selector)

    # Wait for dropdown menu to appear.
    time.sleep(1.0)

    # Try to select the "50" option from dropdown menu.
    option_selectors = [
        "text=50",
        "[role='option']:text('50')",
        "button:has-text('50')",
        "a:has-text('50')",
        "[role='option']:has-text('50')",
        "li:has-text('50')",
        "[role='menuitem']:has-text('50')",
        ".dropdown-item:has-text('50')",
        "[data-value='50']",
        "option[value='50']",
    ]

    selected = False
    used_option_selector = None
    for selector in option_selectors:
        try:
            _LOG.info("Trying option selector: %s", selector)
            page.click(selector, timeout=2000)
            _LOG.info("✓ Selected 50 from dropdown using selector: %s", selector)
            selected = True
            used_option_selector = selector
            break
        except (TimeoutError, Exception) as e:
            _LOG.debug("Failed with selector %s: %s", selector, str(e))
            continue
    # Assert that option "50" was selected.
    hdbg.dassert(
        selected,
        "FAILED: Could not find or click option '50' in dropdown menu. Tried selectors:",
        option_selectors,
    )

    _LOG.info(
        "✓ Option '50' selected successfully with: %s", used_option_selector
    )

    # Wait for page to reload with new page size.
    page.wait_for_load_state("networkidle", timeout=10000)
    _LOG.info("✓ Page size set to 50 - page reloaded")


def _click_page_one(*, page) -> None:
    """
    Click on pagination button '1' to go to first page.

    :param page: Playwright page object.
    """
    # Try to find and click the '1' pagination button.
    # Common selectors for pagination buttons.
    selectors = [
        "button:has-text('1')",
        "a:has-text('1')",
        "[role='button']:has-text('1')",
        ".pagination button:has-text('1')",
        ".pagination a:has-text('1')",
    ]
    clicked = False
    for selector in selectors:
        try:
            page.click(selector, timeout=2000)
            _LOG.info(
                "Clicked pagination button '1' using selector: %s", selector
            )
            clicked = True
            break
        except TimeoutError:
            continue
    if not clicked:
        _LOG.warning(
            "Could not find pagination button '1', assuming already on page 1"
        )


def _has_next_button(*, page) -> bool:
    """
    Check if Next button exists and is enabled.

    :param page: Playwright page object.
    :return: True if Next button exists and is clickable.
    """
    # Common selectors for Next button.
    selectors = [
        "button:has-text('Next'):not([disabled])",
        "a:has-text('Next')",
        "[role='button']:has-text('Next'):not([disabled])",
        ".pagination button:has-text('Next'):not([disabled])",
        ".pagination a:has-text('Next')",
    ]
    for selector in selectors:
        try:
            element = page.query_selector(selector)
            if element and element.is_visible():
                return True
        except Exception:
            continue
    return False


def _click_next_button(*, page) -> bool:
    """
    Click the Next button to go to next page.

    :param page: Playwright page object.
    :return: True if clicked successfully, False otherwise.
    """
    # Common selectors for Next button.
    selectors = [
        "button:has-text('Next'):not([disabled])",
        "a:has-text('Next')",
        "[role='button']:has-text('Next'):not([disabled])",
        ".pagination button:has-text('Next'):not([disabled])",
        ".pagination a:has-text('Next')",
    ]
    for selector in selectors:
        try:
            page.click(selector, timeout=2000)
            _LOG.info("Clicked Next button using selector: %s", selector)
            # Wait for navigation to complete.
            page.wait_for_load_state("networkidle", timeout=10000)
            return True
        except TimeoutError:
            continue
    _LOG.warning("Could not find or click Next button")
    return False


def _get_total_pages(*, page) -> int:
    """
    Try to extract total number of pages from pagination UI.

    :param page: Playwright page object.
    :return: Total number of pages if found, 0 otherwise.
    """
    try:
        # Look for pagination text patterns like "Page 1 of 25" or "1 / 25".
        pagination_text = page.inner_text(".pagination", timeout=2000)
        _LOG.debug("Pagination text: %s", pagination_text)
        # Try to match patterns like "of X" or "/ X".
        match = re.search(r"(?:of|/)\s*(\d+)", pagination_text)
        if match:
            total = int(match.group(1))
            _LOG.info("Detected total pages: %d", total)
            return total
    except Exception as e:
        _LOG.debug("Could not extract total pages: %s", str(e))
    return 0


def _main(parser: argparse.ArgumentParser) -> None:
    """Main function to automate pagination and save MHTML files."""
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Create output directory.
    out_dir = Path(args.output_dir)
    hio.create_dir(str(out_dir), incremental=False)
    _LOG.info("Output directory: %s", out_dir)
    # Connect to Chrome via CDP.
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(CDP_URL)
        # Pick the already-open page.
        context = browser.contexts[0]
        page = context.pages[0]
        _LOG.info("Connected to Chrome browser")
        # Set page size to 50 results per page.
        _select_page_size_50(page=page)
        # Click on page 1 to start from the beginning.
        _click_page_one(page=page)
        # Wait for page to load after clicking.
        page.wait_for_load_state("networkidle")
        # Try to get total number of pages.
        total_pages = _get_total_pages(page=page)
        # Initialize page counter.
        page_num = 1
        # Create progress bar with or without total.
        if total_pages > 0:
            pbar = tqdm(total=total_pages, desc="Processing pages", unit="page")
        else:
            pbar = tqdm(desc="Processing pages", unit="page")
        # Loop through all pages.
        try:
            while True:
                # Generate output filename with zero-padded 3-digit number.
                output_path = out_dir / f"out{page_num:03d}.mhtml"
                # Save current page as MHTML.
                _save_page_as_mhtml(
                    page=page, context=context, output_path=output_path
                )
                # Update progress bar.
                pbar.update(1)
                # Check if Next button exists and is enabled.
                if not _has_next_button(page=page):
                    _LOG.info("No more pages to process")
                    break
                # Click Next button.
                if not _click_next_button(page=page):
                    _LOG.info("Could not click Next button, stopping")
                    break
                # Increment page counter.
                page_num += 1
        finally:
            # Close progress bar.
            pbar.close()
        # Log total number of pages processed.
        _LOG.info("Total pages processed: %d", page_num)


if __name__ == "__main__":
    _main(_parse())
