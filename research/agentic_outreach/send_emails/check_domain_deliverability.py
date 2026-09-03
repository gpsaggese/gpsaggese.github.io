#!/usr/bin/env python

"""
Check if a domain is healthy to send emails from.

This script performs comprehensive domain email deliverability checks including:
- MX records
- SPF records
- DKIM records (multiple selectors)
- DMARC records
- Reverse DNS
- SMTP banner and STARTTLS support
- DNSBL (Spamhaus) checking
- WHOIS domain age
- Final health score and verdict

Example usage:
    ./check_domain_deliverability.py example.com
    ./check_domain_deliverability.py example.com --dkim-selectors default,google
    ./check_domain_deliverability.py example.com --verbose

Import as:

import check_domain_deliverability as chdomde
"""

import argparse
import dns.resolver
import dns.reversename
import logging
import re
import smtplib
import socket
import whois
from typing import Dict, List, Optional, Tuple

import helpers.hdbg as hdbg
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

# Default DKIM selectors to check.
DEFAULT_DKIM_SELECTORS = ["default", "selector1", "selector2", "google", "k1"]

# Spamhaus DNSBL zone.
SPAMHAUS_DNSBL = "zen.spamhaus.org"

# Weight configuration for scoring.
SCORE_WEIGHTS = {
    "mx": 15,
    "spf": 15,
    "dkim": 15,
    "dmarc": 15,
    "rdns": 10,
    "smtp": 10,
    "starttls": 10,
    "dnsbl": 5,
    "domain_age": 5,
}

# #############################################################################
# DNS Checking Functions
# #############################################################################


def _check_mx_records(domain: str) -> Tuple[bool, List[str], str]:
    """
    Check if domain has valid MX records.

    :param domain: domain name to check
    :return: tuple of (success, mx_records, message)
    """
    _LOG.debug("Checking MX records for domain='%s'", domain)
    try:
        mx_records = dns.resolver.resolve(domain, "MX")
        mx_hosts = [str(r.exchange).rstrip(".") for r in mx_records]
        if mx_hosts:
            message = (
                f"Found {len(mx_hosts)} MX record(s): {', '.join(mx_hosts[:3])}"
            )
            _LOG.info("MX check passed: %s", message)
            return True, mx_hosts, message
        else:
            message = "No MX records found"
            _LOG.warning("MX check failed: %s", message)
            return False, [], message
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN) as e:
        message = f"DNS error: {str(e)}"
        _LOG.warning("MX check failed: %s", message)
        return False, [], message
    except Exception as e:
        message = f"Unexpected error: {str(e)}"
        _LOG.error("MX check error: %s", message)
        return False, [], message


def _check_spf_record(domain: str) -> Tuple[bool, Optional[str], str]:
    """
    Check if domain has valid SPF record.

    :param domain: domain name to check
    :return: tuple of (success, spf_record, message)
    """
    _LOG.debug("Checking SPF record for domain='%s'", domain)
    try:
        txt_records = dns.resolver.resolve(domain, "TXT")
        for record in txt_records:
            txt_value = record.to_text().strip('"')
            if txt_value.startswith("v=spf1"):
                message = f"SPF record found: {txt_value[:80]}"
                _LOG.info("SPF check passed: %s", message)
                return True, txt_value, message
        # No SPF record found.
        message = "No SPF record found"
        _LOG.warning("SPF check failed: %s", message)
        return False, None, message
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN) as e:
        message = f"DNS error: {str(e)}"
        _LOG.warning("SPF check failed: %s", message)
        return False, None, message
    except Exception as e:
        message = f"Unexpected error: {str(e)}"
        _LOG.error("SPF check error: %s", message)
        return False, None, message


def _check_dkim_records(
    domain: str, *, selectors: List[str] = DEFAULT_DKIM_SELECTORS
) -> Tuple[bool, List[str], str]:
    """
    Check if domain has valid DKIM records for given selectors.

    :param domain: domain name to check
    :param selectors: list of DKIM selectors to check
    :return: tuple of (success, found_selectors, message)
    """
    _LOG.debug(
        "Checking DKIM records for domain='%s' with selectors=%s",
        domain,
        selectors,
    )
    found_selectors = []
    for selector in selectors:
        dkim_domain = f"{selector}._domainkey.{domain}"
        try:
            txt_records = dns.resolver.resolve(dkim_domain, "TXT")
            for record in txt_records:
                txt_value = record.to_text()
                if "v=DKIM1" in txt_value or "p=" in txt_value:
                    found_selectors.append(selector)
                    _LOG.debug("DKIM selector '%s' found", selector)
                    break
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
            _LOG.debug("DKIM selector '%s' not found", selector)
            continue
        except Exception as e:
            _LOG.debug("Error checking selector '%s': %s", selector, str(e))
            continue
    # Evaluate results.
    if found_selectors:
        message = f"Found DKIM for selector(s): {', '.join(found_selectors)}"
        _LOG.info("DKIM check passed: %s", message)
        return True, found_selectors, message
    else:
        message = f"No DKIM records found for selectors: {', '.join(selectors)}"
        _LOG.warning("DKIM check failed: %s", message)
        return False, [], message


def _check_dmarc_record(domain: str) -> Tuple[bool, Optional[str], str]:
    """
    Check if domain has valid DMARC record.

    :param domain: domain name to check
    :return: tuple of (success, dmarc_record, message)
    """
    _LOG.debug("Checking DMARC record for domain='%s'", domain)
    dmarc_domain = f"_dmarc.{domain}"
    try:
        txt_records = dns.resolver.resolve(dmarc_domain, "TXT")
        for record in txt_records:
            txt_value = record.to_text().strip('"')
            if txt_value.startswith("v=DMARC1"):
                # Extract policy.
                policy_match = re.search(r"p=(\w+)", txt_value)
                policy = policy_match.group(1) if policy_match else "unknown"
                message = f"DMARC record found with policy={policy}"
                _LOG.info("DMARC check passed: %s", message)
                return True, txt_value, message
        # No DMARC record found.
        message = "No DMARC record found"
        _LOG.warning("DMARC check failed: %s", message)
        return False, None, message
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN) as e:
        message = f"DNS error: {str(e)}"
        _LOG.warning("DMARC check failed: %s", message)
        return False, None, message
    except Exception as e:
        message = f"Unexpected error: {str(e)}"
        _LOG.error("DMARC check error: %s", message)
        return False, None, message


def _check_reverse_dns(domain: str, *, mx_hosts: List[str]) -> Tuple[bool, str]:
    """
    Check if MX hosts have valid reverse DNS.

    :param domain: domain name being checked
    :param mx_hosts: list of MX hosts to check
    :return: tuple of (success, message)
    """
    _LOG.debug("Checking reverse DNS for MX hosts: %s", mx_hosts)
    if not mx_hosts:
        message = "No MX hosts to check"
        _LOG.warning("Reverse DNS check skipped: %s", message)
        return False, message
    # Check first MX host.
    mx_host = mx_hosts[0]
    try:
        # Get IP address of MX host.
        ip_address = socket.gethostbyname(mx_host)
        _LOG.debug("MX host '%s' resolves to IP '%s'", mx_host, ip_address)
        # Get reverse DNS.
        rev_name = dns.reversename.from_address(ip_address)
        ptr_records = dns.resolver.resolve(rev_name, "PTR")
        ptr_hostname = str(ptr_records[0]).rstrip(".")
        message = f"Reverse DNS: {ip_address} -> {ptr_hostname}"
        _LOG.info("Reverse DNS check passed: %s", message)
        return True, message
    except socket.gaierror as e:
        message = f"Cannot resolve MX host '{mx_host}': {str(e)}"
        _LOG.warning("Reverse DNS check failed: %s", message)
        return False, message
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN) as e:
        message = f"No PTR record for {ip_address}: {str(e)}"
        _LOG.warning("Reverse DNS check failed: %s", message)
        return False, message
    except Exception as e:
        message = f"Unexpected error: {str(e)}"
        _LOG.error("Reverse DNS check error: %s", message)
        return False, message


# #############################################################################
# SMTP Checking Functions
# #############################################################################


def _check_smtp_connection(
    domain: str, *, mx_hosts: List[str]
) -> Tuple[bool, str]:
    """
    Check if can connect to SMTP server and get banner.

    :param domain: domain name being checked
    :param mx_hosts: list of MX hosts to check
    :return: tuple of (success, message)
    """
    _LOG.debug("Checking SMTP connection for MX hosts: %s", mx_hosts)
    if not mx_hosts:
        message = "No MX hosts to check"
        _LOG.warning("SMTP check skipped: %s", message)
        return False, message
    # Try first MX host.
    mx_host = mx_hosts[0]
    try:
        smtp = smtplib.SMTP(mx_host, 25, timeout=10)
        banner = smtp.ehlo_resp.decode() if smtp.ehlo_resp else "No banner"
        smtp.quit()
        message = f"SMTP connection successful: {banner[:60]}"
        _LOG.info("SMTP check passed: %s", message)
        return True, message
    except (smtplib.SMTPException, socket.error) as e:
        message = f"SMTP connection failed: {str(e)}"
        _LOG.warning("SMTP check failed: %s", message)
        return False, message
    except Exception as e:
        message = f"Unexpected error: {str(e)}"
        _LOG.error("SMTP check error: %s", message)
        return False, message


def _check_starttls(domain: str, *, mx_hosts: List[str]) -> Tuple[bool, str]:
    """
    Check if SMTP server supports STARTTLS.

    :param domain: domain name being checked
    :param mx_hosts: list of MX hosts to check
    :return: tuple of (success, message)
    """
    _LOG.debug("Checking STARTTLS support for MX hosts: %s", mx_hosts)
    if not mx_hosts:
        message = "No MX hosts to check"
        _LOG.warning("STARTTLS check skipped: %s", message)
        return False, message
    # Try first MX host.
    mx_host = mx_hosts[0]
    try:
        smtp = smtplib.SMTP(mx_host, 25, timeout=10)
        smtp.starttls()
        smtp.quit()
        message = "STARTTLS is supported"
        _LOG.info("STARTTLS check passed: %s", message)
        return True, message
    except smtplib.SMTPNotSupportedError:
        message = "STARTTLS not supported"
        _LOG.warning("STARTTLS check failed: %s", message)
        return False, message
    except (smtplib.SMTPException, socket.error) as e:
        message = f"STARTTLS check failed: {str(e)}"
        _LOG.warning("STARTTLS check failed: %s", message)
        return False, message
    except Exception as e:
        message = f"Unexpected error: {str(e)}"
        _LOG.error("STARTTLS check error: %s", message)
        return False, message


# #############################################################################
# Blacklist and Domain Age Checking
# #############################################################################


def _check_dnsbl(domain: str, *, mx_hosts: List[str]) -> Tuple[bool, str]:
    """
    Check if domain's MX hosts are listed in Spamhaus DNSBL.

    :param domain: domain name being checked
    :param mx_hosts: list of MX hosts to check
    :return: tuple of (success, message)
    """
    _LOG.debug("Checking DNSBL for MX hosts: %s", mx_hosts)
    if not mx_hosts:
        message = "No MX hosts to check"
        _LOG.warning("DNSBL check skipped: %s", message)
        return False, message
    # Check first MX host.
    mx_host = mx_hosts[0]
    try:
        # Get IP address.
        ip_address = socket.gethostbyname(mx_host)
        _LOG.debug("Checking IP '%s' against Spamhaus", ip_address)
        # Reverse IP octets for DNSBL query.
        ip_parts = ip_address.split(".")
        reversed_ip = ".".join(reversed(ip_parts))
        dnsbl_query = f"{reversed_ip}.{SPAMHAUS_DNSBL}"
        # Query DNSBL.
        dns.resolver.resolve(dnsbl_query, "A")
        # If we get here, IP is listed.
        message = f"IP {ip_address} is listed in Spamhaus"
        _LOG.warning("DNSBL check failed: %s", message)
        return False, message
    except dns.resolver.NXDOMAIN:
        # Not listed (expected for clean IPs).
        message = "IP not listed in Spamhaus (clean)"
        _LOG.info("DNSBL check passed: %s", message)
        return True, message
    except socket.gaierror as e:
        message = f"Cannot resolve MX host '{mx_host}': {str(e)}"
        _LOG.warning("DNSBL check failed: %s", message)
        return False, message
    except Exception as e:
        message = f"DNSBL check error: {str(e)}"
        _LOG.debug("DNSBL check error: %s", message)
        # Assume clean on error.
        return True, message


def _check_domain_age(domain: str) -> Tuple[bool, str]:
    """
    Check domain age using WHOIS.

    :param domain: domain name to check
    :return: tuple of (success, message)
    """
    _LOG.debug("Checking domain age for domain='%s'", domain)
    try:
        w = whois.whois(domain)
        creation_date = w.creation_date
        # Handle multiple dates (some WHOIS return list).
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        # Calculate age.
        if creation_date:
            from datetime import datetime

            # Make creation_date timezone-naive if it's timezone-aware.
            if (
                hasattr(creation_date, "tzinfo")
                and creation_date.tzinfo is not None
            ):
                creation_date = creation_date.replace(tzinfo=None)
            age_days = (datetime.now() - creation_date).days
            age_years = age_days / 365.25
            # Domain older than 6 months is good.
            if age_days >= 180:
                message = f"Domain age: {age_years:.1f} years (good)"
                _LOG.info("Domain age check passed: %s", message)
                return True, message
            else:
                message = f"Domain age: {age_days} days (too new)"
                _LOG.warning("Domain age check failed: %s", message)
                return False, message
        else:
            message = "Creation date not available"
            _LOG.warning("Domain age check failed: %s", message)
            return False, message
    except Exception as e:
        message = f"WHOIS error: {str(e)}"
        _LOG.warning("Domain age check failed: %s", message)
        return False, message


# #############################################################################
# Scoring and Reporting
# #############################################################################


def _calculate_health_score(results: Dict[str, bool]) -> Tuple[int, str]:
    """
    Calculate overall health score based on check results.

    :param results: dictionary of check results
    :return: tuple of (score, verdict)
    """
    _LOG.debug("Calculating health score from results: %s", results)
    total_score = 0
    max_score = sum(SCORE_WEIGHTS.values())
    # Calculate weighted score.
    for check_name, passed in results.items():
        if passed:
            weight = SCORE_WEIGHTS.get(check_name, 0)
            total_score += weight
            _LOG.debug("Check '%s' passed, adding %d points", check_name, weight)
    # Calculate percentage.
    score_pct = int((total_score / max_score) * 100)
    # Determine verdict.
    if score_pct >= 90:
        verdict = "EXCELLENT"
    elif score_pct >= 75:
        verdict = "GOOD"
    elif score_pct >= 60:
        verdict = "FAIR"
    elif score_pct >= 40:
        verdict = "POOR"
    else:
        verdict = "CRITICAL"
    _LOG.info("Health score: %d%% (%s)", score_pct, verdict)
    return score_pct, verdict


def _print_report(
    domain: str,
    *,
    results: Dict[str, bool],
    messages: Dict[str, str],
    score: int,
    verdict: str,
) -> None:
    """
    Print comprehensive health report.

    :param domain: domain name checked
    :param results: dictionary of check results
    :param messages: dictionary of check messages
    :param score: overall health score
    :param verdict: health verdict
    """
    _LOG.info("=" * 70)
    _LOG.info("Domain Deliverability Report for: %s", domain)
    _LOG.info("=" * 70)
    _LOG.info("")
    # Print individual checks.
    check_names = {
        "mx": "MX Records",
        "spf": "SPF Record",
        "dkim": "DKIM Records",
        "dmarc": "DMARC Record",
        "rdns": "Reverse DNS",
        "smtp": "SMTP Connection",
        "starttls": "STARTTLS Support",
        "dnsbl": "DNSBL Check (Spamhaus)",
        "domain_age": "Domain Age",
    }
    for check_key, check_name in check_names.items():
        status = "✓ PASS" if results.get(check_key, False) else "✗ FAIL"
        message = messages.get(check_key, "No information")
        _LOG.info("%s: %s", check_name.ljust(25), status)
        _LOG.info("  %s", message)
        _LOG.info("")
    # Print summary.
    _LOG.info("=" * 70)
    _LOG.info("OVERALL HEALTH SCORE: %d%% (%s)", score, verdict)
    _LOG.info("=" * 70)


# #############################################################################
# Main Script Functions
# #############################################################################


def _run_all_checks(
    domain: str, *, dkim_selectors: List[str]
) -> Tuple[Dict[str, bool], Dict[str, str]]:
    """
    Run all deliverability checks for the domain.

    :param domain: domain name to check
    :param dkim_selectors: list of DKIM selectors to check
    :return: tuple of (results_dict, messages_dict)
    """
    _LOG.info("Starting deliverability checks for domain='%s'", domain)
    results = {}
    messages = {}
    # Check MX records.
    mx_success, mx_hosts, mx_msg = _check_mx_records(domain)
    results["mx"] = mx_success
    messages["mx"] = mx_msg
    # Check SPF.
    spf_success, _, spf_msg = _check_spf_record(domain)
    results["spf"] = spf_success
    messages["spf"] = spf_msg
    # Check DKIM.
    dkim_success, _, dkim_msg = _check_dkim_records(
        domain, selectors=dkim_selectors
    )
    results["dkim"] = dkim_success
    messages["dkim"] = dkim_msg
    # Check DMARC.
    dmarc_success, _, dmarc_msg = _check_dmarc_record(domain)
    results["dmarc"] = dmarc_success
    messages["dmarc"] = dmarc_msg
    # Check reverse DNS.
    rdns_success, rdns_msg = _check_reverse_dns(domain, mx_hosts=mx_hosts)
    results["rdns"] = rdns_success
    messages["rdns"] = rdns_msg
    # Check SMTP connection.
    smtp_success, smtp_msg = _check_smtp_connection(domain, mx_hosts=mx_hosts)
    results["smtp"] = smtp_success
    messages["smtp"] = smtp_msg
    # Check STARTTLS.
    starttls_success, starttls_msg = _check_starttls(domain, mx_hosts=mx_hosts)
    results["starttls"] = starttls_success
    messages["starttls"] = starttls_msg
    # Check DNSBL.
    dnsbl_success, dnsbl_msg = _check_dnsbl(domain, mx_hosts=mx_hosts)
    results["dnsbl"] = dnsbl_success
    messages["dnsbl"] = dnsbl_msg
    # Check domain age.
    age_success, age_msg = _check_domain_age(domain)
    results["domain_age"] = age_success
    messages["domain_age"] = age_msg
    _LOG.info("Completed all checks for domain='%s'", domain)
    return results, messages


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("domain", help="Domain name to check")
    parser.add_argument(
        "--dkim-selectors",
        action="store",
        default=",".join(DEFAULT_DKIM_SELECTORS),
        help=f"Comma-separated list of DKIM selectors (default: {','.join(DEFAULT_DKIM_SELECTORS)})",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    # Validate domain.
    domain = args.domain.strip().lower()
    hdbg.dassert_ne(domain, "", "Domain cannot be empty")
    # Parse DKIM selectors.
    dkim_selectors = [s.strip() for s in args.dkim_selectors.split(",")]
    hdbg.dassert_lt(
        0, len(dkim_selectors), "At least one DKIM selector required"
    )
    # Run all checks.
    results, messages = _run_all_checks(domain, dkim_selectors=dkim_selectors)
    # Calculate score.
    score, verdict = _calculate_health_score(results)
    # Print report.
    _print_report(
        domain, results=results, messages=messages, score=score, verdict=verdict
    )


if __name__ == "__main__":
    _main(_parse())
