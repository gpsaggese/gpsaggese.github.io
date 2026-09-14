#!/usr/bin/env python

r"""
Send emails using SendGrid API and track their status.

This script reads a CSV file with recipient information (first_name, last_name,
email), sends personalized emails using SendGrid, and tracks email status
including opens and unsubscribes by polling the SendGrid Stats API.

Usage examples:
    # Send emails and track status
    ./send_emails.py \\
        --csv-file recipients.csv \\
        --template-file email_template.json \\
        --from-email sender@example.com \\
        --output-file email_tracking.csv

    # With verbose logging
    ./send_emails.py \\
        --csv-file recipients.csv \\
        --template-file email_template.json \\
        --from-email sender@example.com \\
        --output-file email_tracking.csv \\
        -v DEBUG

Template file format (JSON):
    {
        "subject": "Hello {first_name}!",
        "body": "Dear {first_name} {last_name}, ..."
    }

CSV file format:
    first_name,last_name,email
    John,Doe,john@example.com
    Jane,Smith,jane@example.com

Environment variables required:
    SENDGRID_API_KEY: SendGrid API key for authentication

Import as:

import ai_outreach.send_emails.send_emails as aoseseem
"""

import argparse
import csv
import datetime
import json
import logging
import os
import time
from typing import Any, Dict, List, cast

import helpers.hdbg as hdbg
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)

try:
    import sendgrid  # type: ignore[import-not-found]
    import sendgrid.helpers.mail as sghema  # type: ignore[import-not-found]
except ImportError:
    os.system("sudo bin/venv/pip install sendgrid")
    raise

# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv-file",
        action="store",
        required=True,
        help="Path to CSV file with columns: first_name, last_name, email",
    )
    parser.add_argument(
        "--template-file",
        action="store",
        required=True,
        help="Path to JSON file with email template (keys: subject, body)",
    )
    parser.add_argument(
        "--from-email",
        action="store",
        required=True,
        help="Sender email address",
    )
    parser.add_argument(
        "--output-file",
        action="store",
        default="email_tracking.csv",
        help="Path to output CSV file for tracking email status (default: email_tracking.csv)",
    )
    parser.add_argument(
        "--poll-delay",
        action="store",
        type=int,
        default=60,
        help="Delay in seconds between sending emails and polling for stats (default: 60)",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def load_template(template_file: str) -> Dict[str, str]:
    """
    Load email template from JSON file.

    :param template_file: path to JSON file with 'subject' and 'body'
        keys
    :return: dictionary with 'subject' and 'body' keys
    """
    _LOG.info("Loading email template from: %s", template_file)
    hdbg.dassert_path_exists(template_file)
    with open(template_file, "r", encoding="utf-8") as f:
        template = json.load(f)
    hdbg.dassert_in("subject", template, "Template must contain 'subject' key")
    hdbg.dassert_in("body", template, "Template must contain 'body' key")
    _LOG.debug("Template loaded - subject: %s", template["subject"])
    return cast(Dict[str, str], template)


def load_recipients(csv_file: str) -> List[Dict[str, str]]:
    """
    Load recipient information from CSV file.

    :param csv_file: path to CSV file with columns: first_name,
        last_name, email
    :return: list of dictionaries with recipient information
    """
    _LOG.info("Loading recipients from: %s", csv_file)
    hdbg.dassert_path_exists(csv_file)
    recipients = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = ["first_name", "last_name", "email"]
        for col in required_cols:
            hdbg.dassert_in(
                col,
                reader.fieldnames,
                "CSV must contain column:",
                col,
            )
        for row in reader:
            recipients.append(row)
    _LOG.info("Loaded %d recipients", len(recipients))
    return recipients


def personalize_template(
    template: Dict[str, str],
    *,
    first_name: str,
    last_name: str,
) -> Dict[str, str]:
    """
    Personalize email template with recipient information.

    :param template: template dictionary with 'subject' and 'body' keys
    :param first_name: recipient's first name
    :param last_name: recipient's last name
    :return: personalized template with placeholders replaced
    """
    personalized = {}
    personalized["subject"] = template["subject"].format(
        first_name=first_name,
        last_name=last_name,
    )
    personalized["body"] = template["body"].format(
        first_name=first_name,
        last_name=last_name,
    )
    return personalized


def send_email(
    sg_client: "sendgrid.SendGridAPIClient",
    *,
    from_email: str,
    to_email: str,
    subject: str,
    body: str,
) -> Dict[str, Any]:
    """
    Send email using SendGrid API.

    :param sg_client: SendGrid API client
    :param from_email: sender email address
    :param to_email: recipient email address
    :param subject: email subject
    :param body: email body
    :return: dictionary with send result information
    """
    _LOG.debug("Sending email to: %s", to_email)
    from_email_obj = sghema.Email(from_email)
    to_email_obj = sghema.To(to_email)
    content = sghema.Content("text/plain", body)
    mail = sghema.Mail(from_email_obj, to_email_obj, subject, content)
    mail.tracking_settings = {
        "click_tracking": {"enable": True},
        "open_tracking": {"enable": True},
    }
    response = sg_client.client.mail.send.post(request_body=mail.get())
    result = {
        "to_email": to_email,
        "status_code": response.status_code,
        "success": response.status_code in [200, 202],
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
    if result["success"]:
        _LOG.info("Email sent successfully to: %s", to_email)
    else:
        _LOG.warning(
            "Email send failed to: %s (status: %d)",
            to_email,
            response.status_code,
        )
    return result


def send_emails(
    recipients: List[Dict[str, str]],
    template: Dict[str, str],
    *,
    from_email: str,
    api_key: str,
) -> List[Dict[str, Any]]:
    """
    Send emails to all recipients.

    :param recipients: list of recipient dictionaries
    :param template: email template dictionary
    :param from_email: sender email address
    :param api_key: SendGrid API key
    :return: list of send result dictionaries
    """
    _LOG.info("Sending emails to %d recipients", len(recipients))
    sg_client = sendgrid.SendGridAPIClient(api_key=api_key)
    results = []
    for idx, recipient in enumerate(recipients, 1):
        _LOG.info(
            "Processing recipient %d/%d: %s",
            idx,
            len(recipients),
            recipient["email"],
        )
        personalized = personalize_template(
            template,
            first_name=recipient["first_name"],
            last_name=recipient["last_name"],
        )
        result = send_email(
            sg_client,
            from_email=from_email,
            to_email=recipient["email"],
            subject=personalized["subject"],
            body=personalized["body"],
        )
        result.update(
            {
                "first_name": recipient["first_name"],
                "last_name": recipient["last_name"],
            }
        )
        results.append(result)
        if idx < len(recipients):
            time.sleep(0.5)
    _LOG.info("Completed sending emails")
    return results


def poll_email_stats(
    api_key: str,
    *,
    start_date: str,
) -> Dict[str, Dict[str, int]]:
    """
    Poll SendGrid Stats API to get email statistics.

    :param api_key: SendGrid API key
    :param start_date: start date in YYYY-MM-DD format
    :return: dictionary mapping email addresses to stats
    """
    _LOG.info("Polling SendGrid Stats API for date: %s", start_date)
    sg_client = sendgrid.SendGridAPIClient(api_key=api_key)
    params = {
        "start_date": start_date,
        "aggregated_by": "day",
    }
    response = sg_client.client.stats.get(query_params=params)
    stats = {}
    if response.status_code == 200:
        data = json.loads(response.body)
        _LOG.debug("Stats API response: %s", data)
        _LOG.warning(
            "SendGrid Stats API provides aggregate data only. "
            "For per-email tracking, consider using Event Webhook."
        )
        if data:
            for day_stats in data:
                stats["aggregate"] = {
                    "delivered": day_stats.get("stats", [{}])[0]
                    .get("metrics", {})
                    .get("delivered", 0),
                    "opens": day_stats.get("stats", [{}])[0]
                    .get("metrics", {})
                    .get("opens", 0),
                    "unique_opens": day_stats.get("stats", [{}])[0]
                    .get("metrics", {})
                    .get("unique_opens", 0),
                    "clicks": day_stats.get("stats", [{}])[0]
                    .get("metrics", {})
                    .get("clicks", 0),
                    "unsubscribes": day_stats.get("stats", [{}])[0]
                    .get("metrics", {})
                    .get("unsubscribes", 0),
                }
    else:
        _LOG.warning(
            "Stats API request failed with status: %d", response.status_code
        )
    return stats


def write_tracking_csv(
    results: List[Dict[str, Any]],
    output_file: str,
    *,
    stats: Dict[str, Dict[str, int]],
) -> None:
    """
    Write email tracking results to CSV file.

    :param results: list of send result dictionaries
    :param output_file: path to output CSV file
    :param stats: dictionary with aggregate email statistics
    """
    _LOG.info("Writing tracking results to: %s", output_file)
    fieldnames = [
        "first_name",
        "last_name",
        "to_email",
        "sent_timestamp",
        "send_success",
        "status_code",
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "first_name": result["first_name"],
                "last_name": result["last_name"],
                "to_email": result["to_email"],
                "sent_timestamp": result["timestamp"],
                "send_success": result["success"],
                "status_code": result["status_code"],
            }
            writer.writerow(row)
    _LOG.info("Wrote %d tracking records to: %s", len(results), output_file)
    if "aggregate" in stats:
        _LOG.info("Aggregate stats:")
        for metric, value in stats["aggregate"].items():
            _LOG.info("  %s: %d", metric, value)


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    api_key = os.environ.get("SENDGRID_API_KEY")
    hdbg.dassert_is_not(
        api_key,
        None,
        "SENDGRID_API_KEY environment variable must be set",
    )
    template = load_template(args.template_file)
    recipients = load_recipients(args.csv_file)
    results = send_emails(
        recipients,
        template,
        from_email=args.from_email,
        api_key=api_key,
    )
    _LOG.info("Waiting %d seconds before polling stats...", args.poll_delay)
    time.sleep(args.poll_delay)
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    stats = poll_email_stats(api_key, start_date=today)
    write_tracking_csv(results, args.output_file, stats=stats)
    _LOG.info("Email sending and tracking completed")


if __name__ == "__main__":
    _main(_parse())
