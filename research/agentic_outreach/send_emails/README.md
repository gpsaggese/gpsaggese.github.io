# SendGrid Email Sender Script

## Overview

`send_emails.py` is a script that sends personalized emails using the SendGrid API and tracks their status (opens, unsubscribes, etc.).

## Prerequisites

1. Install SendGrid Python library:
   ```bash
   pip install sendgrid
   ```

2. Set up SendGrid API key as environment variable:
   ```bash
   export SENDGRID_API_KEY='your-api-key-here'
   ```

## Usage

### Basic Command

```bash
./send_emails.py \
    --csv-file recipients.csv \
    --template-file email_template.json \
    --from-email sender@example.com \
    --output-file email_tracking.csv
```

### With Verbose Logging

```bash
./send_emails.py \
    --csv-file recipients.csv \
    --template-file email_template.json \
    --from-email sender@example.com \
    --output-file email_tracking.csv \
    -v DEBUG
```

### Custom Poll Delay

```bash
./send_emails.py \
    --csv-file recipients.csv \
    --template-file email_template.json \
    --from-email sender@example.com \
    --output-file email_tracking.csv \
    --poll-delay 120
```

## File Formats

### Input CSV Format

The CSV file must contain the following columns:
- `first_name`: Recipient's first name
- `last_name`: Recipient's last name
- `email`: Recipient's email address

Example (`recipients_example.csv`):
```csv
first_name,last_name,email
John,Doe,john.doe@example.com
Jane,Smith,jane.smith@example.com
Bob,Johnson,bob.johnson@example.com
```

### Email Template Format

The template file must be in JSON format with two keys:
- `subject`: Email subject line (can use {first_name} and {last_name} placeholders)
- `body`: Email body text (can use {first_name} and {last_name} placeholders)

Example (`email_template_example.json`):
```json
{
  "subject": "Hello {first_name}!",
  "body": "Dear {first_name} {last_name},\n\nThis is a personalized email sent to you.\n\nBest regards,\nThe Team"
}
```

### Output CSV Format

The output CSV file will contain:
- `first_name`: Recipient's first name
- `last_name`: Recipient's last name
- `to_email`: Recipient's email address
- `sent_timestamp`: ISO format timestamp when email was sent
- `send_success`: Boolean indicating if send was successful
- `status_code`: HTTP status code from SendGrid API

## Command-Line Arguments

- `--csv-file` (required): Path to input CSV file with recipients
- `--template-file` (required): Path to JSON file with email template
- `--from-email` (required): Sender email address
- `--output-file` (optional): Path to output CSV file for tracking (default: `email_tracking.csv`)
- `--poll-delay` (optional): Delay in seconds between sending emails and polling for stats (default: 60)
- `-v`, `--verbosity` (optional): Logging verbosity level (DEBUG, INFO, WARNING, ERROR)

## Email Tracking

The script tracks emails in two ways:

1. **Send Status**: Immediately tracked when sending each email (success/failure, status code, timestamp)

2. **Aggregate Statistics**: After sending all emails, the script polls the SendGrid Stats API to get aggregate statistics including:
   - Delivered count
   - Opens and unique opens
   - Clicks
   - Unsubscribes

**Important Note**: The SendGrid Stats API provides aggregate data only, not per-email granularity. For per-email tracking of opens and clicks, you would need to set up SendGrid Event Webhook, which requires a publicly accessible endpoint to receive webhook events.

## Example Files

This directory includes example files to help you get started:

- `recipients_example.csv`: Sample recipient list
- `email_template_example.json`: Sample email template

## Troubleshooting

### "SENDGRID_API_KEY environment variable must be set"

Make sure you've exported your SendGrid API key:
```bash
export SENDGRID_API_KEY='your-api-key-here'
```

### "sendgrid library not installed"

Install the SendGrid library:
```bash
pip install sendgrid
```

### Rate Limiting

The script includes a 0.5-second delay between sending emails to avoid rate limiting. If you encounter rate limit errors, you can:
1. Reduce the batch size
2. Increase the delay in the code
3. Use SendGrid's Marketing Campaigns feature for bulk sends

## SendGrid Setup

To use this script, you need:

1. A SendGrid account (free tier available)
2. An API key with "Mail Send" permissions
3. A verified sender email address in SendGrid

Visit https://sendgrid.com to set up your account.
