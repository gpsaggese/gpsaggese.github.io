#!/usr/bin/env python3
"""Count teams by National I-Corps Status from CSV file."""

import csv
import sys
from collections import defaultdict


def main(csv_file):
    # Define all statuses from count_national.md
    all_statuses = [
        "03",
        "04",
        "A.01",
        "A.02",
        "A.03",
        "A.04",
        "A.05",
        "A.06",
        "A.07",
        "B.01",
        "B.02",
        "B.03",
        "C.01",
        "C.02",
        "C.03",
        "C.04",
        "D.01",
        "D.02",
        "D.03",
        "D.04",
        "D.05",
        "E.02",
        "E.04",
        "E.08",
    ]

    status_names = {
        "03": "Eligible",
        "04": "Expressed Interest",
        "A.01": "Executive Summary Review",
        "A.02": "Executive Summary Feedback Sent",
        "A.03": "Mentor Search",
        "A.04": "PI Search",
        "A.05": "Send LOR instructions",
        "A.06": "Waiting for LOR Draft from Team",
        "A.07": "Pre-Interview Document Review",
        "B.01": "Going through Regionals",
        "B.02": "Paused",
        "B.03": "Withdrawn",
        "C.01": "Scheduling for Interview",
        "C.02": "Interview Scheduled",
        "C.03": "Interview done",
        "C.04": "Post-Interview Action Required",
        "D.01": "Accepted by Hub",
        "D.02": "Waiting for LOR from Dan",
        "D.03": "Send NSF Application Instructions",
        "D.04": "NSF Application Sent",
        "D.05": "Applied to NSF",
        "E.02": "Accepted by NSF",
        "E.04": "Rejected",
        "E.08": "Completed",
    }

    # Count status occurrences
    status_counts = defaultdict(int)

    # Read CSV file
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get("National I-Corps Status", "").strip()
            # Extract just the status code (e.g., "A.04" from "A.04: PI Search")
            if status:
                status_code = status.split(":")[0].strip()
                status_counts[status_code] += 1

    # Print results with tab separation
    print("Status\tCount")
    for status in all_statuses:
        count = status_counts.get(status, 0)
        name = status_names[status]
        print(f"{status}: {name}\t{count}")

    # Print total
    total = sum(status_counts.values())
    print(f"\nTotal\t{total}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_national_csv.py <csv_file>")
        sys.exit(1)

    main(sys.argv[1])
