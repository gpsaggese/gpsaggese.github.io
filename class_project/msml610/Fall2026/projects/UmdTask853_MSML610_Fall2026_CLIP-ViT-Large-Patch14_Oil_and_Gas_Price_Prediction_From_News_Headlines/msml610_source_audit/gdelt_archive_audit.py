#!/usr/bin/env python3
"""Small, reproducible GDELT 2.0 GKG news-title pilot via public ZIP archives.

Uses downloaded 15-minute GKG slices, NOT the rate-limited DOC API.
Samples are not exhaustive historical news coverage.
Requires the project's audit_fnspid_gdelt.py next to this file for user's exact regex.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
from html import unescape
from io import BytesIO, TextIOWrapper
import json
from pathlib import Path
import re
import zipfile

import pandas as pd
import requests
from audit_fnspid_gdelt import news_flags

BASE = 'https://data.gdeltproject.org/gdeltv2/'
PAGE_TITLE = re.compile(r'<PAGE_TITLE>(.*?)</PAGE_TITLE>', re.DOTALL | re.I)
PUB_TIME = re.compile(r'<PAGE_PRECISEPUBTIMESTAMP>(\d{14})</PAGE_PRECISEPUBTIMESTAMP>', re.I)
GKG_DATE = re.compile(r'^\d{14}$')


def parse_gkg_rows(zipped_bytes: bytes, archive_url: str):
    """Parse selected GKG v2 fields from ZIP: record time, domain, URL, tone, title, pub time."""
    scanned = titles = missing = 0
    rows = []
    with zipfile.ZipFile(BytesIO(zipped_bytes)) as zin:
        fname = next((name for name in zin.namelist() if not name.endswith('/')), None)
        if not fname:
            raise ValueError('Empty GDELT GKG ZIP')
        with zin.open(fname) as raw:
            for line in TextIOWrapper(raw, encoding='utf-8', errors='replace'):
                cols = line.rstrip('\r\n').split('\t')
                scanned += 1
                if len(cols) < 27:
                    missing += 1
                    continue
                extras = cols[26]
                match = PAGE_TITLE.search(extras)
                if not match:
                    continue
                title = unescape(match.group(1)).strip()
                if not title:
                    continue
                titles += 1
                gkg_seen_raw = cols[1]
                seen = (datetime.strptime(gkg_seen_raw, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
                        .isoformat() if GKG_DATE.match(gkg_seen_raw) else '')
                pub = PUB_TIME.search(extras)
                pub_raw = pub.group(1) if pub else ''
                published = (datetime.strptime(pub_raw,'%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
                             .isoformat() if pub_raw else '')
                flags = news_flags(title)
                rows.append({
                    'archive_url': archive_url, 'gkg_record_id': cols[0],
                    'gkg_seen_utc': seen, 'publication_time_reported_utc': published,
                    'source_domain': cols[3], 'article_url': cols[4], 'title': title,
                    'gkg_tone_raw': cols[15].split(',')[0], **flags,
                })
    return rows, {'records_scanned': scanned, 'records_with_page_title': titles,
                  'malformed_short_rows': missing}


def get_latest_gkg_url(session):
    response = session.get(BASE + 'lastupdate.txt', timeout=40)
    response.raise_for_status()
    for line in response.text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2].lower().endswith('.gkg.csv.zip'):
            return parts[2].replace('http://data.gdeltproject.org/',
                                    'https://data.gdeltproject.org/')
    raise RuntimeError('No GKG archive found in lastupdate.txt')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--timestamps', default='20200115120000,20210115120000,20230115120000,20250115120000',
                    help='Comma-separated historical YYYYMMDDHHMMSS GKG slice times')
    ap.add_argument('--include-latest', action='store_true')
    ap.add_argument('--output-dir', type=Path, default=Path('results/source_audit/gdelt_archives'))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({'User-Agent': 'MSML610-research-GDELT-GKG-pilot/0.1'})
    urls = []
    for t in args.timestamps.split(','):
        t = t.strip()
        if not t:
            continue
        try:
            datetime.strptime(t, '%Y%m%d%H%M%S')
        except ValueError as e:
            ap.error(f'Bad timestamp {t}: {e}')
        urls.append(BASE + t + '.gkg.csv.zip')
    errors = []
    if args.include_latest:
        try:
            urls.append(get_latest_gkg_url(session))
        except Exception as ex:
            errors.append(f'Latest manifest: {type(ex).__name__}: {ex}')
    all_rows = []
    files = []
    for url in dict.fromkeys(urls):
        try:
            response = session.get(url, timeout=100)
            response.raise_for_status()
            rows, stats = parse_gkg_rows(response.content, url)
            all_rows.extend(rows)
            record = {'url': url, 'bytes_downloaded': len(response.content), **stats}
            files.append(record)
            print('[GDELT GKG]', url.rsplit('/',1)[-1], 'scanned', stats['records_scanned'],
                  'titles',stats['records_with_page_title'], 'matches',
                  sum(r['keep_candidate'] for r in rows), flush=True)
        except Exception as ex:
            err = f'{url}: {type(ex).__name__}: {str(ex)[:240]}'
            errors.append(err)
            print('[GDELT GKG] FAILED', err, flush=True)
    df = pd.DataFrame(all_rows)
    summary = {
        'source': 'GDELT 2.0 GKG 15-minute public ZIP archives',
        'retrieval_utc': datetime.now(timezone.utc).isoformat(),
        'requested_archive_count': len(urls),
        'successful_archives': len(files), 'files': files, 'errors': errors,
        'rows_with_titles_before_dedupe': len(df),
        'note': 'This is a convenience sample of a few 15-minute intervals, NOT complete days or full historical coverage. GKG ingestion time is not necessarily article publication time.',
    }
    if not df.empty:
        df = df.drop_duplicates(subset=['article_url'], keep='first')
        summary['unique_articles'] = len(df)
        summary['oil_matches'] = int(df['oil_candidate'].sum())
        summary['gas_matches'] = int(df['gas_candidate'].sum())
        summary['direct_matches'] = int((df['oil_candidate'] | df['gas_candidate']).sum())
        summary['broad_matches'] = int(df['keep_candidate'].sum())
        summary['publication_timestamp_present'] = int(df['publication_time_reported_utc'].ne('').sum())
        df.to_csv(args.output_dir/'gdelt_gkg_all_sampled_titles.csv', index=False)
        subset = df.loc[df['keep_candidate']].copy()
        subset.to_csv(args.output_dir/'gdelt_gkg_energy_candidates.csv', index=False)
        if not subset.empty:
            sample = subset.sample(n=min(80,len(subset)),random_state=42)
            sample['human_relevance_0_1_2'] = ''
            sample['human_notes'] = ''
            sample.to_csv(args.output_dir/'gdelt_gkg_manual_review.csv', index=False)
    (args.output_dir/'gdelt_gkg_summary.json').write_text(json.dumps(summary, indent=2))
    print('\n=== GDELT GKG ARCHIVE PILOT ===')
    print(json.dumps({k:v for k,v in summary.items() if k != 'files'}, indent=2))
    print('\nSaved:',args.output_dir)
    if not files:
        raise SystemExit('No archives downloaded successfully; see gdelt_gkg_summary.json')

if __name__ == '__main__':
    main()
