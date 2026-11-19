"""Bounded read-only corporate-action capture and offline receipt/date audit.

Raw successful HTTP bytes are retained. Pagination completion covers only the
specified process-date query, not ex-date completeness or total-return evidence.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

if __package__:
    from .collect_research import _local_credentials, _symbols
else:
    from collect_research import _local_credentials, _symbols

URL = 'https://data.alpaca.markets/v1/corporate-actions'


def _now():return datetime.now(timezone.utc)
def _iso(value):
    if value.tzinfo is None or value.utcoffset() is None:raise ValueError('Aware clock required')
    return value.astimezone(timezone.utc).isoformat()
def _time(value):
    value=datetime.fromisoformat(value.replace('Z','+00:00'))
    if value.tzinfo is None:raise ValueError('Aware receipt required')
    return value.astimezone(timezone.utc)
def _day(value):
    if not isinstance(value,str) or date.fromisoformat(value).isoformat()!=value:raise ValueError('Explicit ISO date required')
    return value

def _sha(data):return hashlib.sha256(data).hexdigest()
def _save(path,data):
    with Path(path).open('xb') as stream:stream.write(data)
def _json(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
def _payload(raw):
    value=json.loads(raw)
    if not isinstance(value,dict) or not isinstance(value.get('corporate_actions'),dict):raise ValueError('Corporate-action object missing')
    if 'next_page_token' not in value or (value['next_page_token'] is not None and not isinstance(value['next_page_token'],str)):
        raise ValueError('Explicit pagination token required')
    if value['next_page_token']=='':raise ValueError('Empty pagination token')
    for family,records in value['corporate_actions'].items():
        if not isinstance(family,str) or not isinstance(records,list) or any(not isinstance(r,dict) for r in records):
            raise ValueError('Invalid corporate-action category')
    return value


def capture(output,*,symbols,start,end,max_pages=3,limit=1000,data_quality='complete',session=None,credential_loader=_local_credentials,clock=_now):
    symbols=_symbols(symbols)
    if _day(start)>_day(end) or type(max_pages) is not int or max_pages<1 or type(limit) is not int or not 1<=limit<=1000:
        raise ValueError('Ordered process dates, positive page budget and limit 1..1000 required')
    if data_quality not in ['complete','all']:raise ValueError('Explicit data_quality required')
    output=Path(output);output.mkdir(parents=True,mode=0o700,exist_ok=False)
    _save(output/'.gitignore',b'*\n')
    manifest={'schema_version':'corporate-action-capture-1.0','url':URL,'date_basis':'process_date',
              'status':'running','pagination_complete':False,'created_at':_iso(clock()),
              'query':{'symbols':','.join(symbols),'start':start,'end':end,'limit':limit,'sort':'asc','data_quality':data_quality},
              'page_budget':max_pages,'requests':[],
              'generator_sha256':_sha(Path(__file__).read_bytes())}
    def save_manifest():(output/'manifest.json').write_bytes(_json(manifest))
    save_manifest()
    try:
        key,secret=credential_loader()
        if session is None:
            import requests
            session=requests.Session()
        # Only request headers receive credentials; no headers are persisted.
        auth={'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret}
        tokens=set();token=None
        for index in range(max_pages):
            params=dict(manifest['query'])
            if token is not None:params['page_token']=token
            record={'params':params,'file':f'page_{index:03d}.json','request_started_at':_iso(clock()),
                    'response_received_at':None,'http_status':None}
            manifest['requests'].append(record);save_manifest()
            response=session.get(URL,params=params,headers=auth,timeout=25,allow_redirects=False)
            raw=response.content  # Receipt is sampled after reading all bytes.
            record.update(http_status=response.status_code,response_received_at=_iso(clock()))
            if response.status_code!=200:raise ValueError('HTTP status '+str(response.status_code))
            _save(output/record['file'],raw)
            record.update(response_sha256=_sha(raw),response_bytes=len(raw),
                          response_headers={k:v for k,v in response.headers.items() if k.lower() in ['date','content-type','x-request-id']})
            payload=_payload(raw);token=payload['next_page_token'];record['next_page_token']=token
            save_manifest()
            if token is None:
                manifest.update(status='complete_under_query',pagination_complete=True);break
            if token in tokens:raise ValueError('Repeated pagination token')
            tokens.add(token)
        else:manifest['status']='truncated_by_page_budget'
    except Exception as error:
        # Persist only exception type / safe local validation, never requests'
        # potentially credential-bearing exception representations.
        manifest.update(status='failed',error_type=type(error).__name__)
    finally:
        manifest['finished_at']=_iso(clock());save_manifest()
    return manifest


def audit_archive(directory,*,entry_session,exit_session):
    entry,exit_=_day(entry_session),_day(exit_session)
    if entry>=exit_:raise ValueError('Increasing decision interval required')
    directory=Path(directory).resolve();mp=directory/'manifest.json';mb=mp.read_bytes();manifest=json.loads(mb)
    if manifest.get('url')!=URL or manifest.get('date_basis')!='process_date':raise ValueError('Fixed endpoint and process-date evidence required')
    # Preserve the first successful instrumented probe without re-requesting it.
    if 'requests' in manifest:
        query=manifest['query'];requests=manifest['requests']
    else:
        query=manifest['params'];requests=[{**manifest,'file':'response.json'}]
    symbols=query.get('symbols','').split(',');_symbols(symbols)
    if _day(query['start'])>_day(query['end']) or query.get('sort')!='asc':raise ValueError('Invalid query')
    fingerprints={str(mp):_sha(mb)};records=[];next_token=None;seen_tokens=set();previous_received=None
    for i,request in enumerate(requests):
        name=request.get('file')
        if not isinstance(name,str) or Path(name).name!=name or (directory/name).is_symlink():raise ValueError('Unsafe response path')
        params=request['params']
        if {k:v for k,v in params.items() if k!='page_token'}!=query or params.get('page_token')!=next_token:
            raise ValueError('Query changed or pagination chain broken')
        started,received=_time(request['request_started_at']),_time(request['response_received_at'])
        if (request.get('http_status')!=200 or received<started or (previous_received and started<previous_received)):
            raise ValueError('Failed or inconsistent HTTP receipt')
        previous_received=received
        path=directory/name;raw=path.read_bytes()
        if _sha(raw)!=request.get('response_sha256') or len(raw)!=request.get('response_bytes'):
            raise ValueError('Raw response hash/length mismatch')
        fingerprints[str(path)]=_sha(raw);payload=_payload(raw)
        for family,rows in payload['corporate_actions'].items():
            for row in rows:
                if row.get('symbol') not in symbols or not isinstance(row.get('id'),str) or not row['id']:
                    raise ValueError('Unexpected symbol or missing provider ID')
                process=_day(row['process_date']) if row.get('process_date') is not None else None
                if process is not None and not query['start']<=process<=query['end']:raise ValueError('Process date outside requested interval')
                records.append({'family':family,'record':row,'response_file':name,'response_sha256':_sha(raw),
                                'actual_receipt_at':request['response_received_at'],
                                'version_sha256':_sha(json.dumps(row,sort_keys=True).encode())})
        next_token=payload['next_page_token']
        if next_token is None and i!=len(requests)-1:raise ValueError('Pages follow terminal response')
        if next_token is not None:
            if next_token in seen_tokens:raise ValueError('Repeated token')
            seen_tokens.add(next_token)
    complete=bool(requests) and next_token is None
    unique={};duplicates=[]
    for item in records:
        key=(item['family'],item['record']['id'])
        if key in unique:
            if item['version_sha256']!=unique[key]['version_sha256']:raise ValueError('Conflicting captured event versions')
            duplicates.append(item['record']['id'])
        else:unique[key]=item
    cash=[];other=[]
    for item in unique.values():
        row=item['record']
        if item['family']!='cash_dividends':other.append(item);continue
        ex=_day(row['ex_date']) if row.get('ex_date') is not None else None
        pay=_day(row.get('payable_date')) if row.get('payable_date') is not None else None
        record_day=_day(row.get('record_date')) if row.get('record_date') is not None else None
        amount=Decimal(str(row['rate'])) if row.get('rate') is not None else None
        if amount is not None and (not amount.is_finite() or amount<=0):raise ValueError('Invalid dividend rate')
        cash.append({**item,'event_id':row['id'],'symbol':row['symbol'],'ex_date':ex,'payment_date':pay,
                     'record_date':record_day,'process_date':row['process_date'],'cash_per_share':str(amount) if amount is not None else None,
                     'currency':row.get('currency'),
                     'in_ex_date_holding_interval':ex is not None and entry<ex<=exit_,
                     'in_payment_date_interval':pay is not None and entry<pay<=exit_,
                     'cash_contract_blockers':['Currency and ordinary-entitlement treatment require independent review'] +
                                              (['ex_date_missing'] if ex is None else []) + (['rate_missing'] if amount is None else []),
                     'historical_availability_verified':False})
    if any(_sha(Path(p).read_bytes())!=h for p,h in fingerprints.items()):raise ValueError('Input changed during audit')
    return {'schema_version':'corporate-action-archive-audit-1.0','input_sha256':fingerprints,'query':query,
            'capture_status':manifest.get('status','instrumented HTTP probe'),'date_basis':'process_date','effective_data_quality':query.get('data_quality','complete (vendor default)'),
            'pagination_complete_under_query':complete,'raw_rows':len(records),'unique_events':len(unique),
            'duplicate_capture_ids':duplicates,'cash_dividends':cash,'other_actions':other,
            'entry_session':entry,'exit_session':exit_,
            'selected_ex_date_event_ids':[r['event_id'] for r in cash if r['in_ex_date_holding_interval']],
            'ex_date_coverage_verified':False,'total_return_series_verified':False,
            'labels_created':0,'eligible_labels':0,'model_evaluation_status':'Not evaluated',
            'limitations':['Pagination completion concerns only this vendor process-date query.',
                           'Default complete quality can omit unprocessed incomplete records; all quality also cannot prove market-wide completeness.',
                           'The API does not guarantee immediate creation/availability; current receipt cannot establish historical availability.',
                           'No total-return index, reinvestment convention, currency assumption or extra dividend addition is supplied.']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--capture',action='store_true');source.add_argument('--archive',type=Path)
    parser.add_argument('--symbols',nargs='+',default=['PLTR','AMZN','MSFT','SPY'])
    parser.add_argument('--start');parser.add_argument('--end');parser.add_argument('--max-pages',type=int,default=3)
    parser.add_argument('--limit',type=int,default=1000);parser.add_argument('--data-quality',choices=['complete','all'],default='complete')
    parser.add_argument('--entry-session',default='2026-09-01');parser.add_argument('--exit-session',default='2026-09-30')
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args(argv)
    output=args.output_dir.resolve()
    if output.exists():raise FileExistsError('Output must be new')
    if args.capture:
        result=capture(output,symbols=args.symbols,start=args.start,end=args.end,max_pages=args.max_pages,limit=args.limit,data_quality=args.data_quality)
        print(json.dumps({'status':result['status'],'pages':len(result['requests'])}))
        return 0 if result['status']=='complete_under_query' else 2
    archive=args.archive.resolve()
    if output.is_relative_to(archive) or archive.is_relative_to(output):raise ValueError('Separate audit output required')
    report=audit_archive(archive,entry_session=args.entry_session,exit_session=args.exit_session)
    output.mkdir(parents=True,mode=0o700,exist_ok=False);_save(output/'.gitignore',b'*\n');_save(output/'report.json',_json(report))
    _save(output/'generation_source.py',Path(__file__).read_bytes())
    print(json.dumps({k:report[k] for k in ['pagination_complete_under_query','unique_events','selected_ex_date_event_ids','eligible_labels']}))
    return 0


if __name__=='__main__':raise SystemExit(main())
