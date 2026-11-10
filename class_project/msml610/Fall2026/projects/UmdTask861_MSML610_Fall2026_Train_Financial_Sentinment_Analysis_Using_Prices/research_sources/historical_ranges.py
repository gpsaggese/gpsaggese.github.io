"""Finite, auditable byte-range retrieval from a pinned ticker-ordered CSV.

Observed ordering and both neighboring ticker boundaries are required. This
does not certify global corpus ordering or historical text availability.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import re

from historical_backfill import REPOSITORY, REVISION, NEWS_FILE, PRICE_URL

START = re.compile(rb'(?m)^(?:\d+(?:\.0)?,)?\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC,')


class CoverageUnavailable(ValueError):
    """A declared company/date region was not found; retain the denominator."""


def stamp(): return datetime.now(timezone.utc).isoformat()
def sha(data): return hashlib.sha256(data).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, obj): Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def ticker_key(value): return value or '\uffff'  # observed missing-tag partition sorts last; never selected


def parse_records(raw, start, header, *, eof=False):
    """Parse only whole date-started CSV records; retain actual byte offsets.

    A date-like line inside a quoted field is ambiguous and fails closed rather
    than silently fabricating a boundary. UTF-8 fragments before the first full
    record and the incomplete suffix are excluded.
    """
    matches = list(START.finditer(raw))
    rows = []
    for index, match in enumerate(matches):
        end = matches[index+1].start() if index+1 < len(matches) else (len(raw) if eof else None)
        if end is None: break
        chunk = raw[match.start():end]
        try:
            parsed = list(csv.reader(io.StringIO(chunk.decode('utf-8')), strict=True))
        except (UnicodeError, csv.Error) as exc:
            raise ValueError('Ambiguous or malformed whole CSV record') from exc
        if len(parsed)!=1 or len(parsed[0])!=len(header):
            raise ValueError('Unexpected whole CSV record/schema')
        row = dict(zip(header, parsed[0]))
        if row['Stock_symbol'] and not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,14}', row['Stock_symbol']):
            raise ValueError('Invalid observed ticker')
        datetime.strptime(row['Date'], '%Y-%m-%d %H:%M:%S UTC')
        rows.append({'offset':start+match.start(), 'end':start+end, 'row':row})
    return rows


class RangeArchive:
    def __init__(self, output, registration, *, session, total, resume_source=None):
        self.output=Path(output);self.output.mkdir(mode=0o700,parents=True,exist_ok=False)
        (self.output/'.gitignore').write_text('*\n')
        self.registration=registration;self.budget=registration['budget'];self.session=session;self.total=total
        self.source_file=registration.get('source_file',NEWS_FILE)
        if self.source_file not in {NEWS_FILE,'Stock_news/nasdaq_exteral_data.csv'}:raise ValueError('Unknown pinned source file')
        self.probe=self.budget['probe_bytes'];self.spent=0;self.cache={};self.observed={};self.header=None
        self.manifest={'status':'running','started_at':stamp(),'source':REPOSITORY,'revision':REVISION,
                       'file':self.source_file,'total_size':total,'registration':registration,'requests':[],
                       'companies':{},'historical_availability_verified':False,
                       'generation_source_sha256':sha(Path(__file__).read_bytes()),
                       'coverage_scope':'Observed ticker-ordered ranges with local boundaries; global ordering not independently certified'}
        if resume_source is not None:
            prior=Path(resume_source);old=read(prior/'manifest.json')
            if old['revision']!=REVISION or old['total_size']!=total or old['file']!=self.source_file:raise ValueError('Different resume source')
            for req in old['requests']:
                if req.get('status')!=206 or not req.get('sha256'):raise ValueError('Cannot reuse incomplete response')
                raw=(prior/req['file']).read_bytes()
                if sha(raw)!=req['sha256'] or len(raw)!=req['end_exclusive']-req['start']:raise ValueError('Changed resume bytes')
                (self.output/req['file']).write_bytes(raw)
                self.manifest['requests'].append(dict(req))
                self.cache[req['start'],req['end_exclusive']]=raw;self.spent+=len(raw)
            self.manifest['reused_from_manifest_sha256']=sha((prior/'manifest.json').read_bytes())
        self.flush()

    def flush(self):
        self.manifest['retained_public_bytes']=self.spent
        self.manifest['updated_at']=stamp()
        save(self.output/'manifest.json',self.manifest)

    def fetch(self,start,end):
        start=max(0,start);end=min(self.total,end)
        if not 0<=start<end<=self.total:raise ValueError('Invalid byte interval')
        key=(start,end)
        if key in self.cache:return self.cache[key]
        for (a,b),raw in self.cache.items():
            if a<=start and end<=b:return raw[start-a:end-a]
        if len(self.manifest['requests'])>=self.budget['max_public_range_requests'] or self.spent+end-start>self.budget['public_source_bytes']:
            raise ValueError('Registered public retrieval budget exhausted')
        request={'start':start,'end_exclusive':end,'started_at':stamp(),'file':f'range_{len(self.manifest["requests"]):03d}.csv'}
        self.manifest['requests'].append(request);self.flush()
        url=f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{self.source_file}'
        with self.session.get(url,headers={'Range':f'bytes={start}-{end-1}'},stream=True,timeout=30) as response:
            request.update(status=response.status_code,content_range=response.headers.get('Content-Range'))
            expected=f'bytes {start}-{end-1}/{self.total}'
            if response.status_code!=206 or request['content_range']!=expected:
                self.flush();raise ValueError('Server did not return the exact registered byte range')
            parts=[];retained=0
            for part in response.iter_content(chunk_size=65536):
                if retained+len(part)>end-start:
                    self.flush();raise ValueError('Oversized range response')
                parts.append(part);retained+=len(part)
            raw=b''.join(parts)
        self.spent+=len(raw)
        (self.output/request['file']).write_bytes(raw)
        request.update(received_at=stamp(),bytes=len(raw),sha256=sha(raw));self.flush()
        if len(raw)!=end-start:raise ValueError('Incomplete range response')
        self.cache[key]=raw
        return raw

    def records(self,start,end):
        raw=self.fetch(start,end)
        if start==0 and self.header is None:
            self.header=next(csv.reader(io.StringIO(raw.decode('utf-8',errors='replace'))))
            if not {'Date','Article_title','Stock_symbol','Url'}.issubset(self.header):
                raise ValueError('Unexpected source header')
        rows=parse_records(raw,start,self.header,eof=end==self.total)
        if not rows:raise ValueError('No complete records within the registered probe')
        receipt=next(r for r in reversed(self.manifest['requests']) if r['start']<=start and end<=r['end_exclusive'] and 'received_at' in r)
        for r in rows:
            r['received_at']=receipt['received_at']
            symbol=r['row']['Stock_symbol']
            if r['offset'] in self.observed and self.observed[r['offset']]!=symbol:
                raise ValueError('Contradictory observed ticker at byte offset')
            self.observed[r['offset']]=symbol
        ordered=[ticker_key(v) for _,v in sorted(self.observed.items())]
        if ordered!=sorted(ordered):raise ValueError('Observed ticker ordering contradicts range search')
        return rows

    def boundary(self,symbol,*,upper=False):
        """Locate first complete record >= symbol (or > symbol), no return data."""
        def after(value):return ticker_key(value)>symbol if upper else ticker_key(value)>=symbol
        lo,hi=0,self.total
        first=self.records(0,min(self.total,self.probe))
        if after(first[0]['row']['Stock_symbol']):return first[0]['offset']
        iterations=0
        while hi-lo>self.probe:
            prior=(lo,hi);iterations+=1
            if iterations>32:raise ValueError('Finite boundary-search probe limit exhausted')
            mid=(lo+hi)//2
            rows=self.records(mid,min(self.total,mid+self.probe))
            if after(rows[0]['row']['Stock_symbol']):
                hi=min(hi,rows[0]['offset'])
            elif not after(rows[-1]['row']['Stock_symbol']):
                lo=max(lo,rows[-1]['end'])
            else:
                return next(r['offset'] for r in rows if after(r['row']['Stock_symbol']))
            if (lo,hi)==prior:break
        rows=self.records(lo,min(self.total,hi+2*self.probe))
        candidates=[r['offset'] for r in rows if after(r['row']['Stock_symbol'])]
        if candidates:return candidates[0]
        if rows[-1]['end']==self.total:return self.total
        raise ValueError('Could not certify local company boundary')

    def company(self,symbol):
        left,right=self.boundary(symbol),self.boundary(symbol,upper=True)
        if right<=left:raise CoverageUnavailable('Declared company absent from observed sorted region')
        company_left,company_right=left,right
        dated=self.registration.get('date_window_seek',False)
        if dated:
            left=self.date_boundary(symbol,company_left,company_right,self.registration['source_date_end_exclusive'])
            right=self.date_boundary(symbol,company_left,company_right,self.registration['source_date_start'])
            if right<=left:raise CoverageUnavailable('Declared date window absent from observed company region')
        if right-left>self.budget['max_company_span_bytes']:
            raise ValueError('Company region exceeds registered span budget')
        # Include the first complete following record to certify the final one.
        rows=self.records(left,min(self.total,right+self.probe))
        selected=[r for r in rows if left<=r['offset']<right]
        if not selected or selected[0]['offset']!=left or selected[-1]['end']!=right or any(r['row']['Stock_symbol']!=symbol for r in selected):
            raise ValueError('Company span has gaps or mixed tickers')
        if right<self.total and not any(r['offset']==right and (ticker_key(r['row']['Stock_symbol'])>symbol or
                (dated and r['row']['Stock_symbol']==symbol and r['row']['Date'][:10]<self.registration['source_date_start'])) for r in rows):
            raise ValueError('Missing following ticker boundary')
        summary={'start_byte':left,'end_byte_exclusive':right,'whole_company_records_in_observed_contiguous_region':len(selected),
                 'local_following_boundary_verified':True,'global_order_certified':False}
        if dated:
            summary['whole_records_in_registered_date_region']=summary.pop('whole_company_records_in_observed_contiguous_region')
            summary.update(company_start_byte=company_left,company_end_byte_exclusive=company_right,
                           full_company_region_downloaded=left==company_left and right==company_right,
                           date_order_scope='Descending dates verified only on observed same-company probes and retained window')
        self.manifest['companies'][symbol]=summary;self.flush()
        return selected

    def date_boundary(self,symbol,start,end,threshold):
        """First same-company record older than a fixed date in descending dates."""
        dates={}
        def get(a,b):
            rows=self.records(a,min(self.total,b))
            for r in rows:
                if r['row']['Stock_symbol']==symbol and start<=r['offset']<end:
                    dates[r['offset']]=r['row']['Date'][:10]
            observed=[d for _,d in sorted(dates.items())]
            if observed!=sorted(observed,reverse=True):raise ValueError('Observed company date order contradicts date-window search')
            return rows
        def older(r):
            if r['offset']>=end:return True
            if r['row']['Stock_symbol']!=symbol:raise ValueError('Unexpected company inside dated region')
            return r['row']['Date'][:10]<threshold
        lo,hi=start,end
        first=get(start,start+self.probe)
        if older(first[0]):return start
        for _ in range(32):
            if hi-lo<=self.probe:break
            prior=(lo,hi);mid=(lo+hi)//2;rows=get(mid,mid+self.probe)
            if older(rows[0]):hi=min(hi,rows[0]['offset'])
            elif not older(rows[-1]):lo=max(lo,rows[-1]['end'])
            else:return next(r['offset'] for r in rows if older(r))
            if (lo,hi)==prior:break
        rows=get(lo,hi+2*self.probe)
        candidates=[r['offset'] for r in rows if older(r)]
        if candidates:return min(end,candidates[0])
        if end==self.total:return end
        raise ValueError('Could not certify registered date boundary')


def verify_archive(source):
    source=Path(source);m=read(source/'manifest.json')
    if m['status']!='complete_bounded_targeted_capture':raise ValueError('Source range archive incomplete')
    for request in m['requests']:
        if Path(request['file']).name!=request['file'] or sha((source/request['file']).read_bytes())!=request['sha256']:
            raise ValueError('Source range hash mismatch')
    return m


def retrieve_source(output,registration,*,session=None,total=None,resume_source=None):
    import requests
    registration_bytes=Path(registration).read_bytes();reg=json.loads(registration_bytes)
    registration_hash=sha(registration_bytes)
    total=total if total is not None else reg.get('source_total_bytes',5731397037)
    if reg['revision']!=REVISION or reg['symbols']!=['A','AMZN','MSFT']:
        raise ValueError('Use the fixed registered source/company pool')
    owned=session is None;session=session or requests.Session()
    archive=RangeArchive(output,reg,session=session,total=total,resume_source=resume_source)
    try:
        for symbol in reg['symbols']:
            try:
                records=archive.company(symbol)
            except CoverageUnavailable as exc:
                archive.manifest['companies'][symbol]={'status':'Unavailable','reason':str(exc),'selected_headlines':0,
                    'global_order_certified':False,'not_evidence_of_no_company_news':True}
                save(archive.output/f'news_{symbol}.json',[]);archive.flush()
                print(json.dumps({'symbol':symbol,**archive.manifest['companies'][symbol]}),flush=True)
                continue
            selected=[]
            for item in records:
                row=item['row'];day=row['Date'][:10]
                if reg['source_date_start']<=day<reg['source_date_end_exclusive'] and row['Article_title'].strip():
                    selected.append({'headline':row['Article_title'],'source_date':day,'reported_timestamp':row['Date'],
                                     'symbol':symbol,'url':row['Url'],'publisher':row.get('Publisher'),
                                     'source_row_index':item['offset'],'source_position_kind':'actual_csv_byte_offset_not_row_ordinal',
                                     'source_byte_offset':item['offset'],'response_received_at':item['received_at'],
                                     'historical_availability_verified':False})
            save(archive.output/f'news_{symbol}.json',selected)
            archive.manifest['companies'][symbol].update(selected_headlines=len(selected),
                source_day_min=min((r['source_date'] for r in selected),default=None),
                source_day_max=max((r['source_date'] for r in selected),default=None))
            print(json.dumps({'symbol':symbol,**archive.manifest['companies'][symbol],'retained_public_bytes':archive.spent}),flush=True)
        if sha(Path(registration).read_bytes())!=registration_hash:raise ValueError('Registration changed during source retrieval')
        if sha(Path(__file__).read_bytes())!=archive.manifest['generation_source_sha256']:raise ValueError('Retrieval source changed during capture')
        archive.manifest.update(status='complete_bounded_targeted_capture',finished_at=stamp(),
            derived_news_sha256={s:sha((archive.output/f'news_{s}.json').read_bytes()) for s in reg['symbols']},
            registration_sha256=registration_hash)
    except Exception as exc:
        archive.manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc) if isinstance(exc,ValueError) else 'Source retrieval failed')
        raise
    finally:
        archive.flush()
        if owned:session.close()
    return archive.manifest


def capture_company(source,symbol,output,*,session=None,credential_loader=None):
    import requests
    from collect_research import _local_credentials
    from calendar_audit import export_calendar
    source=Path(source);manifest=verify_archive(source);reg=manifest['registration']
    if symbol not in reg['symbols']:raise ValueError('Unregistered company')
    news=read(source/f'news_{symbol}.json')
    if sha((source/f'news_{symbol}.json').read_bytes())!=manifest['derived_news_sha256'][symbol] or not news:
        raise ValueError('Missing or changed derived headlines')
    out=Path(output)
    if out.exists() or out.is_symlink():raise FileExistsError('Choose a new company archive')
    out.mkdir(mode=0o700,parents=True);(out/'.gitignore').write_text('*\n')
    protocol={k:reg[k] for k in ['source_date_start','source_date_end_exclusive','test_start','entry_delay_sessions','horizon_sessions','class_neutral_band','ridge_alpha','tfidf','assumed_roundtrip_cost_bps']}
    protocol.update(symbol=symbol,revision=REVISION,source=REPOSITORY,
        registered_at=reg['registered_at'],target=reg['target'],
        limitations=['Final historical text versions/availability not independently verified',
                     'Observed ticker-ordered byte regions do not certify full corpus or global ticker order',
                     'Vendor tags are not independent relevance annotations; development study, not alpha evidence'])
    save(out/'protocol.json',protocol);save(out/'news.json',news)
    first=date.fromisoformat(reg['source_date_start'])-timedelta(days=61)
    last=date.fromisoformat(reg['source_date_end_exclusive'])+timedelta(days=45)
    m={'status':'running','started_at':stamp(),'requests':[],'data_kind':'observed_retrospective',
       'historical_availability_verified':False,'protocol_sha256':sha((out/'protocol.json').read_bytes()),
       'source_archive':str(source.resolve()),'source_manifest_sha256':sha((source/'manifest.json').read_bytes()),
       'coverage_scope':manifest['coverage_scope'],'selected_headlines':len(news)}
    owned=session is None;session=session or requests.Session()
    try:
        key,secret=(credential_loader or _local_credentials)()
        params={'symbols':f'{symbol},SPY','start':first.isoformat()+'T00:00:00Z','end':last.isoformat()+'T00:00:00Z',
                'timeframe':'1Day','adjustment':'split','feed':'sip','limit':10000,'sort':'asc'}
        tokens=set();counts={}
        for page in range(reg['budget']['max_company_price_pages']):
            request={'url':PRICE_URL,'params':dict(params),'file':f'prices_{page:03}.json','started_at':stamp()};m['requests'].append(request)
            response=session.get(PRICE_URL,params=params,headers={'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret},timeout=30,allow_redirects=False)
            raw=response.content;(out/request['file']).write_bytes(raw)
            request.update(status=response.status_code,response_received_at=stamp(),sha256=sha(raw),bytes=len(raw))
            if response.status_code!=200:raise ValueError('Existing Alpaca read-only access rejected bars')
            payload=json.loads(raw)
            if not isinstance(payload.get('bars'),dict) or any(not isinstance(v,list) for v in payload['bars'].values()):
                raise ValueError('Invalid observed bars payload')
            for ticker,bars in payload['bars'].items():counts[ticker]=counts.get(ticker,0)+len(bars)
            token=payload.get('next_page_token');request['next_page_token']=token
            if token is not None and not isinstance(token,str):raise ValueError('Invalid pagination token')
            if not token:break
            if token in tokens:raise ValueError('Repeated price pagination token')
            tokens.add(token);params['page_token']=token
        else:raise ValueError('Registered price-page budget exhausted')
        if set(counts)!={symbol,'SPY'} or any(not n for n in counts.values()):raise ValueError('Missing/unexpected required bars')
        m['bars_summary']={'pagination_complete':True,'records':counts,'missing_symbols':[]}
        save(out/'calendar.json',export_calendar(first.isoformat(),last.isoformat()))
        verify_archive(source)
        if sha((source/'manifest.json').read_bytes())!=m['source_manifest_sha256']:raise ValueError('Source changed during price capture')
        m.update(status='complete_bounded_capture',finished_at=stamp())
    except Exception as exc:
        m.update(status='failed',error_type=type(exc).__name__,error=str(exc) if isinstance(exc,ValueError) else 'Price capture failed');raise
    finally:
        m['files_sha256']={p.name:sha(p.read_bytes()) for p in out.iterdir() if p.is_file() and p.name not in {'.gitignore','manifest.json'}}
        save(out/'manifest.json',m)
        if owned:session.close()
    return m


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    source=sub.add_parser('source');source.add_argument('--registration',type=Path,required=True);source.add_argument('--output-dir',type=Path,required=True);source.add_argument('--resume-source',type=Path)
    company=sub.add_parser('company');company.add_argument('--source',type=Path,required=True);company.add_argument('--symbol',required=True);company.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv)
    if a.action=='source':r=retrieve_source(a.output_dir,a.registration,resume_source=a.resume_source)
    else:r=capture_company(a.source,a.symbol,a.output_dir)
    print(json.dumps({'status':r['status']},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
