"""Artificial finite-range/CSV boundaries; no network, credentials or returns."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from historical_ranges import RangeArchive, parse_records, retrieve_source, verify_archive, REVISION

HEADER=['Date','Article_title','Stock_symbol','Url','Publisher','Author','Article','Lsa_summary','Luhn_summary','Textrank_summary','Lexrank_summary']
def dataset(symbols=('A','AMZN','MSFT','Z')):
    f=io.StringIO(newline='');w=csv.writer(f,lineterminator='\n');w.writerow(HEADER)
    for s in symbols:
        for i in range(30):w.writerow(['2018-01-02 13:00:00 UTC',f'{s} title {i} é',s,f'https://example.test/{s}/{i}','publisher','','body\nsecond line','','','',''])
    return f.getvalue().encode()

def registration():
    return {'symbols':['A','AMZN','MSFT'],'revision':REVISION,'source_date_start':'2016-01-01','source_date_end_exclusive':'2020-06-06',
            'budget':{'probe_bytes':512,'public_source_bytes':1000000,'max_public_range_requests':96,'max_company_span_bytes':100000}}

class Response:
    def __init__(self,raw,start,end,total,status=206):
        self.status_code=status;self.headers={'Content-Range':f'bytes {start}-{end-1}/{total}'};self.raw=raw
    def __enter__(self):return self
    def __exit__(self,*_):pass
    def iter_content(self,chunk_size):
        for offset in range(0,len(self.raw),chunk_size):yield self.raw[offset:offset+chunk_size]

class Session:
    def __init__(self,raw,status=206):self.raw=raw;self.calls=[];self.status=status
    def get(self,url,*,headers,**kwargs):
        self.calls.append((url,headers));a,b=map(int,headers['Range'][6:].split('-'));b+=1
        return Response(self.raw[a:b],a,b,len(self.raw),self.status)
    def close(self):pass

class RangeTests(unittest.TestCase):
    def test_unavailable_company_does_not_hide_other_registered_companies(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);raw=dataset(('AMZN','MSFT','Z'));reg=p/'reg.json';reg.write_text(json.dumps(registration()))
            m=retrieve_source(p/'s',reg,session=Session(raw),total=len(raw))
            self.assertEqual(m['status'],'complete_bounded_targeted_capture')
            self.assertEqual(m['companies']['A']['status'],'Unavailable')
            self.assertEqual(m['companies']['AMZN']['selected_headlines'],30)
            self.assertEqual(m['companies']['MSFT']['selected_headlines'],30)

    def test_indexed_source_records_preserve_actual_byte_offsets(self):
        raw=dataset();text=raw.decode().splitlines(keepends=True)
        # Add a leading CSV index only on record boundaries, not quoted continuations.
        indexed=b'';index=0
        for line in text:
            if line.startswith('Date,'):line='Unnamed: 0,'+line
            elif line.startswith('2018-'):
                line=str(index)+'.0,'+line;index+=1
            indexed+=line.encode()
        rows=parse_records(indexed,0,['Unnamed: 0']+HEADER,eof=True)
        self.assertEqual(len(rows),120)
        self.assertEqual(rows[0]['row']['Stock_symbol'],'A')
        self.assertTrue(indexed[rows[1]['offset']:].startswith(b'1.0,2018-'))

    def test_date_window_is_located_before_large_full_company_download(self):
        from datetime import date,timedelta
        f=io.StringIO(newline='');w=csv.writer(f,lineterminator='\n');w.writerow(HEADER)
        for symbol in ['A','AMZN','MSFT','Z']:
            for i in range(200):w.writerow([(date(2020,1,1)-timedelta(days=i)).isoformat()+' 13:00:00 UTC',f'title {i}',symbol,'https://example.test/','','','','','','',''])
        raw=f.getvalue().encode();reg=registration();reg.update(date_window_seek=True,source_date_start='2019-10-01',source_date_end_exclusive='2019-11-01')
        reg['budget']['max_company_span_bytes']=4000
        with tempfile.TemporaryDirectory() as d:
            a=RangeArchive(Path(d)/'s',reg,session=Session(raw),total=len(raw));rows=a.company('MSFT')
            self.assertEqual(len(rows),31)
            self.assertTrue(all('2019-10-01'<=r['row']['Date'][:10]<'2019-11-01' for r in rows))
            self.assertFalse(a.manifest['companies']['MSFT']['full_company_region_downloaded'])

    def test_missing_ticker_tail_is_excluded_and_can_bound_last_company(self):
        with tempfile.TemporaryDirectory() as d:
            raw=dataset(('A','AMZN','MSFT',''));a=RangeArchive(Path(d)/'s',registration(),session=Session(raw),total=len(raw))
            self.assertEqual(len(a.company('MSFT')),30)

    def test_failed_attempt_bytes_can_be_reused_with_original_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);raw=dataset();s=Session(raw)
            old=RangeArchive(p/'old',registration(),session=s,total=len(raw));old.records(0,512);old.flush()
            new=RangeArchive(p/'new',registration(),session=s,total=len(raw),resume_source=p/'old')
            new.records(0,512)
            self.assertEqual(len(s.calls),1);self.assertEqual(new.spent,512)
            self.assertEqual(new.manifest['requests'][0]['received_at'],old.manifest['requests'][0]['received_at'])

    def test_random_utf8_and_multiline_fragments_keep_only_whole_records(self):
        raw=dataset();a=raw.index('é'.encode())+1
        rows=parse_records(raw[a:-30],a,HEADER)
        self.assertTrue(rows)
        self.assertTrue(all(r['row']['Article']=='body\nsecond line' for r in rows))
        for r in rows:self.assertTrue(raw[r['offset']:r['end']].startswith(b'2018-01-02'))

    def test_ambiguous_date_line_inside_body_is_rejected(self):
        raw=b'2018-01-02 13:00:00 UTC,title,A,url,pub,,"body\n2018-01-03 13:00:00 UTC,inside",,,,\n2018-01-04 13:00:00 UTC,title,A,url,pub,,,,,,\n'
        with self.assertRaisesRegex(ValueError,'Ambiguous|schema'):parse_records(raw,0,HEADER,eof=True)

    def test_binary_boundaries_recover_every_selected_record(self):
        with tempfile.TemporaryDirectory() as d:
            raw=dataset();s=Session(raw);a=RangeArchive(Path(d)/'source',registration(),session=s,total=len(raw))
            for symbol in ['A','AMZN','MSFT']:
                rows=a.company(symbol);self.assertEqual(len(rows),30)
                self.assertTrue(all(r['row']['Stock_symbol']==symbol for r in rows))
                self.assertTrue(all(r['received_at'] for r in rows))
            self.assertLessEqual(a.spent,a.budget['public_source_bytes'])
            self.assertTrue(all(set(headers)=={'Range'} for _,headers in s.calls))

    def test_file_end_can_bound_last_company(self):
        with tempfile.TemporaryDirectory() as d:
            raw=dataset(('A','AMZN','MSFT'));a=RangeArchive(Path(d)/'s',registration(),session=Session(raw),total=len(raw))
            rows=a.company('MSFT');self.assertEqual(len(rows),30);self.assertEqual(rows[-1]['end'],len(raw))

    def test_absent_company_is_not_treated_as_zero_news(self):
        with tempfile.TemporaryDirectory() as d:
            raw=dataset(('A','MSFT','Z'));a=RangeArchive(Path(d)/'s',registration(),session=Session(raw),total=len(raw))
            with self.assertRaisesRegex(ValueError,'absent'):a.company('AMZN')

    def test_server_ignoring_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            raw=dataset();a=RangeArchive(Path(d)/'s',registration(),session=Session(raw,status=200),total=len(raw))
            with self.assertRaisesRegex(ValueError,'exact'):a.fetch(0,512)

    def test_byte_and_request_budgets_are_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            reg=registration();reg['budget']['public_source_bytes']=511
            raw=dataset();s=Session(raw);a=RangeArchive(Path(d)/'s',reg,session=s,total=len(raw))
            with self.assertRaisesRegex(ValueError,'budget'):a.fetch(0,512)
            self.assertEqual(s.calls,[])
        with tempfile.TemporaryDirectory() as d:
            reg=registration();reg['budget']['max_public_range_requests']=1
            raw=dataset();a=RangeArchive(Path(d)/'s',reg,session=Session(raw),total=len(raw));a.fetch(0,512)
            with self.assertRaisesRegex(ValueError,'budget'):a.fetch(512,1024)

    def test_wrong_response_offsets_are_rejected(self):
        class Wrong(Session):
            def get(self,*a,**k):
                r=super().get(*a,**k);r.headers['Content-Range']='bytes 0-1/2';return r
        with tempfile.TemporaryDirectory() as d:
            raw=dataset();a=RangeArchive(Path(d)/'s',registration(),session=Wrong(raw),total=len(raw))
            with self.assertRaisesRegex(ValueError,'exact'):a.fetch(0,512)

    def test_observed_order_contradiction_fails(self):
        with tempfile.TemporaryDirectory() as d:
            raw=dataset(('MSFT','A'));a=RangeArchive(Path(d)/'s',registration(),session=Session(raw),total=len(raw))
            with self.assertRaisesRegex(ValueError,'ordering'):a.records(0,len(raw))

    def test_full_retrieval_binds_raw_ranges_and_actual_offsets(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reg=root/'reg.json';reg.write_text(json.dumps(registration()));raw=dataset()
            m=retrieve_source(root/'s',reg,session=Session(raw),total=len(raw))
            self.assertEqual(m['status'],'complete_bounded_targeted_capture');verify_archive(root/'s')
            news=json.loads((root/'s/news_AMZN.json').read_text());self.assertEqual(len(news),30)
            self.assertTrue(all(raw[r['source_byte_offset']:].startswith(b'2018-01-02') for r in news))
            request=m['requests'][0];p=root/'s'/request['file'];p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'hash'):verify_archive(root/'s')

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'keep').write_text('keep')
            with self.assertRaises(FileExistsError):RangeArchive(p,registration(),session=Session(dataset()),total=100)
            self.assertEqual((p/'keep').read_text(),'keep')


if __name__=='__main__':unittest.main()
