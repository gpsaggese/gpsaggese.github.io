"""Freeze real retained text inputs and report why price-model samples are unready.

This is not a label constructor, prediction, training run or event-identity
certification. Publication, revision and actual receipt remain separate clocks.
"""
from __future__ import annotations

import argparse
from datetime import date,datetime,time,timezone
import hashlib,json
from pathlib import Path
from zoneinfo import ZoneInfo

if __package__:
    from .run_research import read_inputs,timestamp
    from .price_labels import _calendar
else:
    from run_research import read_inputs,timestamp
    from price_labels import _calendar

NY=ZoneInfo('America/New_York')

def _sha(data):return hashlib.sha256(data).hexdigest()
def _iso(clock):return clock.astimezone(timezone.utc).isoformat().replace('+00:00','Z')
def _json(value):return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


def prepare_candidates(articles,calendar,*,symbols,decision_session,as_of):
    """Build dated text-only candidates without weakening existing label gates.

    ``articles`` must have been validated against captured raw responses by
    read_inputs. Direct callers are responsible for that binding; a boolean in
    this report is not independent authentication. No event ID is invented.
    """
    clock=timestamp(as_of);day=date.fromisoformat(decision_session)
    if day.isoformat()!=decision_session:raise ValueError('ISO decision_session required')
    if not symbols or len(set(symbols))!=len(symbols) or any(not isinstance(s,str) or not s.strip() for s in symbols):
        raise ValueError('Distinct explicit company symbols required')
    sessions,positions=_calendar(calendar)
    if decision_session not in positions:raise ValueError('Decision is not an explicit calendar session')
    position=positions[decision_session]
    cutoff=datetime.combine(day,time(8,45),NY).astimezone(timezone.utc)
    deadline=datetime.combine(day,time(9,15),NY).astimezone(timezone.utc)
    prospective=clock<=cutoff
    seen=set();retained=[];excluded=[]
    for original in articles:
        identity=str(original['id'])
        if identity in seen:raise ValueError('Duplicate input article ID')
        seen.add(identity)
        published,updated,receipt=[timestamp(original[k]) for k in ['created_at','updated_at','retrieved_at']]
        reasons=[]
        if updated<published or receipt<updated:reasons.append('inconsistent_source_times')
        for name,value in [('published_at',published),('version_updated_at',updated),('actual_receipt_at',receipt)]:
            if value>clock:reasons.append(name+'_after_as_of')
            if value>cutoff:reasons.append(name+'_after_input_cutoff')
        if not isinstance(original.get('headline'),str) or not isinstance(original.get('text'),str):
            raise ValueError('Retained headline/body strings required')
        if not original['headline'].strip() and not original['text'].strip():reasons.append('retained_text_empty')
        if reasons:
            excluded.append({'article_id':identity,'reasons':reasons});continue
        text=original['headline']+'\n'+original['text']
        retained.append({'article_id':identity,'source_id':'alpaca-news:'+identity,'symbols':list(original['symbols']),
                         'published_at':_iso(published),'version_updated_at':_iso(updated),'actual_receipt_at':_iso(receipt),
                         'headline':original['headline'],'body':original['text'],'retained_text_sha256':_sha(text.encode()),
                         'body_present':bool(original['text'].strip()),'url':original['url']})
    retained.sort(key=lambda x:(x['published_at'],x['article_id']))
    rows=[]
    for symbol in symbols:
        texts=[row for row in retained if symbol in row['symbols']]
        horizons={}
        for horizon in [5,10,20]:
            exit_=sessions[position+horizon] if position+horizon<len(sessions) else None
            horizons[str(horizon)]={'entry_at':_iso(sessions[position]['open_at']),
                                    'exit_session':exit_['date'] if exit_ else None,
                                    'exit_at':_iso(exit_['open_at']) if exit_ else None,
                                    'calendar_horizon_mature_as_of':clock>=exit_['open_at'] if exit_ else None,
                                    'label_status':'Not constructed','return_value':None}
        blockers=['reviewed_event_identity_missing','verified_open_total_return_series_missing',
                  'ex_date_action_coverage_not_certified','trained_model_and_frozen_prediction_missing']
        if calendar['complete'] is not True:blockers.append('calendar_not_complete')
        if not prospective:blockers.append('input_freeze_after_decision_cutoff')
        if not texts:blockers.append('no_retained_company_text')
        if any(h['exit_at'] is None for h in horizons.values()):blockers.append('calendar_tail_missing')
        rows.append({'sample_id':symbol+':'+decision_session,'symbol':symbol,'decision_session':decision_session,
                     'input_cutoff':_iso(cutoff),'prediction_deadline':_iso(deadline),'input_snapshot_created_at':_iso(clock),
                     'input_frozen_before_cutoff':prospective,'source_ids':[r['source_id'] for r in texts],
                     'article_company_pairs':len(texts),'body_article_company_pairs':sum(r['body_present'] for r in texts),
                     'source_event_ids':None,'event_identity_status':'Not independently reviewed',
                     'texts':texts,'horizons':horizons,'candidate_status':'Text inputs retained; not an eligible model sample',
                     'eligible_for_training':False,'prediction':None,'prediction_completed_at':None,'blockers':blockers})
    return {'schema_version':'financial-real-text-candidates-1.0','data_kind':'observed_retained_text_candidates',
            'as_of':_iso(clock),'decision_session':decision_session,'symbols':symbols,
            'text_policy':'Every retained matching company-tagged headline plus full retained body; no scores, outcomes or neutral imputation',
            'calendar_id':calendar['calendar_id'],'calendar_version':calendar['calendar_version'],'calendar_complete':calendar['complete'],
            'summary':{'input_unique_articles':len(articles),'time_qualified_unique_articles':len(retained),
                       'article_company_pairs':sum(r['article_company_pairs'] for r in rows),
                       'candidate_company_days_with_text':sum(bool(r['texts']) for r in rows),
                       'eligible_training_samples':0,'labels_created':0,'baseline_status':'Not run: no eligible labeled samples'},
            'excluded_articles':excluded,'candidates':rows,
            'limitations':['Company tags are not independent relevance or event-identity review.',
                           'This frozen input snapshot is not a frozen trained model or prediction.',
                           'A rule-calendar exit time is not an available return label.',
                           'No historical receipt or prediction-completion time is backfilled.',
                           'Shared source IDs are retained across companies for later source/event isolation.']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True);parser.add_argument('--calendar',type=Path,required=True)
    parser.add_argument('--decision-session',required=True);parser.add_argument('--as-of',required=True)
    parser.add_argument('--symbols',nargs='+',default=['PLTR','AMZN','MSFT']);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv);output=args.output_dir.resolve();archive=args.input_dir.resolve();cp=args.calendar.resolve()
    if output.exists():raise FileExistsError('A new output directory is required')
    if output.is_relative_to(archive) or archive.is_relative_to(output) or cp.is_relative_to(output):
        raise ValueError('Output must be separate from input evidence')
    articles,_,provenance=read_inputs(archive)
    calendar_bytes=cp.read_bytes();calendar=json.loads(calendar_bytes)
    report=prepare_candidates(articles,calendar,symbols=args.symbols,decision_session=args.decision_session,as_of=args.as_of)
    fingerprints={str(archive/name):value for name,value in provenance['input_files'].items()}
    manifest=json.loads((archive/'manifest.json').read_text())
    fingerprints.update({str(archive/request['file']):request['sha256'] for request in manifest['requests']})
    fingerprints[str(cp)]=_sha(calendar_bytes)
    source=Path(__file__).read_bytes();documents={'candidates.json':_json(report),'generation_source.py':source}
    for path,h in fingerprints.items():
        if _sha(Path(path).read_bytes())!=h:raise ValueError('Input changed during preparation')
    run={'input_sha256':fingerprints,'generation_source_sha256':_sha(source),
         'dependency_sha256':{name:_sha(Path(__file__).with_name(name).read_bytes()) for name in
                               ['run_research.py','price_labels.py','corporate_actions.py','event_extraction.py','price_context.py']},
         'outputs':{name:{'sha256':_sha(data),'bytes':len(data)} for name,data in documents.items()}}
    output.mkdir(parents=True,mode=0o700,exist_ok=False);(output/'.gitignore').write_text('*\n')
    for name,data in documents.items():
        with (output/name).open('xb') as stream:stream.write(data)
    with (output/'run_manifest.json').open('xb') as stream:stream.write(_json(run))
    print(json.dumps(report['summary']))
    return report


if __name__=='__main__':main()
