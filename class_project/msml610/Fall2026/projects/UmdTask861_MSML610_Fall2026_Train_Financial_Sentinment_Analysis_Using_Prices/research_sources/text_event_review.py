"""Validate assistant source-text review and carry associations to split gates.

This review cannot authenticate issuer facts, certify independent human review,
create return labels, or turn a post-open selection into a frozen prediction.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import argparse,hashlib,json
from pathlib import Path

if __package__:
    from .run_research import read_inputs,timestamp
    from .price_labels import make_time_split
else:
    from run_research import read_inputs,timestamp
    from price_labels import make_time_split

RELEVANCE={'direct','contextual','unrelated','uncertain'}
KINDS={'reported_event','reported_historical_event','reported_external_event','conservative_context'}

def _sha(data):return hashlib.sha256(data).hexdigest()
def _json(value):return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def _text(value):return isinstance(value,str) and bool(value.strip()) and value==value.strip()


def apply_review(candidates,review,*,candidate_sha256,as_of):
    """Attach version-bound annotations while preserving original inputs/clocks.

    Direct callers must bind candidate_sha256 to the original retained bytes.
    CLI callers additionally verify the original raw archive and its manifest.
    Association IDs include conservative context as well as reported events;
    none is represented as an independently validated financial event/sample.
    """
    clock=timestamp(as_of)
    if review.get('schema_version')!='financial-text-event-review-1.0' or review.get('candidate_sha256')!=candidate_sha256:
        raise ValueError('Review is not bound to this candidate snapshot')
    created=timestamp(review['review_created_at'])
    if created>clock:raise ValueError('Review is not available as_of; never backdate it')
    reviewer=review.get('reviewer',{})
    if reviewer.get('kind')!='assistant' or reviewer.get('independent_human_review') is not False:
        raise ValueError('This contract is assistant review, not independent human certification')
    groups={}
    for item in review.get('groups',[]):
        identity=item.get('group_id')
        if not _text(identity) or identity in groups or item.get('kind') not in KINDS or not _text(item.get('description')):
            raise ValueError('Distinct explicit group definitions required')
        if item.get('independent_event_verified') is not False:
            raise ValueError('Reported groups cannot be independently certified by this review')
        groups[identity]=item
    pairs={};source_records={}
    for candidate in candidates['candidates']:
        for article in candidate['texts']:
            key=(article['article_id'],candidate['symbol'])
            if key in pairs:raise ValueError('Duplicate original company/article pair')
            pairs[key]=article
            if article['article_id'] in source_records and source_records[article['article_id']]!=article:
                raise ValueError('Conflicting original text versions')
            source_records[article['article_id']]=article
    decisions={};members={g:set() for g in groups}
    def evidence(article,spans):
        if not isinstance(spans,list) or not spans:raise ValueError('Review evidence spans required')
        for item in spans:
            field=item.get('field');start,end=item.get('start'),item.get('end')
            if field not in ['headline','body'] or type(start) is not int or type(end) is not int:
                raise ValueError('Invalid evidence coordinates')
            value=article[field]
            if not 0<=start<end<=len(value) or _sha(value[start:end].encode())!=item.get('span_sha256'):
                raise ValueError('Review evidence differs from retained text')
    for item in review.get('decisions',[]):
        key=(item.get('article_id'),item.get('symbol'))
        if key not in pairs or key in decisions:raise ValueError('Review pair missing, duplicated or outside snapshot')
        article=pairs[key]
        if item.get('retained_text_sha256')!=article['retained_text_sha256'] or item.get('relevance') not in RELEVANCE or not _text(item.get('rationale')):
            raise ValueError('Text version, relevance and rationale must be explicit')
        expected='body_and_headline' if article['body_present'] else 'headline_only'
        if item.get('review_basis')!=expected:raise ValueError('Missing body cannot be represented as body review')
        evidence(article,item.get('evidence_spans'))
        events=item.get('reported_event_ids');contexts=item.get('association_group_ids')
        if not isinstance(events,list) or not isinstance(contexts,list) or len(set(events+contexts))!=len(events+contexts):
            raise ValueError('Distinct event/context ID lists required')
        if item['relevance']=='direct' and not events:raise ValueError('Direct review requires a text-supported reported event')
        if item['relevance'] in ['uncertain','unrelated'] and (events or contexts):
            raise ValueError('Do not invent events for unsubstantiated company relevance')
        refs={}
        for entry in item.get('group_evidence',[]):
            group=entry.get('group_id')
            if group in refs:raise ValueError('Duplicate group evidence')
            evidence(article,entry.get('evidence_spans'));refs[group]=entry
        if set(refs)!=set(events+contexts):raise ValueError('Every association needs retained-text evidence')
        for group in events+contexts:
            if group not in groups:raise ValueError('Undefined group')
            if (group in contexts)!=(groups[group]['kind']=='conservative_context'):
                raise ValueError('Context association must not be called a reported event')
            members[group].add(item['article_id'])
        decisions[key]=item
    if set(decisions)!=set(pairs):raise ValueError('Every retained pair requires an explicit review disposition')
    if any(not member for member in members.values()):raise ValueError('Unused event definition')
    result=deepcopy(candidates)
    result['review_as_of']=as_of;result['review_created_at']=review['review_created_at']
    result['reviewer']=deepcopy(reviewer);result['reported_groups']=deepcopy(review['groups'])
    for row in result['candidates']:
        if created<timestamp(row['input_snapshot_created_at']):raise ValueError('Review cannot precede its input snapshot')
        annotations=[deepcopy(decisions[(a['article_id'],row['symbol'])]) for a in row['texts']]
        row['relevance_review']=annotations
        row['proposed_direct_source_ids']=[a['source_id'] for a in row['texts'] if decisions[(a['article_id'],row['symbol'])]['relevance']=='direct']
        row['source_event_ids']=sorted({g for a in annotations for g in a['reported_event_ids']})
        row['association_group_ids']=sorted({g for a in annotations for g in a['association_group_ids']})
        row['event_cluster_ids']=sorted(set(row['source_event_ids']+row['association_group_ids']))
        row['event_identity_status']='Assistant-reviewed retained-text associations; independent verification pending'
        row['review_available_before_input_cutoff']=created<=timestamp(row['input_cutoff'])
        row['review_available_before_prediction_deadline']=created<=timestamp(row['prediction_deadline'])
        row['blockers']=[b for b in row['blockers'] if b!='reviewed_event_identity_missing']
        row['blockers'].append('independent_event_confirmation_pending')
        if any(a['relevance']=='uncertain' for a in annotations):row['blockers'].append('unresolved_relevance_in_original_inputs')
        if not row['review_available_before_input_cutoff']:row['blockers'].append('review_selection_after_input_cutoff')
        if not row['review_available_before_prediction_deadline']:row['blockers'].append('review_after_prediction_deadline')
        row['blockers']=list(dict.fromkeys(row['blockers']))
        row['eligible_for_training']=False  # This adapter has no label/model constructor.
        # Preserve all original source IDs/texts, even excluded/contextual inputs,
        # so they remain visible to transitive leakage isolation.
    counts=Counter(item['relevance'] for item in decisions.values())
    result['review_summary']={'pairs_reviewed':len(decisions),'relevance_counts':dict(counts),
        'proposed_direct_company_article_pairs':counts['direct'],
        'reported_group_definitions':sum(g['kind']!='conservative_context' for g in groups.values()),
        'conservative_context_definitions':sum(g['kind']=='conservative_context' for g in groups.values()),
        'multi_article_reported_groups':sum(len(members[key])>1 and groups[key]['kind']!='conservative_context' for key in groups),
        'independent_human_review_completed':False,'eligible_training_samples':0,'labels_created':0,'baseline_status':'Not run: no eligible labeled samples'}
    return result


def split_gate_inputs(reviewed):
    """Non-label stubs carry full source/event graph into the existing gate."""
    return [{'sample_id':row['sample_id'],'source_ids':deepcopy(row['source_ids']),
             'event_cluster_ids':deepcopy(row['event_cluster_ids']),'horizon_sessions':10,
             'calendar_id':reviewed['calendar_id'],'calendar_version':reviewed['calendar_version'],
             'return_basis':None,'status':'Unavailable','label_status':'Unavailable',
             'entry_at':row['horizons']['10']['entry_at'],'exit_at':row['horizons']['10']['exit_at'],
             'label_available_at':None,'prediction_completed_at':row['prediction_completed_at'],
             'prediction_deadline':row['prediction_deadline'],'is_label':False,
             'blocking_reasons':deepcopy(row['blockers'])} for row in reviewed['candidates']]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['candidates','review','input-dir','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    for name in ['as-of','training-cutoff','test-start','test-end']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv);output=args.output_dir.resolve();cp=args.candidates.resolve();rp=args.review.resolve();archive=args.input_dir.resolve()
    if output.exists():raise FileExistsError('New review output required')
    if output.is_relative_to(archive) or archive.is_relative_to(output) or cp.is_relative_to(output) or rp.is_relative_to(output):
        raise ValueError('Output must be separate from retained evidence')
    candidate_bytes=cp.read_bytes();review_bytes=rp.read_bytes();candidates=json.loads(candidate_bytes)
    original_manifest=cp.with_name('run_manifest.json');om=json.loads(original_manifest.read_text())
    if om['outputs']['candidates.json']['sha256']!=_sha(candidate_bytes):raise ValueError('Original snapshot hash mismatch')
    articles,_,provenance=read_inputs(archive);by_id={str(a['id']):a for a in articles}
    for row in candidates['candidates']:
        for article in row['texts']:
            raw=by_id[article['article_id']]
            if (row['symbol'] not in raw['symbols'] or article['headline']!=raw['headline'] or article['body']!=raw['text']
                    or article['source_id']!='alpaca-news:'+str(raw['id']) or article['url']!=raw['url']
                    or _sha((raw['headline']+'\n'+raw['text']).encode())!=article['retained_text_sha256']):
                raise ValueError('Candidate differs from retained archive')
            for field,raw_field in [('published_at','created_at'),('version_updated_at','updated_at'),('actual_receipt_at','retrieved_at')]:
                if timestamp(article[field])!=timestamp(raw[raw_field]):raise ValueError('Candidate clock changed')
    reviewed=apply_review(candidates,json.loads(review_bytes),candidate_sha256=_sha(candidate_bytes),as_of=args.as_of)
    stubs=split_gate_inputs(reviewed)
    split=make_time_split(stubs,args.training_cutoff,args.test_start,args.test_end)
    split['scope']='Existing strict split invoked with non-label Unavailable stubs; no returns or model evaluation'
    inputs={str(cp):_sha(candidate_bytes),str(rp):_sha(review_bytes),str(original_manifest):_sha(original_manifest.read_bytes())}
    inputs.update({str(archive/name):h for name,h in provenance['input_files'].items()})
    manifest=json.loads((archive/'manifest.json').read_text());inputs.update({str(archive/r['file']):r['sha256'] for r in manifest['requests']})
    documents={'reviewed_candidates.json':_json(reviewed),'split_gate_inputs.json':_json(stubs),'split_gate.json':_json(split),
               'generation_source.py':Path(__file__).read_bytes()}
    for path,h in inputs.items():
        if _sha(Path(path).read_bytes())!=h:raise ValueError('Input changed during review')
    run={'input_sha256':inputs,'dependency_sha256':{n:_sha(Path(__file__).with_name(n).read_bytes()) for n in
        ['price_labels.py','corporate_actions.py','run_research.py','event_extraction.py','price_context.py']},
        'outputs':{name:{'sha256':_sha(data),'bytes':len(data)} for name,data in documents.items()}}
    output.mkdir(parents=True,mode=0o700,exist_ok=False);(output/'.gitignore').write_text('*\n')
    for name,data in documents.items():
        with (output/name).open('xb') as stream:stream.write(data)
    with (output/'run_manifest.json').open('xb') as stream:stream.write(_json(run))
    print(json.dumps({**reviewed['review_summary'],'split_train':len(split['train']),'split_test':len(split['test']),'split_excluded':len(split['excluded'])}))
    return reviewed,split


if __name__=='__main__':main()
