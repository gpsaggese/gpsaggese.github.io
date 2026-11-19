"""Offline daily execution manifest and MOCK boundary replay, not a trainer.

Training-only identity/normalization reuse the existing packet/collation audit.
One company-day has one zero/backward/clip/step MARKER after all of its title
chunks. Mock completion never authorizes heldout evaluation or claims weights
were updated. No torch, model, optimizer, tokenizer or network is invoked.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from unittest.mock import Mock

from grouped_loss import TrainingTargets, validate_batch
from price_transformer import read, save, sha, new_output


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
        separators=(',', ':')).encode()).hexdigest()


def build_manifest(reg, training, batches, partition):
    """Verify all canonical TRAIN rows before any shuffle; never ingest test labels."""
    companies = reg['companies'];cfg = reg['training'];chunk_size = reg['inputs']['headline_chunk_size']
    if ((reg['horizon_sessions'], reg['target'], reg['pool']) != (10, 'excess_price_return', 'horizon_specific')
            or cfg['epochs'] != 1 or type(cfg['seed']) is not int or chunk_size != 8
            or reg['inputs']['max_tokens'] != 64 or cfg['gradient_clip_norm'] != 1.0):
        raise ValueError('Use the unchanged registered one-epoch day/chunk contract')
    if len(set(companies)) != len(companies) or any(list(x) != companies for x in [training, batches, partition]):
        raise ValueError('Company identity/order/coverage changed')
    heldout = {s for company in partition.values() for s in company['test_ids']}
    plan = {'kind': 'mock_daily_execution_manifest', 'version': 1,
        'company_order': companies[:], 'seed': cfg['seed'], 'epochs': 1,
        'headline_chunk_size': chunk_size, 'optimization_unit': 'company_day',
        'max_day_steps': cfg['max_total_steps'], 'gradient_clip_norm': cfg['gradient_clip_norm'],
        'target_values_exported': False, 'real_training_started': False, 'companies': []}
    total = 0
    for index, symbol in enumerate(companies):
        targets = TrainingTargets(training[symbol], heldout)
        canonical = partition[symbol]['train_ids']
        if list(targets.ids) != canonical or any(s.split(':', 1)[0] != symbol for s in canonical):
            raise ValueError('Canonical original training ID order changed')
        if [s for b in batches[symbol] for s in b['sample_ids']] != canonical:
            raise ValueError('Missing, duplicate or reordered collation day coverage')
        cursor = 0
        for batch in batches[symbol]:
            validate_batch(batch, targets, canonical[cursor:cursor + len(batch['sample_ids'])])
            cursor += len(batch['sample_ids'])
        normalized = [(s, (targets.values[s] - targets.mean) / targets.scale) for s in canonical]
        if not all(math.isfinite(y) for _, y in normalized):
            raise ValueError('Nonfinite train-only normalization')
        order = list(range(len(canonical)));seed = cfg['seed'] + index
        random.Random(seed).shuffle(order)
        days = []
        for row_index in order:
            sid = canonical[row_index];n = len(targets.headlines[sid])
            days.append({'sample_id': sid, 'source_row_index': row_index, 'headline_count': n,
                'headline_identity_sha256': digest(targets.headlines[sid]),
                'chunks': [[i, min(i + chunk_size, n)] for i in range(0, n, chunk_size)]})
        plan['companies'].append({'symbol': symbol, 'seed': seed, 'canonical_ids': canonical[:],
            'normalization': {'source': 'original_training_packet_only', 'count': len(canonical),
                'mean': targets.mean, 'scale': targets.scale, 'training_ids_sha256': digest(canonical),
                'normalized_targets_sha256': digest(normalized)}, 'days': days})
        total += len(days)
    if total != cfg['max_total_steps']:
        raise ValueError('Registered total company-day step count changed')
    plan['payload_sha256'] = digest(plan)
    validate_manifest(plan)
    return plan


def validate_manifest(plan):
    if plan.get('payload_sha256') != digest({k:v for k,v in plan.items() if k != 'payload_sha256'}):
        raise ValueError('Manifest identity changed')
    if (plan['kind'] != 'mock_daily_execution_manifest' or plan['optimization_unit'] != 'company_day'
            or plan['epochs'] != 1 or plan['headline_chunk_size'] != 8
            or plan['real_training_started'] is not False or plan['target_values_exported'] is not False):
        raise ValueError('Not a mock registered daily plan')
    if [c['symbol'] for c in plan['companies']] != plan['company_order']:
        raise ValueError('Company reset/order mismatch')
    all_ids = []
    for index, company in enumerate(plan['companies']):
        canonical = company['canonical_ids'];days = company['days']
        order = list(range(len(canonical)));random.Random(plan['seed'] + index).shuffle(order)
        if (company['seed'] != plan['seed'] + index or not canonical or len(set(canonical)) != len(canonical)
                or [d['sample_id'] for d in days] != [canonical[i] for i in order]
                or [d['source_row_index'] for d in days] != order):
            raise ValueError('Deterministic training-day identity/order mismatch')
        norm = company['normalization']
        if (norm['count'] != len(canonical) or norm['source'] != 'original_training_packet_only'
                or norm['training_ids_sha256'] != digest(canonical)
                or not math.isfinite(norm['mean']) or not math.isfinite(norm['scale']) or norm['scale'] <= 0):
            raise ValueError('Training normalization provenance mismatch')
        for day in days:
            n = day['headline_count']
            if type(n) is not int or n < 1 or day['chunks'] != [[i, min(i+8,n)] for i in range(0,n,8)]:
                raise ValueError('Missing, duplicate or crossing day chunk boundary')
        all_ids.extend(canonical)
    if len(set(all_ids)) != len(all_ids) or len(all_ids) != plan['max_day_steps']:
        raise ValueError('Duplicate/missing global training-day coverage')


class MockBudgetExhausted(Exception):
    pass


def mock_replay(plan, budget_checks=None):
    """Only unittest.mock calls: no real gradient clear, backward or optimizer."""
    validate_manifest(plan)
    if budget_checks is not None and (type(budget_checks) is not int or budget_checks < 0):
        raise ValueError('Mock budget must be a nonnegative integer or None')
    hooks = Mock(name='boundary_markers');allowed = 0;active = None;completed = []
    ordered_ids = [d['sample_id'] for c in plan['companies'] for d in c['days']]
    def check():
        nonlocal allowed
        if budget_checks is not None and allowed >= budget_checks:
            raise MockBudgetExhausted('Simulated budget exhaustion, not a runtime resource measurement')
        allowed += 1
    try:
        for company in plan['companies']:
            hooks.company_reset_marker(company['symbol'])  # does not load/reset an actual checkpoint
            for day in company['days']:
                check();active = day['sample_id'];hooks.zero_grad_marker(active)
                for start, stop in day['chunks']:
                    check();hooks.headline_chunk_marker(active, start, stop)
                hooks.day_loss_marker(active);hooks.backward_marker(active)
                hooks.clip_marker(active, plan['gradient_clip_norm']);hooks.day_step_marker(active)
                completed.append(active);active = None
        status = 'mock_complete'
    except MockBudgetExhausted:
        status = 'mock_partial'
    pending = [s for s in ordered_ids if s not in set(completed) and s != active]
    return {'status': status, 'completed_mock_day_ids': completed, 'incomplete_mock_day': active,
        'not_started_mock_day_ids': pending, 'allowed_mock_budget_checks': allowed,
        'events': [[call[0], *call[1]] for call in hooks.mock_calls],
        'real_training_complete': False, 'real_optimizer_steps': 0,
        'heldout_evaluation_authorized': False, 'checkpoint_state_saved': False}


def run(collation_dir, prepared, registration, output):
    started = time.monotonic();out = new_output(output)
    receipt = {'status': 'running', 'model_loaded': False, 'optimizer_steps': 0,
        'market_evaluation': False, 'strict_labels_created': 0, 'alpha_established': False}
    try:
        collation_dir, prepared, registration = Path(collation_dir), Path(prepared), Path(registration)
        names = [registration, collation_dir/'training_batches.json', collation_dir/'collation_receipt.json',
            prepared/'training_packet.json', prepared/'preparation_receipt.json']
        bindings = {str(p):sha(p) for p in names};reg = read(registration)
        audit = read(collation_dir/'collation_receipt.json');partition = read(prepared/'preparation_receipt.json')
        if (audit['status'] != 'complete' or partition['status'] != 'prepared'
                or any(x['registration_sha256'] != sha(registration) for x in [audit, partition])
                or sha(collation_dir/'training_batches.json') != audit['output_files_sha256']['training_batches.json']
                or any(sha(prepared/n) != audit['original_packet_files_sha256'][n]
                       for n in ['training_packet.json','preparation_receipt.json'])):
            raise ValueError('Incomplete/changed registered preparation or collation')
        training = read(prepared/'training_packet.json')['companies']
        batches = read(collation_dir/'training_batches.json')['companies']
        a = build_manifest(reg, training, batches, partition['companies'])
        b = build_manifest(reg, training, batches, partition['companies'])
        save(out/'manifest_a.json', a);save(out/'manifest_b.json', b)
        if (out/'manifest_a.json').read_bytes() != (out/'manifest_b.json').read_bytes():
            raise ValueError('Deterministic manifest byte replay failed')
        full = mock_replay(a);zero = mock_replay(a, 0)
        prefix = 0;stop_limit = None
        for c in a['companies']:
            for day in c['days']:
                if len(day['chunks']) > 1:
                    stop_limit = prefix + 2;break
                prefix += 1 + len(day['chunks'])
            if stop_limit is not None:break
        if stop_limit is None:raise ValueError('No real multi-chunk day for partial boundary acceptance')
        partial = mock_replay(a, stop_limit)
        if (full['status'] != 'mock_complete' or len(full['completed_mock_day_ids']) != reg['training']['max_total_steps']
                or zero['completed_mock_day_ids'] or zero['incomplete_mock_day'] is not None
                or partial['status'] != 'mock_partial' or partial['incomplete_mock_day'] is None):
            raise ValueError('Mock full/partial day accounting failed')
        for name,value in [('mock_complete.json',full),('mock_before_first_day.json',zero),('mock_inside_day.json',partial)]:
            save(out/name,value)
        counts = {}
        for company in a['companies']:
            days = company['days'];symbol=company['symbol']
            counts[symbol] = {'seed':company['seed'], 'days':len(days), 'collation_batches':len(batches[symbol]),
                'retained_tail_days':len(batches[symbol][-1]['sample_ids']) if len(batches[symbol][-1]['sample_ids'])<audit['groups_per_batch'] else 0,
                'headline_uses':sum(d['headline_count'] for d in days), 'title_chunks':sum(len(d['chunks']) for d in days),
                'partial_title_chunks':sum(stop-start<8 for d in days for start,stop in d['chunks'])}
        if any(sha(Path(p))!=h for p,h in bindings.items()):raise ValueError('Input changed during acceptance')
        if any(n in sys.modules for n in ['torch','transformers','safetensors']):
            raise ValueError('This increment must remain stdlib/mock only')
        receipt.update(status='complete', interpreter=sys.executable, source_sha256=sha(__file__),
            input_sha256=bindings, manifest_sha256=sha(out/'manifest_a.json'), manifest_bytes_identical=True,
            counts=counts, completed_mock_day_step_markers=len(full['completed_mock_day_ids']),
            zero_grad_markers=sum(e[0]=='zero_grad_marker' for e in full['events']),
            backward_markers=sum(e[0]=='backward_marker' for e in full['events']),
            company_reset_markers=sum(e[0]=='company_reset_marker' for e in full['events']),
            partial_incomplete_day=partial['incomplete_mock_day'], partial_completed_days=len(partial['completed_mock_day_ids']),
            heldout_text_or_outcomes_loaded=False, target_values_exported=False, torch_imported=False,
            real_training_complete=False, heldout_evaluation_authorized=False,
            output_sha256={p.name:sha(p) for p in out.glob('*.json')},
            LM_caveat=reg['LM_caveat'])
    except Exception as exc:
        receipt.update(status='failed',error_type=type(exc).__name__,reason=str(exc));raise
    finally:
        receipt['elapsed_seconds']=time.monotonic()-started;save(out/'acceptance_receipt.json',receipt)
    return receipt


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collation-dir',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();r=run(a.collation_dir,a.prepared,a.registration,a.output_dir)
    print({k:r[k] for k in ['status','counts','completed_mock_day_step_markers','elapsed_seconds']},flush=True)
