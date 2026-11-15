"""Data-only grouped right padding and explicit training-label alignment.

No tokenizer, model, optimizer or prediction is invoked. Heldout feature
collation has no target argument. Existing retrospective labels remain separate
from strict total-return eligibility; original registrations are read-only.
"""
from __future__ import annotations
import argparse
import hashlib
import math
from pathlib import Path
import re
import resource
import sys
import time

from price_transformer import new_output, read, save, sha


def sample_ids(rows):
    ids = [r.get('sample_id') for r in rows]
    if not rows or any(not isinstance(s, str) or ':' not in s or not all(s.split(':', 1)) for s in ids):
        raise ValueError('Empty batch or invalid sample ID')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate sample ID')
    if len({s.split(':', 1)[0] for s in ids}) != 1:
        raise ValueError('Company models must not share a collation batch')
    return ids


def align_training_labels(features, original_training_rows, heldout_ids):
    """Join verified original TRAIN labels by ID, never by input position."""
    expected = sample_ids(features)
    source = {}
    for row in original_training_rows:
        if set(row) != {'sample_id', 'headlines', 'target'}:
            raise ValueError('Require original training rows with explicit target')
        sid = row['sample_id']
        if sid in source:
            raise ValueError('Duplicate training-label ID')
        if sid in heldout_ids:
            raise ValueError('Heldout ID cannot supply a training target')
        value = row['target']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Nonfinite or invalid training target')
        source[sid] = row
    if set(source) != set(expected):
        raise ValueError('Missing or extra training-label IDs')
    result = {}
    for row in features:
        sid = row['sample_id']
        if sid in heldout_ids:
            raise ValueError('Heldout ID cannot enter a training batch')
        titles = source[sid]['headlines']
        if not isinstance(titles, list) or any(not isinstance(t, str) or not t.strip() for t in titles):
            raise ValueError('Invalid original training headlines')
        hashes = [hashlib.sha256(t.encode()).hexdigest() for t in titles]
        if hashes != [h['input_sha256'] for h in row['headlines']]:
            raise ValueError('Training ID has mismatched original headlines')
        result[sid] = source[sid]['target']
    return result


def collate_features(rows, *, pad_id=0, max_tokens=64, vocab_size=30522):
    """Feature-only API; rejects attached outcomes and preserves row/title order."""
    ids = sample_ids(rows)
    if type(pad_id) is not int or not 0 <= pad_id < vocab_size or max_tokens != 64:
        raise ValueError('Use the registered BERT padding and token limit')
    headlines, hashes, offsets, group_index, weights, lengths = [], [], [0], [], [], []
    fields = {'input_sha256', 'input_ids', 'attention_mask', 'token_type_ids', 'length',
              'untruncated_length', 'truncated', 'unknown_tokens'}
    for group, row in enumerate(rows):
        if set(row) != {'sample_id', 'headlines'}:
            raise ValueError('Feature rows must not contain labels, targets or prices')
        items = row['headlines']
        if not isinstance(items, list) or not items:
            raise ValueError('Empty company-day headline group')
        seen = set()
        for h in items:
            if set(h) != fields:
                raise ValueError('Unexpected token feature schema')
            digest = h['input_sha256']
            n = h['length']
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest) or digest in seen:
                raise ValueError('Invalid or duplicate grouped headline identity')
            seen.add(digest)
            if (type(n) is not int or not 2 < n <= max_tokens or h['untruncated_length'] != n
                    or h['truncated'] is not False or len(h['input_ids']) != n
                    or len(h['attention_mask']) != n or len(h['token_type_ids']) != n
                    or h['attention_mask'] != [1] * n or h['token_type_ids'] != [0] * n
                    or any(type(v) is not int or not 0 <= v < vocab_size for v in h['input_ids'])):
                raise ValueError('Invalid unpadded IDs/masks/lengths or forbidden truncation')
            headlines.append(h);hashes.append(digest);lengths.append(n)
            group_index.append(group);weights.append(1.0 / len(items))
        offsets.append(len(headlines))
    width = max(lengths)
    return {'sample_ids': ids, 'group_offsets': offsets, 'headline_to_group': group_index,
        'headline_weights': weights, 'headline_hashes': hashes, 'headline_lengths': lengths,
        'input_ids': [h['input_ids'] + [pad_id] * (width - h['length']) for h in headlines],
        'attention_mask': [h['attention_mask'] + [0] * (width - h['length']) for h in headlines],
        'token_type_ids': [h['token_type_ids'] + [0] * (width - h['length']) for h in headlines],
        'padded_length': width}


def collate_training(rows, label_map, training_ids, heldout_ids, **kwargs):
    ids = sample_ids(rows)
    if len(set(training_ids)) != len(training_ids) or set(label_map) != set(training_ids):
        raise ValueError('Training-label capability must match the full training ID set')
    if set(label_map) & set(heldout_ids):
        raise ValueError('Heldout labels cannot enter the training-label capability')
    if any(s in heldout_ids or s not in label_map for s in ids):
        raise ValueError('Missing training ID or heldout ID in training batch')
    batch = collate_features(rows, **kwargs)
    values = [label_map[s] for s in ids]
    if any(isinstance(y, bool) or not isinstance(y, (int, float)) or not math.isfinite(y) for y in values):
        raise ValueError('Invalid training labels')
    batch.update(labels=values, label_sample_ids=list(ids))  # one target per DAY, not per title
    return batch


def chunks(rows, groups_per_batch):
    if type(groups_per_batch) is not int or groups_per_batch < 1:
        raise ValueError('Require a positive company-day batch size')
    for offset in range(0, len(rows), groups_per_batch):
        yield rows[offset:offset + groups_per_batch]  # keep the final partial batch


def run(token_dir, prepared, registration, output, groups_per_batch=4):
    token_dir, prepared, registration = Path(token_dir), Path(prepared), Path(registration)
    out = new_output(output)
    started = time.monotonic()
    receipt = {'status': 'running', 'groups_per_batch': groups_per_batch,
        'model_training': False, 'model_inference': False, 'market_metrics': None, 'strict_labels_created': 0}
    try:
        reg = read(registration);reg_sha = sha(registration);source_sha = sha(__file__)
        if (reg['companies'], reg['horizon_sessions'], reg['target'], reg['pool']) != (
                ['A', 'AMZN', 'MSFT'], 10, 'excess_price_return', 'horizon_specific'):
            raise ValueError('Keep the registered retrospective development target')
        token_receipt = read(token_dir / 'tokenization_receipt.json')
        if token_receipt['status'] != 'complete' or token_receipt['registration_sha256'] != reg_sha:
            raise ValueError('Token export is incomplete or registration changed')
        source_hashes = {n: sha(token_dir / n) for n in [
            'training_tokens.json', 'heldout_tokens.json', 'tokenization_receipt.json']}
        for name, digest in token_receipt['output_files_sha256'].items():
            if source_hashes[name] != digest:
                raise ValueError('Token file changed: ' + name)
        original_hashes = {n: sha(prepared / n) for n in token_receipt['source_files_sha256']}
        if original_hashes != token_receipt['source_files_sha256']:
            raise ValueError('Original split/packet audit bytes changed')
        manifest = read(prepared / 'preparation_receipt.json')
        if manifest['registration_sha256'] != reg_sha or manifest['status'] != 'prepared':
            raise ValueError('Preparation audit binding changed')
        config_path = Path(reg['tokenizer_snapshot']) / 'config.json'
        if sha(config_path) != reg['config_sha256']:
            raise ValueError('BERT configuration changed')
        config = read(config_path)
        settings = {'pad_id': config['pad_token_id'], 'vocab_size': config['vocab_size'], 'max_tokens': reg['inputs']['max_tokens']}
        tokens = {p: read(token_dir / n) for p, n in [
            ('train', 'training_tokens.json'), ('heldout', 'heldout_tokens.json')]}
        original_train = read(prepared / 'training_packet.json')['companies']
        companies = reg['companies'];all_ids = {}
        for part, key in [('train', 'train_ids'), ('heldout', 'test_ids')]:
            if list(tokens[part]['companies']) != companies:
                raise ValueError('Company order/coverage changed')
            all_ids[part] = set()
            for symbol, rows in tokens[part]['companies'].items():
                ids = sample_ids(rows)
                if ids != manifest['companies'][symbol][key] or ids != token_receipt['ordered_ids'][part][symbol]:
                    raise ValueError('Ordered partition IDs changed')
                all_ids[part].update(ids)
        if all_ids['train'] & all_ids['heldout']:
            raise ValueError('Train/heldout identity overlap')
        def check():
            if time.monotonic() - started >= 60:
                raise ValueError('Collation acceptance wall budget exhausted')
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if (peak if sys.platform == 'darwin' else peak * 1024) > reg['resources']['max_rss_bytes']:
                raise ValueError('Collation acceptance RSS budget exhausted')
        output_packets, counts = {}, {}
        for part in ['train', 'heldout']:
            output_packets[part] = {'companies': {}};counts[part] = {}
            for symbol in companies:
                rows = tokens[part]['companies'][symbol]
                label_map = align_training_labels(rows, original_train[symbol], all_ids['heldout']) if part == 'train' else None
                batches = []
                for group in chunks(rows, groups_per_batch):
                    check()
                    batch = (collate_training(group, label_map, manifest['companies'][symbol]['train_ids'], all_ids['heldout'], **settings)
                             if part == 'train' else collate_features(group, **settings))
                    batches.append(batch)
                if [sid for b in batches for sid in b['sample_ids']] != [r['sample_id'] for r in rows]:
                    raise ValueError('Batching changed row order or dropped a tail group')
                output_packets[part]['companies'][symbol] = batches
                counts[part][symbol] = {'rows': len(rows), 'batches': len(batches),
                    'headline_uses': sum(len(b['input_ids']) for b in batches),
                    'valid_tokens': sum(sum(m) for b in batches for m in b['attention_mask']),
                    'padding_tokens': sum(sum(1 - v for v in m) for b in batches for m in b['attention_mask'])}
        check()
        for part, name in [('train', 'training_batches.json'), ('heldout', 'heldout_batches.json')]:
            save(out / name, output_packets[part])
        if (sha(registration) != reg_sha or sha(__file__) != source_sha
                or any(sha(token_dir / n) != h for n, h in source_hashes.items())
                or any(sha(prepared / n) != h for n, h in original_hashes.items())):
            raise ValueError('An input changed during collation')
        receipt.update(status='complete', registration_sha256=reg_sha, generation_source_sha256=source_sha,
            source_token_files_sha256=source_hashes, original_packet_files_sha256=original_hashes,
            interpreter=sys.executable, counts=counts, pad_token_id=settings['pad_id'], max_tokens=64,
            input_rows_removed=0, original_order_preserved=True, training_label_alignment='Exact ID join, original training packet only',
            heldout_labels_exported=False, normalization_fitted=False, LM_caveat=reg['LM_caveat'],
            output_files_sha256={n: sha(out / n) for n in ['training_batches.json', 'heldout_batches.json']})
    except Exception as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - started
        save(out / 'collation_receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--token-dir', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--groups-per-batch', type=int, default=4)
    a = p.parse_args()
    result = run(a.token_dir, a.prepared, a.registration, a.output_dir, a.groups_per_batch)
    print({k: result[k] for k in ['status', 'counts', 'elapsed_seconds', 'model_training', 'heldout_labels_exported']}, flush=True)
