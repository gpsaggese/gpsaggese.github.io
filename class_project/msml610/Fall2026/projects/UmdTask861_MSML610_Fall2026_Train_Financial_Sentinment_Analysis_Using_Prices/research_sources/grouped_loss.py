"""Tensor-only company-day loss acceptance, not a trainer or market model.

L = mean_days((mean_headlines(p) - (y - train_mean)/train_std)**2).
Chunk sums use original day counts; never average chunk means or headline losses.
All graphs are retained until one backward. This is not a streaming-memory claim.
Multi-day mean is an acceptance statistic, not a replacement for the registered
one-optimizer-step-per-day schedule. No optimizer or pretrained weights are used.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys
import time

from grouped_collation import sample_ids, collate_training
from price_transformer import read, save, sha, new_output, target_scale


class TrainingTargets:
    """Training-only scale bound to original ID/value/headline provenance."""
    def __init__(self, rows, heldout_ids):
        self.ids = tuple(sample_ids(rows))
        self.heldout_ids = frozenset(heldout_ids)
        if set(self.ids) & self.heldout_ids:
            raise ValueError('Heldout ID in training targets')
        for row in rows:
            if set(row) != {'sample_id', 'headlines', 'target'}:
                raise ValueError('Require original training rows')
            y = row['target']
            if isinstance(y, bool) or not isinstance(y, (int, float)) or not math.isfinite(y):
                raise ValueError('Require finite numeric training targets')
            if (not isinstance(row['headlines'], list) or not row['headlines']
                    or len(set(row['headlines'])) != len(row['headlines'])
                    or any(not isinstance(t, str) or not t.strip() for t in row['headlines'])):
                raise ValueError('Require original nonempty training headlines')
        self.values = dict(zip(self.ids, (r['target'] for r in rows)))
        self.headlines = {r['sample_id']: [hashlib.sha256(t.encode()).hexdigest()
            for t in r['headlines']] for r in rows}
        self.mean, self.scale = target_scale(rows)  # existing train-only population std policy
        if not math.isfinite(self.mean) or not math.isfinite(self.scale):
            raise ValueError('Nonfinite training normalization')


def validate_batch(batch, targets, expected_ids):
    """Exact ordered IDs and complete groups; labels must match original TRAIN."""
    ids = batch['sample_ids']
    sample_ids([{'sample_id': s} for s in ids])
    if ids != list(expected_ids) or len(set(expected_ids)) != len(expected_ids):
        raise ValueError('Missing, extra or reordered expected training IDs')
    if set(ids) & targets.heldout_ids or not set(ids) <= set(targets.ids):
        raise ValueError('Batch contains heldout or unknown training ID')
    if batch.get('label_sample_ids') != ids or batch.get('labels') != [targets.values[s] for s in ids]:
        raise ValueError('Labels must align by ID to original training values')
    if any(isinstance(y, bool) or not isinstance(y, (int, float)) or not math.isfinite(y)
           for y in batch['labels']):
        raise ValueError('Invalid training labels')
    offsets = batch['group_offsets']; n = len(batch['input_ids']); d = len(ids)
    if (len(offsets) != d + 1 or any(type(x) is not int for x in offsets)
            or offsets[0] != 0 or offsets[-1] != n
            or any(a >= b for a, b in zip(offsets, offsets[1:]))):
        raise ValueError('Missing, duplicate or empty group boundary')
    fields = ['headline_to_group', 'headline_weights', 'headline_hashes', 'headline_lengths',
              'attention_mask', 'token_type_ids']
    if any(len(batch[k]) != n for k in fields):
        raise ValueError('Incomplete grouped feature coverage')
    for g, (a, b) in enumerate(zip(offsets, offsets[1:])):
        if batch['headline_hashes'][a:b] != targets.headlines[ids[g]]:
            raise ValueError('Group headline identities differ from original training row')
        if batch['headline_to_group'][a:b] != [g] * (b - a):
            raise ValueError('Mismatched headline group indices')
        weights = batch['headline_weights'][a:b]
        if any(not math.isfinite(w) or not math.isclose(w, 1 / (b - a), rel_tol=1e-12) for w in weights):
            raise ValueError('Incorrect within-day weights')
    width = batch['padded_length']
    if type(width) is not int or not 3 <= width <= 64:
        raise ValueError('Invalid padded width')
    for i, length in enumerate(batch['headline_lengths']):
        tokens, mask, types = (batch[k][i] for k in ['input_ids', 'attention_mask', 'token_type_ids'])
        if (type(length) is not int or not 3 <= length <= width
                or any(len(x) != width for x in [tokens, mask, types])
                or any(type(v) is not int or not 0 <= v < 30522 for v in tokens)
                or tokens[length:] != [0] * (width - length)
                or mask != [1] * length + [0] * (width - length)
                or types != [0] * width):
            raise ValueError('Invalid right padding or mask')
    return offsets


def _predictions(values):
    import torch
    if (not isinstance(values, torch.Tensor) or values.ndim != 1 or values.numel() == 0
            or values.device.type != 'cpu' or not values.is_floating_point()
            or not bool(torch.isfinite(values).all())):
        raise ValueError('Require finite nonempty 1D CPU prediction tensor')


def _loss(means, batch, targets):
    import torch
    labels = means.new_tensor([(targets.values[s] - targets.mean) / targets.scale
                              for s in batch['sample_ids']])
    loss = ((means - labels) ** 2).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite grouped loss')
    return loss, means


def full_mse(predictions, batch, targets, expected_ids):
    import torch
    offsets = validate_batch(batch, targets, expected_ids)
    _predictions(predictions)
    if len(predictions) != offsets[-1]:
        raise ValueError('Missing or extra headline predictions')
    means = torch.stack([predictions[a:b].mean() for a, b in zip(offsets, offsets[1:])])
    return _loss(means, batch, targets)


def chunked_mse(pieces, batch, targets, expected_ids):
    """Pieces are (original headline start, tensor); exact contiguous coverage."""
    import torch
    offsets = validate_batch(batch, targets, expected_ids)
    contributions = [[] for _ in batch['sample_ids']]; cursor = 0; signature = None
    for start, values in pieces:
        _predictions(values)
        if type(start) is not int or start != cursor or start + len(values) > offsets[-1]:
            raise ValueError('Missing, duplicate or reordered chunk coverage')
        current = (values.dtype, values.device)
        if signature is not None and current != signature:
            raise ValueError('Chunk tensor dtype/device mismatch')
        signature = current
        end = start + len(values)
        for g, (a, b) in enumerate(zip(offsets, offsets[1:])):
            left, right = max(a, start), min(b, end)
            if left < right:
                contributions[g].append(values[left-start:right-start].sum() / (b - a))
        cursor = end
    if cursor != offsets[-1] or any(not parts for parts in contributions):
        raise ValueError('Missing headline or group chunk coverage')
    means = torch.stack([torch.stack(parts).sum() for parts in contributions])
    return _loss(means, batch, targets)


def synthetic_fixture():
    rows, features = [], []
    for day, count in enumerate([1, 3, 11]):
        sid = 'synthetic:' + str(day); titles = [sid + '/' + str(i) for i in range(count)]
        rows.append({'sample_id': sid, 'headlines': titles, 'target': [.01, .02, -.03][day]})
        items = []
        for i, text in enumerate(titles):
            ids = [101] + [10 + day + i] * (1 + i % 4) + [102]
            items.append({'input_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'input_ids': ids, 'attention_mask': [1] * len(ids), 'token_type_ids': [0] * len(ids),
                'length': len(ids), 'untruncated_length': len(ids), 'truncated': False, 'unknown_tokens': 0})
        features.append({'sample_id': sid, 'headlines': items})
    targets = TrainingTargets(rows, {'synthetic:heldout'})
    batch = collate_training(features, targets.values, targets.ids, targets.heldout_ids)
    return batch, targets, rows


def acceptance(collation_dir, prepared, registration, output):
    import torch
    started = time.monotonic();out = new_output(output)
    receipt = {'status': 'running', 'model_loaded': False, 'optimizer_steps': 0,
               'market_predictions': False, 'strict_labels_created': 0, 'alpha_established': False}
    try:
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        reg = read(registration);reg_hash = sha(registration)
        if reg['inputs']['headline_chunk_size'] != 8 or reg['inputs']['max_tokens'] != 64:
            raise ValueError('Registered chunk/token protocol changed')
        collation_dir, prepared = Path(collation_dir), Path(prepared)
        audit = read(collation_dir / 'collation_receipt.json');manifest = read(prepared / 'preparation_receipt.json')
        if audit['status'] != 'complete' or audit['registration_sha256'] != reg_hash or manifest['registration_sha256'] != reg_hash:
            raise ValueError('Incomplete or mismatched input provenance')
        if any(sha(prepared / n) != h for n, h in audit['original_packet_files_sha256'].items()):
            raise ValueError('Original preparation bytes changed')
        batch_path = collation_dir / 'training_batches.json'
        if sha(batch_path) != audit['output_files_sha256'][batch_path.name]:
            raise ValueError('Training collation bytes changed')
        original = read(prepared / 'training_packet.json')['companies'];batches = read(batch_path)['companies']
        if list(batches) != reg['companies'] or list(original) != reg['companies']:
            raise ValueError('Registered company order changed')
        heldout_ids = {s for c in manifest['companies'].values() for s in c['test_ids']}
        counts = {}
        for symbol in reg['companies']:
            targets = TrainingTargets(original[symbol], heldout_ids)
            expected = manifest['companies'][symbol]['train_ids']
            if list(targets.ids) != expected or [s for b in batches[symbol] for s in b['sample_ids']] != expected:
                raise ValueError('Duplicate, missing or reordered company-day coverage')
            cursor = 0
            for b in batches[symbol]:
                validate_batch(b, targets, expected[cursor:cursor+len(b['sample_ids'])]);cursor += len(b['sample_ids'])
            counts[symbol] = {'training_days': cursor, 'batches': len(batches[symbol])}
        # Artificial scalars only: do not apply a surrogate predictor to historical texts.
        b, targets, _ = synthetic_fixture();ids = b['sample_ids'];cases = []
        x = torch.linspace(-.8, 1.2, b['group_offsets'][-1], dtype=torch.float64)
        def evaluate(size):
            w = torch.tensor([.3, -.2], dtype=torch.float64, requires_grad=True)
            p = x * w[0] + w[1]
            result = (full_mse(p, b, targets, ids) if size is None else chunked_mse(
                [(i, p[i:i+size]) for i in range(0, len(p), size)], b, targets, ids))
            result[0].backward()
            return result[0].detach(), result[1].detach(), w.grad.detach().clone()
        reference = evaluate(None)
        for size in [8, 4, 1, 15]:
            result = evaluate(size)
            errors = [float((a-z).abs().max()) for a, z in zip(reference, result)]
            if max(errors) > 1e-12:
                raise ValueError('Synthetic full/chunk loss or gradient discrepancy')
            cases.append({'chunk_size': size, 'loss_abs_error': errors[0], 'day_mean_max_abs_error': errors[1],
                          'shared_parameter_gradient_max_abs_error': errors[2]})
        receipt.update(status='complete', interpreter=sys.executable, torch_version=torch.__version__,
            device='cpu', numerical_dtype='float64', training_structure_counts=counts,
            historical_losses_computed=False, heldout_outcomes_read=False,
            synthetic_groups=3, synthetic_headlines=15, cases=cases, tolerance=1e-12,
            normalization='Training-only company mean/population std; scale1 below1e-8, unchanged policy',
            objective='mean_days((mean_headlines(p)-(y-train_mean)/train_std)^2)',
            schedule='No optimizer; original one-step-per-day registration unchanged',
            registration_sha256=reg_hash, source_sha256=sha(__file__),
            input_sha256={str(p):sha(p) for p in [batch_path, collation_dir/'collation_receipt.json',
                prepared/'training_packet.json', prepared/'preparation_receipt.json', Path(registration)]})
    except Exception as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, reason=str(exc));raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - started
        save(out / 'acceptance_receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collation-dir', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args();result = acceptance(a.collation_dir, a.prepared, a.registration, a.output_dir)
    print({k:result[k] for k in ['status', 'training_structure_counts', 'cases', 'elapsed_seconds']}, flush=True)
