"""Offline transformer plumbing acceptance; no real pretrained training command.

Prepare the registered retrospective company-day packets and test a tiny random
transformer's freezing/group-loss contracts with existing PyTorch. Synthetic
updates are not FinBERT fine-tuning, market predictions, total return or alpha.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import random
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def new_output(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError('Choose a new private output directory')
    path.mkdir(mode=0o700, parents=True)
    (path / '.gitignore').write_text('*\n')
    return path


def aware_time(value):
    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Require an aware label timestamp')
    return value.astimezone(timezone.utc)


def packet_rows(full_rows, split, news, test_start):
    """Preserve the full proxy graph, even excluded transitive bridge rows."""
    from price_labels import _connected_groups

    full = {r['sample_id']: r for r in full_rows}
    if len(full) != len(full_rows):
        raise ValueError('Duplicate full-graph sample identity')
    train_ids = [r['sample_id'] for r in split['train']]
    test_ids = [r['sample_id'] for r in split['test']]
    if (len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids)
            or set(train_ids) & set(test_ids)):
        raise ValueError('Duplicate or overlapping partition identity')
    groups = _connected_groups(full_rows)
    if {groups[s] for s in train_ids} & {groups[s] for s in test_ids}:
        raise ValueError('Full source/event proxy graph crosses split')
    lookup = {n['source_row_index']: n for n in news}
    if len(lookup) != len(news):
        raise ValueError('Duplicate source row identity')
    cutoff = aware_time(test_start + 'T00:00:00Z')
    packets = {}
    for part in ['train', 'test']:
        packets[part] = []
        for row in split[part]:
            if full.get(row['sample_id']) != row:
                raise ValueError('Retained row differs from hash-bound full graph')
            entry, exit_at = aware_time(row['entry_at']), aware_time(row['exit_at'])
            if exit_at <= entry or row['exit_session_index'] - row['entry_session_index'] != 10:
                raise ValueError('Require the original ten-session endpoints')
            if part == 'train' and not entry < exit_at < cutoff:
                raise ValueError('Training outcome is not mature before cutoff')
            if part == 'test' and entry < cutoff:
                raise ValueError('Heldout entry precedes cutoff')
            target = row['excess_return']
            if (row['label_status'] != 'RetrospectiveAvailable'
                    or row['historical_availability_verified'] is not False
                    or not math.isfinite(target)
                    or not math.isclose(target, row['stock_return'] - row['spy_return'], abs_tol=1e-12)):
                raise ValueError('Mixed or invalid retrospective excess-price label')
            titles = list(dict.fromkeys(lookup[i]['headline'] for i in row['source_row_indices']))
            if (not titles or any(not isinstance(t, str) or not t.strip() for t in titles)
                    or '\n'.join(titles) != row['text']):
                raise ValueError('Original distinct-headline budget changed')
            item = {'sample_id': row['sample_id'], 'headlines': titles}
            if part == 'train':
                item['target'] = target
            packets[part].append(item)
    return packets


def prepare(repo, registration, output):
    from lm_baseline import case_split, verify_inputs

    repo, registration = Path(repo), Path(registration)
    reg = read(registration)
    if (reg['companies'], reg['horizon_sessions'], reg['target'], reg['pool']) != (
            ['A', 'AMZN', 'MSFT'], 10, 'excess_price_return', 'horizon_specific'):
        raise ValueError('Use the unchanged registered ten-session development pool')
    lm_path = repo / reg['lm_registration']
    if sha(lm_path) != reg['lm_registration_sha256']:
        raise ValueError('LM registration binding changed')
    lm_reg = read(lm_path)
    prior, _, _ = verify_inputs(repo, lm_reg)
    multi_reg = read(repo / lm_reg['prior_registration'])
    base = repo / reg['baseline_evaluation']
    if sha(base / 'summary.json') != reg['baseline_summary_sha256']:
        raise ValueError('Saved baseline summary changed')
    summary = read(base / 'summary.json')
    for name, digest in summary['output_files_sha256'].items():
        if sha(base / name) != digest:
            raise ValueError('Saved baseline artifact changed: ' + name)
    cases = [c for c in summary['cases'] if (c['horizon'], c['target'], c['pool']) == (
        10, 'excess_price_return', 'horizon_specific')]
    if [c['company'] for c in cases] != reg['companies']:
        raise ValueError('Registered company coverage/order changed')
    train, heldout, audits = {'companies': {}}, {'companies': {}}, {}
    for case in cases:
        symbol = case['company']
        full = read(prior / symbol / 'h10/company_days.json')
        split = case_split(prior, case)
        capture = repo / multi_reg['company_bindings'][symbol]['capture']
        packets = packet_rows(full, split, read(capture / 'news.json'), reg['test_start'])
        train['companies'][symbol] = packets['train']
        heldout['companies'][symbol] = packets['test']
        audits[symbol] = {'train_rows': len(packets['train']), 'test_rows': len(packets['test']),
            'train_ids': [r['sample_id'] for r in packets['train']],
            'test_ids': [r['sample_id'] for r in packets['test']],
            'last_train_outcome': max(r['exit_at'] for r in split['train']),
            'shared_full_graph_proxy_groups': 0}
    out = new_output(output)
    save(out / 'training_packet.json', train)
    save(out / 'heldout_text_packet.json', heldout)
    titles = [t for p in [train, heldout] for rows in p['companies'].values() for r in rows for t in r['headlines']]
    receipt = {'status': 'prepared', 'registration_sha256': sha(registration), 'companies': audits,
        'distinct_headline_uses': len(titles), 'globally_unique_headlines': len(set(titles)),
        'heldout_outcomes_in_training': False, 'heldout_packet_fields': ['sample_id', 'headlines'],
        'real_model_training': False, 'market_metrics': None, 'strict_labels_created': 0,
        'LM_caveat': reg['LM_caveat'], 'files_sha256': {n: sha(out / n) for n in [
            'training_packet.json', 'heldout_text_packet.json']}}
    save(out / 'preparation_receipt.json', receipt)
    return receipt


def preflight(registration):
    reg = read(registration)
    packages = {n: bool(importlib.util.find_spec(n)) for n in ['torch', 'transformers', 'safetensors', 'psutil']}
    blockers = []
    if reg['trainable_checkpoint'] is None:
        blockers.append('No trainable floating checkpoint registered; existing FinBERT snapshot has quantized ONNX only')
    blockers += ['Missing installed package: ' + n for n, found in packages.items() if not found]
    return {'status': 'Blocked' if blockers else 'NotValidated', 'blockers': blockers,
        'packages': packages, 'torch_distribution': importlib.metadata.version('torch') if packages['torch'] else None,
        'registered_updated_parameters': reg['architecture']['updated'], 'actual_FinBERT_updates': 0,
        'real_training_started': False, 'real_evaluation_started': False, 'market_metrics': None,
        'registration_sha256': sha(registration), 'LM_caveat': reg['LM_caveat']}


class BudgetStop(RuntimeError):
    pass


class Budget:
    def __init__(self, seconds, max_rss, min_available, *, clock=time.monotonic, resources=None):
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in [seconds, max_rss, min_available])
                or seconds <= 0 or max_rss <= 0 or min_available < 0):
            raise ValueError('Require a finite positive research budget')
        self.clock, self.started, self.seconds = clock, clock(), seconds
        self.max_rss, self.min_available = max_rss, min_available
        if resources is None:
            import psutil
            resources = lambda: (psutil.Process().memory_info().rss, psutil.virtual_memory().available)
        self.resources = resources
        self.peak_observed_rss = 0
        self.minimum_observed_available = None

    def check(self):
        if self.clock() - self.started >= self.seconds:
            raise BudgetStop('Wall-clock budget exhausted')
        rss, available = self.resources()
        self.peak_observed_rss = max(self.peak_observed_rss, rss)
        self.minimum_observed_available = available if self.minimum_observed_available is None else min(self.minimum_observed_available, available)
        if rss > self.max_rss:
            raise BudgetStop('Process RSS budget exceeded')
        if available < self.min_available:
            raise BudgetStop('Available-memory floor reached')


def target_scale(training_rows):
    y = [r['target'] for r in training_rows]
    if not y or not all(math.isfinite(v) for v in y):
        raise ValueError('Require finite training-only targets')
    mean = sum(y) / len(y)
    scale = math.sqrt(sum((v - mean) ** 2 for v in y) / len(y))
    return mean, scale if scale >= 1e-8 else 1.0


def partial_model(backbone):
    import torch

    class Partial(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = torch.nn.Linear(backbone.config.hidden_size, 1)
            torch.nn.init.normal_(self.head.weight, std=.02)
            torch.nn.init.zeros_(self.head.bias)
            self.block_index = len(backbone.encoder.layer) - 1
            prefix = f'backbone.encoder.layer.{self.block_index}.'
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name.startswith(prefix) or name.startswith('head.'))
            if not any(n.startswith(prefix) and p.requires_grad for n, p in self.named_parameters()):
                raise ValueError('Head-only configuration is not this contract')
            self.eval()  # dropout disabled; selected parameters still use autograd

        def forward(self, batch):
            return self.head(self.backbone(**batch).last_hidden_state[:, 0, :]).squeeze(-1)

    return Partial()


def parameter_hashes(model):
    return {n: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for n, p in model.named_parameters()}


def bag_prediction(model, headlines, tokens, chunk_size, budget):
    import torch
    if not headlines or chunk_size < 1:
        raise ValueError('Require a complete nonempty company-day bag')
    items = [tokens[hashlib.sha256(t.encode()).hexdigest()] for t in headlines]
    pieces = []
    for offset in range(0, len(items), chunk_size):
        budget.check()
        batch = items[offset:offset + chunk_size]
        length = max(len(x['input_ids']) for x in batch)
        padded = {k: torch.tensor([x[k] + [0] * (length - len(x[k])) for x in batch], dtype=torch.long)
                  for k in ['input_ids', 'attention_mask', 'token_type_ids']}
        pieces.append(model(padded))
    return torch.cat(pieces).mean()  # mean before loss, not separate headline labels


def train_synthetic_epoch(model, rows, tokens, config, budget, seed):
    """Acceptance engine; callers must not pass historical market data here."""
    import torch
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate company-day identity')
    if any(not r['sample_id'].startswith('synthetic:') for r in rows):
        raise ValueError('This daily acceptance engine only accepts synthetic fixtures')
    mean, scale = target_scale(rows)
    model.eval()
    before = parameter_hashes(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config['learning_rate'], weight_decay=config['weight_decay'])
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    steps, status, reason = [], 'complete', None
    try:
        for row in ordered:
            budget.check()
            optimizer.zero_grad(set_to_none=True)
            predicted = bag_prediction(model, row['headlines'], tokens, config['headline_chunk_size'], budget)
            loss = (predicted - (row['target'] - mean) / scale) ** 2
            if not bool(torch.isfinite(loss)):
                raise BudgetStop('Nonfinite synthetic loss')
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not all(bool(torch.isfinite(g).all()) for g in grads):
                raise BudgetStop('Nonfinite synthetic gradient')
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config['gradient_clip_norm'])
            optimizer.step()
            steps.append({'sample_id': row['sample_id'], 'normalized_synthetic_loss': float(loss.detach())})
        budget.check()
    except BudgetStop as exc:
        status, reason = 'partial', str(exc)
    after = parameter_hashes(model)
    selected = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen_changed = [n for n in before if n not in selected and before[n] != after[n]]
    if frozen_changed:
        raise ValueError('Frozen parameter changed')
    changed = [n for n in selected if before[n] != after[n]]
    if status == 'complete' and not any(n.startswith('backbone.encoder.layer.') for n in changed):
        raise ValueError('No transformer block update observed')
    return {'status': status, 'reason': reason, 'completed_steps': len(steps), 'required_steps': len(rows),
        'steps': steps, 'training_ids': [r['sample_id'] for r in rows], 'target_mean': mean, 'target_scale': scale,
        'selected_parameters': selected, 'changed_parameters': changed, 'frozen_parameters_changed': frozen_changed,
        'parameter_hashes_before': before, 'parameter_hashes_after': after,
        'synthetic_only': True, 'actual_FinBERT_updates': 0, 'market_predictions': None,
        'test_outcomes_read': False, 'peak_observed_rss_bytes': budget.peak_observed_rss,
        'minimum_observed_available_bytes': budget.minimum_observed_available}


def tiny_fixture(seed):
    import torch
    from types import SimpleNamespace
    torch.manual_seed(seed)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)
            self.embeddings = torch.nn.Embedding(32, 8)
            self.encoder = torch.nn.Module()
            self.encoder.layer = torch.nn.ModuleList([
                torch.nn.TransformerEncoderLayer(8, 2, 16, dropout=0, batch_first=True) for _ in range(2)])

        def forward(self, input_ids, attention_mask, token_type_ids):
            value = self.embeddings(input_ids)
            for layer in self.encoder.layer:
                value = layer(value, src_key_padding_mask=attention_mask == 0)
            return SimpleNamespace(last_hidden_state=value)

    model, rows, tokens = partial_model(Tiny()), [], {}
    for i in range(4):
        titles = ['artificial day ' + str(i) + ' title ' + str(j) for j in range(1 + i % 3)]
        for j, text in enumerate(titles):
            tokens[hashlib.sha256(text.encode()).hexdigest()] = {
                'input_ids': [1, 2 + i, 7 + j, 3], 'attention_mask': [1] * 4, 'token_type_ids': [0] * 4}
        rows.append({'sample_id': 'synthetic:' + str(i), 'headlines': titles, 'target': (i - 1.5) * .01})
    return model, rows, tokens


def dry_run(registration, output):
    import torch
    reg = read(registration)
    out = new_output(output)
    torch.set_num_threads(reg['resources']['threads'])
    torch.use_deterministic_algorithms(True)
    model, rows, tokens = tiny_fixture(reg['training']['seed'])
    cfg = {**reg['training'], 'headline_chunk_size': reg['inputs']['headline_chunk_size']}
    limits = reg['resources']
    started = time.monotonic()
    receipt = train_synthetic_epoch(model, rows, tokens, cfg, Budget(
        limits['synthetic_seconds'], limits['max_rss_bytes'], limits['min_available_bytes']), reg['training']['seed'])
    receipt.update(registration_sha256=sha(registration), elapsed_seconds=time.monotonic() - started,
        torch_version=torch.__version__, interpreter=__import__('sys').executable, device='cpu', threads=torch.get_num_threads(),
        base_layers=2, updated_block_index=1, pretrained_weights_loaded=False, real_training_started=False)
    save(out / 'synthetic_receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'preflight', 'dry-run'])
    p.add_argument('--repo', type=Path)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if a.mode == 'prepare':
        result = prepare(a.repo, a.registration, a.output_dir)
    elif a.mode == 'preflight':
        result = preflight(a.registration)
        save(new_output(a.output_dir) / 'preflight.json', result)
    else:
        result = dry_run(a.registration, a.output_dir)
    print(json.dumps({'status': result['status'], 'real_training_started': False,
        'completed_steps': result.get('completed_steps'), 'synthetic_only': result.get('synthetic_only', False)}), flush=True)
