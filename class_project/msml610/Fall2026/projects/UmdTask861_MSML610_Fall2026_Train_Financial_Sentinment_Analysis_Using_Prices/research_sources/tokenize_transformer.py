"""Offline, fixed-tokenizer export of existing isolated company-day packets.

No fitting, prices, target values, model weights, inference or network calls.
Run explicitly in the existing registered NLP venv, not the PyTorch interpreter.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import sys
import time

from price_transformer import new_output, read, save, sha


class ExportRejected(ValueError):
    pass


def headline_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def feature_packet(packet, partition, companies):
    """Project text/IDs only; never inspect the training target's value."""
    if set(packet) != {'companies'} or list(packet['companies']) != companies:
        raise ExportRejected('Unexpected packet/company schema or order')
    output, seen = {'companies': {}}, set()
    expected = {'sample_id', 'headlines'} | ({'target'} if partition == 'train' else set())
    for symbol, rows in packet['companies'].items():
        if not isinstance(rows, list) or not rows:
            raise ExportRejected('Empty company partition')
        output['companies'][symbol] = []
        for row in rows:
            if set(row) != expected:
                raise ExportRejected('Unexpected row fields; heldout outcomes/labels are forbidden')
            sid = row['sample_id']
            if not isinstance(sid, str) or not sid.startswith(symbol + ':') or sid in seen:
                raise ExportRejected('Invalid or duplicate sample identity')
            seen.add(sid)
            titles = row['headlines']
            if (not isinstance(titles, list) or not titles
                    or any(not isinstance(t, str) or not t.strip() for t in titles)):
                raise ExportRejected('Missing/empty headline: ' + sid)
            if len(set(titles)) != len(titles):
                raise ExportRejected('Duplicate title would change the original grouped budget')
            output['companies'][symbol].append({'sample_id': sid, 'headlines': list(titles)})
    return output


def tokenizer_metadata(tokenizer):
    # get_vocab_size() is expensive in the existing binding; these immutable
    # values must not be re-queried for every token of every headline.
    return {'vocab_size': tokenizer.get_vocab_size(),
        'cls_id': tokenizer.token_to_id('[CLS]'), 'sep_id': tokenizer.token_to_id('[SEP]'),
        'unk_id': tokenizer.token_to_id('[UNK]')}


def encode_headline(tokenizer, text, max_tokens, metadata=None):
    if not isinstance(text, str) or not text.strip():
        raise ExportRejected('Empty headline')
    metadata = tokenizer_metadata(tokenizer) if metadata is None else metadata
    encoded = tokenizer.encode(text, add_special_tokens=True)
    ids, mask, types, special = list(encoded.ids), list(encoded.attention_mask), list(encoded.type_ids), list(encoded.special_tokens_mask)
    if encoded.overflowing:
        raise ExportRejected('Tokenizer emitted overflow/truncation segments')
    if not (len(ids) == len(mask) == len(types) == len(special)):
        raise ExportRejected('Token/mask lengths differ')
    if not ids or not any(v == 0 for v in special):
        raise ExportRejected('Empty encoded payload')
    if len(ids) > max_tokens:
        raise ExportRejected('Untruncated headline exceeds registered token budget: ' + headline_sha(text))
    if (ids[0] != metadata['cls_id'] or ids[-1] != metadata['sep_id']
            or any(type(v) is not int or not 0 <= v < metadata['vocab_size'] for v in ids)
            or mask != [1] * len(ids) or types != [0] * len(ids)):
        raise ExportRejected('Unexpected BERT token IDs/masks or active padding')
    return {'input_sha256': headline_sha(text), 'input_ids': ids, 'attention_mask': mask,
        'token_type_ids': types, 'length': len(ids), 'untruncated_length': len(ids), 'truncated': False,
        'unknown_tokens': sum(v == metadata['unk_id'] for v in ids)}


def encode_packets(packets, tokenizer, max_tokens, check=lambda: None):
    cache, output = {}, {}
    # Tokenizer is fixed: sharing text encodings across partitions does no fitting.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    metadata = tokenizer_metadata(tokenizer)
    for partition, packet in packets.items():
        output[partition] = {'companies': {}}
        for symbol, rows in packet['companies'].items():
            output[partition]['companies'][symbol] = []
            for row in rows:
                check()
                encoded_titles = []
                for title in row['headlines']:
                    check()
                    h = headline_sha(title)
                    if h not in cache:
                        cache[h] = encode_headline(tokenizer, title, max_tokens, metadata)
                    encoded_titles.append(cache[h])
                output[partition]['companies'][symbol].append({'sample_id': row['sample_id'], 'headlines': encoded_titles})
    return output, cache


def bound_inputs(prepared, registration):
    receipt_path = prepared / 'preparation_receipt.json'
    receipt = read(receipt_path)
    if receipt['status'] != 'prepared' or receipt['registration_sha256'] != sha(registration):
        raise ExportRejected('Prepared data/registration binding changed')
    names = ['training_packet.json', 'heldout_text_packet.json']
    if set(receipt['files_sha256']) != set(names):
        raise ExportRejected('Unexpected prepared input list')
    for name, digest in receipt['files_sha256'].items():
        if sha(prepared / name) != digest:
            raise ExportRejected('Prepared input bytes changed: ' + name)
    return receipt, {**receipt['files_sha256'], 'preparation_receipt.json': sha(receipt_path)}


def runtime_receipt(reg):
    expected = Path(reg['tokenizer_python'])
    if Path(sys.executable).parent != expected.parent or Path(sys.prefix) != expected.parent.parent:
        raise ExportRejected('Use the existing explicitly registered NLP venv')
    return {'interpreter': sys.executable, 'prefix': sys.prefix, 'python': platform.python_version(),
        'tokenizers': importlib.metadata.version('tokenizers'), 'device': 'CPU',
        'operation': 'Sequential fixed tokenizer.encode; no model or fit'}


def export(prepared, registration, output, *, tokenizer_factory=None):
    prepared, registration = Path(prepared), Path(registration)
    out = new_output(output)
    started = time.monotonic()
    receipt = {'status': 'running', 'real_model_training': False, 'model_inference': False,
        'market_metrics': None, 'targets_exported': False, 'strict_labels_created': 0,
        'rows_selectively_removed': 0, 'padding': False, 'truncation': False}
    try:
        reg = read(registration)
        if reg['inputs']['max_tokens'] != 64 or reg['inputs']['truncation_allowed'] is not False:
            raise ExportRejected('Keep the existing 64-token no-truncation registration')
        manifest, source_hashes = bound_inputs(prepared, registration)
        registration_sha = sha(registration)
        snapshot = Path(reg['tokenizer_snapshot'])
        tokenizer_path, config_path = snapshot / 'tokenizer.json', snapshot / 'config.json'
        if sha(tokenizer_path) != reg['tokenizer_sha256'] or sha(config_path) != reg['config_sha256']:
            raise ExportRejected('Pinned tokenizer/config bytes changed')
        runtime = runtime_receipt(reg)
        companies = reg['companies']
        packets = {part: feature_packet(read(prepared / filename), part, companies) for part, filename in [
            ('train', 'training_packet.json'), ('heldout', 'heldout_text_packet.json')]}
        row_ids = {part: {s: [r['sample_id'] for r in rows] for s, rows in p['companies'].items()}
                   for part, p in packets.items()}
        all_train = {sid for rows in row_ids['train'].values() for sid in rows}
        all_test = {sid for rows in row_ids['heldout'].values() for sid in rows}
        if all_train & all_test:
            raise ExportRejected('Train/heldout sample identities overlap')
        for s in companies:
            if (row_ids['train'][s] != manifest['companies'][s]['train_ids']
                    or row_ids['heldout'][s] != manifest['companies'][s]['test_ids']):
                raise ExportRejected('Ordered IDs differ from prepared acceptance receipt')
        if tokenizer_factory is None:
            from tokenizers import Tokenizer
            tokenizer_factory = Tokenizer.from_file
        tokenizer = tokenizer_factory(str(tokenizer_path))
        seconds = reg['resources']['tokenization_seconds']
        max_rss = reg['resources']['max_rss_bytes']
        def check():
            if time.monotonic() - started >= seconds:
                raise ExportRejected('Tokenization wall budget exhausted')
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_bytes = peak if sys.platform == 'darwin' else peak * 1024
            if peak_bytes > max_rss:
                raise ExportRejected('Tokenization RSS budget exceeded')
        encoded, cache = encode_packets(packets, tokenizer, 64, check)
        check()
        lengths = [v['length'] for v in cache.values()]
        counts = {part: {s: {'rows': len(rows), 'headline_uses': sum(len(r['headlines']) for r in rows)}
                         for s, rows in p['companies'].items()} for part, p in encoded.items()}
        # All validation completes before either token dataset is written.
        for part, filename in [('train', 'training_tokens.json'), ('heldout', 'heldout_tokens.json')]:
            save(out / filename, encoded[part])
        _, fresh_hashes = bound_inputs(prepared, registration)
        if (fresh_hashes != source_hashes or sha(registration) != registration_sha
                or sha(tokenizer_path) != reg['tokenizer_sha256'] or sha(config_path) != reg['config_sha256']):
            raise ExportRejected('A source changed during tokenization')
        receipt.update(status='complete', runtime=runtime, registration_sha256=registration_sha,
            source_files_sha256=source_hashes, tokenizer_sha256=reg['tokenizer_sha256'], config_sha256=reg['config_sha256'],
            generation_source_sha256=sha(__file__), counts=counts, ordered_ids=row_ids,
            unique_headlines=len(cache), headline_uses=sum(c['headline_uses'] for p in counts.values() for c in p.values()),
            minimum_untruncated_length=min(lengths), maximum_untruncated_length=max(lengths),
            unknown_token_occurrences_unique_texts=sum(v['unknown_tokens'] for v in cache.values()),
            max_tokens=64, LM_caveat=reg['LM_caveat'],
            output_files_sha256={n: sha(out / n) for n in ['training_tokens.json', 'heldout_tokens.json']})
    except Exception as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - started
        save(out / 'tokenization_receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    result = export(args.prepared, args.registration, args.output_dir)
    print(json.dumps({k: result[k] for k in ['status', 'unique_headlines', 'headline_uses',
        'maximum_untruncated_length', 'targets_exported', 'elapsed_seconds']}), flush=True)
