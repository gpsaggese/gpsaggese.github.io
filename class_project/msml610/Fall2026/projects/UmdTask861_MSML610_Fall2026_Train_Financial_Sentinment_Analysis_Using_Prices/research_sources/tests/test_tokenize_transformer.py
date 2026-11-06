"""Token export contracts: synthetic packets, no model weights or downloads."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenize_transformer import (ExportRejected, feature_packet, encode_headline,
    encode_packets, export, read, save, sha)


class FakeTokenizer:
    def __init__(self, length=4, empty=False, overflow=False):
        self.length, self.empty, self.overflow = length, empty, overflow
        self.padding = self.truncation = True
        self.calls = 0
        self.vocab_calls = 0
    def no_padding(self):self.padding = False
    def no_truncation(self):self.truncation = False
    def token_to_id(self, token):return {'[CLS]': 101, '[SEP]': 102, '[UNK]': 100}[token]
    def get_vocab_size(self):
        self.vocab_calls += 1
        return 1000
    def encode(self, text, add_special_tokens=True):
        self.calls += 1
        n = 2 if self.empty else self.length
        return SimpleNamespace(ids=[101] + [7] * (n - 2) + [102], attention_mask=[1] * n,
            type_ids=[0] * n, special_tokens_mask=[1] + [0] * (n - 2) + [1],
            overflowing=[object()] if self.overflow else [])


def packets():
    train = {'companies': {'A': [
        {'sample_id': 'A:two', 'headlines': ['second headline', 'shared headline'], 'target': .01},
        {'sample_id': 'A:one', 'headlines': ['first headline'], 'target': -.02}]}}
    test = {'companies': {'A': [{'sample_id': 'A:three', 'headlines': ['shared headline']}]}}
    return train, test


def fixture(root):
    prepared = root / 'prepared';prepared.mkdir()
    snapshot = root / 'snapshot';snapshot.mkdir()
    (snapshot / 'tokenizer.json').write_text('{}\n');(snapshot / 'config.json').write_text('{}\n')
    reg = root / 'registration.json'
    save(reg, {'companies': ['A'], 'inputs': {'max_tokens': 64, 'truncation_allowed': False},
        'tokenizer_python': sys.executable, 'tokenizer_snapshot': str(snapshot),
        'tokenizer_sha256': sha(snapshot / 'tokenizer.json'), 'config_sha256': sha(snapshot / 'config.json'),
        'resources': {'tokenization_seconds': 60, 'max_rss_bytes': 1536 * 1024 * 1024},
        'LM_caveat': 'Synthetic fixture only; no actual dictionary use'})
    a, b = packets();save(prepared / 'training_packet.json', a);save(prepared / 'heldout_text_packet.json', b)
    receipt = {'status': 'prepared', 'registration_sha256': sha(reg), 'companies': {
        'A': {'train_ids': ['A:two', 'A:one'], 'test_ids': ['A:three']}},
        'files_sha256': {n: sha(prepared / n) for n in ['training_packet.json', 'heldout_text_packet.json']}}
    save(prepared / 'preparation_receipt.json', receipt)
    return prepared, reg


class TokenizerExportTests(unittest.TestCase):
    def test_target_values_never_enter_text_features(self):
        a, _ = packets();b = deepcopy(a)
        b['companies']['A'][0]['target'] = object()  # fails if code tries to inspect/serialize it
        b['companies']['A'][1]['target'] = {'a future label': 999}
        self.assertEqual(feature_packet(a, 'train', ['A']), feature_packet(b, 'train', ['A']))
        self.assertTrue(all(set(r) == {'sample_id', 'headlines'} for r in feature_packet(a, 'train', ['A'])['companies']['A']))

    def test_heldout_target_or_price_fields_are_rejected(self):
        for key in ['target', 'label', 'excess_return', 'price_features']:
            _, p = packets();p['companies']['A'][0][key] = 1
            with self.subTest(key=key), self.assertRaises(ExportRejected):feature_packet(p, 'heldout', ['A'])

    def test_duplicate_ids_or_titles_are_not_silently_dropped(self):
        for duplicate in ['id', 'title']:
            p, _ = packets()
            if duplicate == 'id':p['companies']['A'][1]['sample_id'] = 'A:two'
            else:p['companies']['A'][0]['headlines'].append('second headline')
            with self.subTest(duplicate=duplicate), self.assertRaises(ExportRejected):feature_packet(p, 'train', ['A'])

    def test_row_title_order_and_partition_identity_are_preserved(self):
        a, b = packets()
        ps = {'train': feature_packet(a, 'train', ['A']), 'heldout': feature_packet(b, 'heldout', ['A'])}
        tokenizer = FakeTokenizer();encoded, cache = encode_packets(ps, tokenizer, 64)
        self.assertEqual([r['sample_id'] for r in encoded['train']['companies']['A']], ['A:two', 'A:one'])
        expected = [hashlib.sha256(t.encode()).hexdigest() for t in a['companies']['A'][0]['headlines']]
        self.assertEqual([x['input_sha256'] for x in encoded['train']['companies']['A'][0]['headlines']], expected)
        self.assertEqual(len(cache), 3);self.assertEqual(tokenizer.calls, 3)
        self.assertEqual(tokenizer.vocab_calls, 1)
        self.assertFalse(tokenizer.padding);self.assertFalse(tokenizer.truncation)

    def test_exact_budget_boundary_and_overlength_rejection(self):
        self.assertEqual(encode_headline(FakeTokenizer(length=64), 'boundary', 64)['length'], 64)
        with self.assertRaisesRegex(ExportRejected, 'exceeds'):encode_headline(FakeTokenizer(length=65), 'long', 64)

    def test_empty_source_or_encoded_payload_is_explicit_failure(self):
        t = FakeTokenizer()
        with self.assertRaisesRegex(ExportRejected, 'Empty headline'):encode_headline(t, ' \n ', 64)
        self.assertEqual(t.calls, 0)
        with self.assertRaisesRegex(ExportRejected, 'Empty encoded'):encode_headline(FakeTokenizer(empty=True), 'removed characters', 64)
        with self.assertRaisesRegex(ExportRejected, 'overflow'):encode_headline(FakeTokenizer(overflow=True), 'overflow', 64)

    def test_export_replays_identical_bytes_when_training_targets_change(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);prepared, reg = fixture(root)
            first = export(prepared, reg, root / 'first', tokenizer_factory=lambda _: FakeTokenizer())
            p = read(prepared / 'training_packet.json');p['companies']['A'][0]['target'] = -999
            save(prepared / 'training_packet.json', p)
            manifest = read(prepared / 'preparation_receipt.json')
            manifest['files_sha256']['training_packet.json'] = sha(prepared / 'training_packet.json')
            save(prepared / 'preparation_receipt.json', manifest)
            second = export(prepared, reg, root / 'second', tokenizer_factory=lambda _: FakeTokenizer())
            self.assertEqual(first['output_files_sha256'], second['output_files_sha256'])
            self.assertNotEqual(first['source_files_sha256'], second['source_files_sha256'])
            self.assertFalse(second['targets_exported'])

    def test_heldout_contamination_is_caught_before_tokenizer_access(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);prepared, reg = fixture(root)
            p = read(prepared / 'heldout_text_packet.json');p['companies']['A'][0]['target'] = .9
            save(prepared / 'heldout_text_packet.json', p)
            manifest = read(prepared / 'preparation_receipt.json')
            manifest['files_sha256']['heldout_text_packet.json'] = sha(prepared / 'heldout_text_packet.json')
            save(prepared / 'preparation_receipt.json', manifest)
            calls = []
            with self.assertRaises(ExportRejected):export(prepared, reg, root / 'out', tokenizer_factory=lambda p: calls.append(p))
            self.assertEqual(calls, [])
            self.assertEqual(read(root / 'out/tokenization_receipt.json')['status'], 'failed')
            self.assertFalse((root / 'out/training_tokens.json').exists())

    def test_failed_overlength_export_emits_no_token_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);prepared, reg = fixture(root)
            with self.assertRaises(ExportRejected):export(prepared, reg, root / 'out', tokenizer_factory=lambda _: FakeTokenizer(length=65))
            self.assertEqual(read(root / 'out/tokenization_receipt.json')['status'], 'failed')
            self.assertFalse((root / 'out/training_tokens.json').exists())
            self.assertFalse((root / 'out/heldout_tokens.json').exists())

    def test_changed_input_bytes_or_tokenizer_are_not_accepted(self):
        for tampered in ['input', 'tokenizer']:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d);prepared, reg = fixture(root)
                p = prepared / 'training_packet.json' if tampered == 'input' else root / 'snapshot/tokenizer.json'
                p.write_text(p.read_text() + ' ')
                with self.subTest(tampered=tampered), self.assertRaises(ExportRejected):
                    export(prepared, reg, root / 'out', tokenizer_factory=lambda _: FakeTokenizer())

    def test_order_mismatch_or_partition_overlap_is_not_repaired_by_sorting(self):
        for scenario in ['order', 'overlap']:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d);prepared, reg = fixture(root)
                if scenario == 'order':
                    p = read(prepared / 'training_packet.json');p['companies']['A'].reverse();name = 'training_packet.json'
                else:
                    p = read(prepared / 'heldout_text_packet.json');p['companies']['A'][0]['sample_id'] = 'A:two';name = 'heldout_text_packet.json'
                save(prepared / name, p);manifest = read(prepared / 'preparation_receipt.json');manifest['files_sha256'][name] = sha(prepared / name)
                save(prepared / 'preparation_receipt.json', manifest)
                with self.subTest(scenario=scenario), self.assertRaises(ExportRejected):
                    export(prepared, reg, root / 'out', tokenizer_factory=lambda _: FakeTokenizer())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);prepared, reg = fixture(root);out = root / 'keep';out.mkdir();(out / 'original').write_text('original')
            with self.assertRaises(FileExistsError):export(prepared, reg, out, tokenizer_factory=lambda _: FakeTokenizer())
            self.assertEqual((out / 'original').read_text(), 'original')

    def test_cached_real_tokenizer_has_padding_and_truncation_disabled(self):
        from tokenizers import Tokenizer
        import price_transformer
        regpath = Path(price_transformer.__file__).parent / 'transformer_2023_protocol.json'
        if not regpath.is_file():self.skipTest('No existing protocol; no download attempted')
        reg = read(regpath);p = Path(reg['tokenizer_snapshot']) / 'tokenizer.json'
        if not p.is_file():self.skipTest('No pinned local cache; no download attempted')
        self.assertEqual(sha(p), reg['tokenizer_sha256'])
        t = Tokenizer.from_file(str(p));t.enable_truncation(max_length=3);t.enable_padding(length=64)
        title = 'Microsoft reports revenue growth and higher operating profit.'
        packet = {'train': {'companies': {'A': [{'sample_id': 'A:fixture', 'headlines': [title]}]}}}
        result, _ = encode_packets(packet, t, 64)
        x = result['train']['companies']['A'][0]['headlines'][0]
        reference = Tokenizer.from_file(str(p));reference.no_padding();reference.no_truncation()
        self.assertEqual(x['input_ids'], reference.encode(title).ids)
        self.assertGreater(x['length'], 3);self.assertLess(x['length'], 64)
        self.assertEqual(x['attention_mask'], [1] * x['length']);self.assertFalse(x['truncated'])


if __name__ == '__main__':unittest.main()
