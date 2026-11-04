"""Artificial token/target alignment fixtures; no tokenizer/model/optimizer."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grouped_collation import (collate_features, collate_training, align_training_labels, chunks)


def feature(sid, lengths):
    headlines = []
    for i, n in enumerate(lengths):
        text = sid + '/title' + str(i)
        headlines.append({'input_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'input_ids': [101] + [10 + i] * (n - 2) + [102], 'attention_mask': [1] * n,
            'token_type_ids': [0] * n, 'length': n, 'untruncated_length': n,
            'truncated': False, 'unknown_tokens': 0})
    return {'sample_id': sid, 'headlines': headlines}


def original(row, target):
    return {'sample_id': row['sample_id'], 'headlines': [row['sample_id'] + '/title' + str(i)
        for i in range(len(row['headlines']))], 'target': target}


def fixture():
    rows = [feature('A:two', [3]), feature('A:one', [5, 4])]
    labels = [original(rows[1], -.02), original(rows[0], .01)]  # deliberately reordered
    return rows, labels


class CollationTests(unittest.TestCase):
    def test_right_padding_masks_and_source_arrays_unchanged(self):
        rows, _ = fixture();before = deepcopy(rows);b = collate_features(rows)
        self.assertEqual(b['padded_length'], 5)
        self.assertEqual(b['input_ids'][0], [101, 10, 102, 0, 0])
        self.assertEqual(b['attention_mask'], [[1, 1, 1, 0, 0], [1] * 5, [1, 1, 1, 1, 0]])
        self.assertTrue(all(x == [0] * 5 for x in b['token_type_ids']))
        self.assertEqual(rows, before);self.assertNotIn('labels', b)

    def test_group_boundaries_weights_and_mean_semantics(self):
        rows, _ = fixture();b = collate_features(rows)
        self.assertEqual(b['sample_ids'], ['A:two', 'A:one'])
        self.assertEqual(b['group_offsets'], [0, 1, 3])
        self.assertEqual(b['headline_to_group'], [0, 1, 1])
        self.assertEqual(b['headline_weights'], [1., .5, .5])
        values = [10., 2., 6.]
        means = [sum(values[i] * b['headline_weights'][i] for i in range(start, end))
                 for start, end in zip(b['group_offsets'], b['group_offsets'][1:])]
        self.assertEqual(means, [10., 4.])

    def test_reordered_label_source_joins_by_id_with_one_label_per_day(self):
        rows, source = fixture();lookup = align_training_labels(rows, source, {'A:test'})
        b = collate_training(rows, lookup, ['A:two', 'A:one'], {'A:test'})
        self.assertEqual(b['labels'], [.01, -.02])
        self.assertEqual(b['label_sample_ids'], b['sample_ids'])
        self.assertEqual(len(b['labels']), 2);self.assertEqual(len(b['input_ids']), 3)

    def test_duplicate_missing_extra_or_test_label_ids_rejected(self):
        for mutation in ['duplicate', 'missing', 'extra', 'test', 'mismatch']:
            rows, source = fixture()
            if mutation == 'duplicate':source.append(deepcopy(source[0]))
            elif mutation == 'missing':source.pop()
            elif mutation == 'extra':source.append(original(feature('A:extra', [3]), .5))
            elif mutation == 'test':source[0]['sample_id'] = 'A:test'
            else:source[0]['sample_id'] = 'A:other'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                align_training_labels(rows, source, {'A:test'})

    def test_same_id_wrong_original_headline_is_rejected(self):
        rows, source = fixture();source[0]['headlines'][0] = 'different economic event'
        with self.assertRaisesRegex(ValueError, 'mismatched original headlines'):
            align_training_labels(rows, source, {'A:test'})

    def test_nonfinite_or_boolean_training_targets_rejected(self):
        for value in [float('nan'), float('inf'), True, '0.1']:
            rows, source = fixture();source[0]['target'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'target'):
                align_training_labels(rows, source, set())

    def test_heldout_feature_api_cannot_accept_or_emit_targets(self):
        rows, _ = fixture()
        with self.assertRaises(TypeError):collate_features(rows, targets={})
        rows[0]['target'] = .3
        with self.assertRaisesRegex(ValueError, 'must not contain'):
            collate_features(rows)

    def test_training_api_rejects_heldout_or_missing_capability_ids(self):
        rows, source = fixture();lookup = align_training_labels(rows, source, {'A:test'})
        with self.assertRaisesRegex(ValueError, 'full training ID set'):
            collate_training(rows, {'A:one': -.02}, ['A:two', 'A:one'], {'A:test'})
        with self.assertRaisesRegex(ValueError, 'Heldout labels'):
            collate_training(rows, {**lookup, 'A:test': .9}, ['A:two', 'A:one', 'A:test'], {'A:test'})
        with self.assertRaisesRegex(ValueError, 'heldout ID'):
            collate_training([feature('A:test', [3])], lookup, ['A:two', 'A:one'], {'A:test'})

    def test_duplicate_or_mixed_company_batch_ids_rejected(self):
        rows, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            collate_features([rows[0], deepcopy(rows[0])])
        with self.assertRaisesRegex(ValueError, 'Company'):
            collate_features([rows[0], feature('MSFT:one', [3])])

    def test_tail_batch_and_original_order_preserved(self):
        rows = [feature('A:' + str(i), [3]) for i in [4, 3, 2, 1, 0]]
        batches = [collate_features(x) for x in chunks(rows, 4)]
        self.assertEqual([len(b['sample_ids']) for b in batches], [4, 1])
        self.assertEqual([s for b in batches for s in b['sample_ids']], [r['sample_id'] for r in rows])

    def test_invalid_mask_id_length_or_truncation_rejected(self):
        for field, value in [('attention_mask', [1, 0, 1]), ('input_ids', [101, 3.5, 102]),
                             ('length', 4), ('truncated', True), ('token_type_ids', [0, 1, 0])]:
            row = feature('A:one', [3]);row['headlines'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):collate_features([row])

    def test_empty_group_duplicate_headline_and_64_token_boundary(self):
        with self.assertRaises(ValueError):collate_features([])
        with self.assertRaises(ValueError):collate_features([feature('A:empty', [])])
        row = feature('A:one', [3]);row['headlines'].append(deepcopy(row['headlines'][0]))
        with self.assertRaises(ValueError):collate_features([row])
        self.assertEqual(collate_features([feature('A:64', [64])])['padded_length'], 64)
        with self.assertRaises(ValueError):collate_features([feature('A:65', [65])])

    def test_repeated_collation_is_deterministic_without_source_mutation(self):
        rows, source = fixture();before = deepcopy(rows)
        lookup = align_training_labels(rows, source, set())
        a = collate_training(rows, lookup, ['A:two', 'A:one'], set())
        b = collate_training(rows, lookup, ['A:two', 'A:one'], set())
        self.assertEqual(a, b);self.assertEqual(rows, before)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Use only existing torch; do not install')
    def test_existing_cpu_tensor_conversion_has_group_not_headline_labels(self):
        import torch
        rows, source = fixture();lookup = align_training_labels(rows, source, set())
        b = collate_training(rows, lookup, ['A:two', 'A:one'], set())
        ids = torch.tensor(b['input_ids'], dtype=torch.long, device='cpu')
        mask = torch.tensor(b['attention_mask'], dtype=torch.long, device='cpu')
        labels = torch.tensor(b['labels'], dtype=torch.float32, device='cpu')
        self.assertEqual(tuple(ids.shape), (3, 5));self.assertEqual(tuple(labels.shape), (2,))
        self.assertEqual(int(mask.sum()), 12)
        self.assertFalse(labels.requires_grad);self.assertEqual(ids.device.type, 'cpu')


if __name__ == '__main__':unittest.main()
