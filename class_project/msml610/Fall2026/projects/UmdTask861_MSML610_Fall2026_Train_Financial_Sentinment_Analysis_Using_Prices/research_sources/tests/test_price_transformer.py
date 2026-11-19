"""Artificial contract fixtures; never load pretrained weights or market models."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from price_transformer import (packet_rows, target_scale, Budget, BudgetStop,
    bag_prediction, tiny_fixture, train_synthetic_epoch, new_output)


def packet_fixture():
    rows = []
    for name, entry, exit_at, index, source in [
            ('train', '2023-08-14T13:30:00Z', '2023-08-28T13:30:00Z', 100, 1),
            ('test', '2023-09-05T13:30:00Z', '2023-09-19T13:30:00Z', 116, 2)]:
        rows.append({'sample_id': name, 'entry_at': entry, 'exit_at': exit_at,
            'entry_session_index': index, 'exit_session_index': index + 10,
            'source_ids': ['source-' + name], 'event_cluster_ids': ['event-' + name],
            'stock_return': .03, 'spy_return': .02, 'excess_return': .01,
            'label_status': 'RetrospectiveAvailable', 'historical_availability_verified': False,
            'source_row_indices': [source], 'text': 'original title ' + name})
    news = [{'source_row_index': i + 1, 'headline': r['text']} for i, r in enumerate(rows)]
    return rows, {'train': [rows[0]], 'test': [rows[1]]}, news


def unlimited_fixture_budget():
    return Budget(20, 1000, 0, clock=lambda: 0, resources=lambda: (1, 1000))


CONFIG = {'learning_rate': 2e-5, 'weight_decay': 0., 'gradient_clip_norm': 1., 'headline_chunk_size': 2}


class PacketTests(unittest.TestCase):
    def test_heldout_packet_contains_no_price_outcome_or_target(self):
        full, split, news = packet_fixture()
        packets = packet_rows(full, split, news, '2023-09-01')
        self.assertEqual(set(packets['test'][0]), {'sample_id', 'headlines'})
        self.assertEqual(packets['train'][0]['target'], .01)
        self.assertEqual(packets['train'][0]['headlines'], ['original title train'])

    def test_offset_timestamp_is_compared_in_utc_for_maturity(self):
        full, split, news = packet_fixture()
        full[0]['exit_at'] = '2023-08-31T23:30:00-04:00'
        with self.assertRaisesRegex(ValueError, 'mature'):
            packet_rows(full, split, news, '2023-09-01')

    def test_excluded_bridge_still_purges_transitive_event_link(self):
        full, split, news = packet_fixture()
        bridge = {**full[0], 'sample_id': 'excluded-bridge',
            'source_ids': full[0]['source_ids'], 'event_cluster_ids': full[1]['event_cluster_ids']}
        full.append(bridge)
        with self.assertRaisesRegex(ValueError, 'graph crosses'):
            packet_rows(full, split, news, '2023-09-01')

    def test_label_basis_or_headline_mismatch_does_not_drop_the_row(self):
        for field, value in [('historical_availability_verified', True), ('text', 'rewritten'), ('exit_session_index', 111)]:
            full, split, news = packet_fixture();full[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                packet_rows(full, split, news, '2023-09-01')

    def test_target_scaler_has_no_test_inputs_and_constant_scale_is_one(self):
        self.assertEqual(target_scale([{'target': 2.}, {'target': 4.}]), (3., 1.))
        self.assertEqual(target_scale([{'target': 2.}, {'target': 2.}]), (2., 1.))
        with self.assertRaises(ValueError):target_scale([{'target': float('nan')}])

    def test_finite_wall_rss_and_memory_budgets(self):
        ticks = iter([0., 2.])
        b = Budget(1, 1000, 0, clock=lambda: next(ticks), resources=lambda: (1, 1000))
        with self.assertRaisesRegex(BudgetStop, 'Wall-clock'):b.check()
        with self.assertRaisesRegex(BudgetStop, 'RSS'):
            Budget(1, 10, 0, clock=lambda: 0, resources=lambda: (11, 1000)).check()
        with self.assertRaisesRegex(BudgetStop, 'memory'):
            Budget(1, 1000, 10, clock=lambda: 0, resources=lambda: (1, 9)).check()
        for invalid in [float('nan'), float('inf'), True]:
            with self.subTest(value=invalid), self.assertRaises(ValueError):Budget(invalid, 100, 0)

    def test_existing_outputs_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d);(p / 'keep').write_text('unchanged')
            with self.assertRaises(FileExistsError):new_output(p)
            self.assertEqual((p / 'keep').read_text(), 'unchanged')


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Existing PyTorch required; never install in tests')
class GradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def test_only_last_transformer_block_and_head_update(self):
        model, rows, tokens = tiny_fixture(610);model.train()
        result = train_synthetic_epoch(model, rows, tokens, CONFIG, unlimited_fixture_budget(), 610)
        self.assertEqual(result['status'], 'complete');self.assertFalse(model.training)
        self.assertTrue(any(n.startswith('backbone.encoder.layer.1.') for n in result['changed_parameters']))
        self.assertTrue(any(n.startswith('head.') for n in result['changed_parameters']))
        self.assertEqual(result['frozen_parameters_changed'], [])
        self.assertTrue(all(n.startswith('backbone.encoder.layer.1.') or n.startswith('head.') for n in result['selected_parameters']))

    def test_same_seed_replays_exact_parameter_updates_and_group_losses(self):
        results = []
        for _ in range(2):
            model, rows, tokens = tiny_fixture(610)
            results.append(train_synthetic_epoch(model, rows, tokens, CONFIG, unlimited_fixture_budget(), 610))
        for key in ['parameter_hashes_before', 'parameter_hashes_after', 'steps', 'target_mean', 'target_scale']:
            self.assertEqual(results[0][key], results[1][key], key)

    def test_one_step_per_company_day_instead_of_per_headline(self):
        model, rows, tokens = tiny_fixture(610)
        result = train_synthetic_epoch(model, rows, tokens, CONFIG, unlimited_fixture_budget(), 610)
        self.assertEqual(result['completed_steps'], 4)
        self.assertEqual(sum(len(r['headlines']) for r in rows), 7)
        self.assertEqual(len({x['sample_id'] for x in result['steps']}), 4)
        self.assertTrue(result['synthetic_only']);self.assertIsNone(result['market_predictions'])
        self.assertFalse(result['test_outcomes_read'])

    def test_loss_of_bag_mean_is_not_average_of_headline_losses(self):
        import torch
        class NumberModel(torch.nn.Module):
            def forward(self, batch):return batch['input_ids'][:, 0].float()
        titles = ['zero', 'two'];tokens = {}
        for t, value in zip(titles, [0, 2]):
            tokens[hashlib.sha256(t.encode()).hexdigest()] = {
                'input_ids': [value], 'attention_mask': [1], 'token_type_ids': [0]}
        values = [bag_prediction(NumberModel(), titles, tokens, n, unlimited_fixture_budget()) for n in [1, 8]]
        self.assertEqual(float(values[0]), 1.);self.assertEqual(float(values[1]), 1.)
        self.assertEqual(float((values[0] - 1.) ** 2), 0.)
        self.assertEqual(sum((v - 1.) ** 2 for v in [0., 2.]) / 2., 1.)

    def test_budget_exhaustion_has_no_fake_complete_result(self):
        model, rows, tokens = tiny_fixture(610)
        b = Budget(1, 1000, 0, clock=lambda: 0, resources=lambda: (1001, 1000))
        result = train_synthetic_epoch(model, rows, tokens, CONFIG, b, 610)
        self.assertEqual(result['status'], 'partial');self.assertEqual(result['completed_steps'], 0)
        self.assertEqual(result['changed_parameters'], []);self.assertIsNone(result['market_predictions'])

    def test_acceptance_engine_rejects_historical_market_rows(self):
        model, rows, tokens = tiny_fixture(610);rows[0]['sample_id'] = 'MSFT:2023-04-03'
        with self.assertRaisesRegex(ValueError, 'only accepts synthetic'):
            train_synthetic_epoch(model, rows, tokens, CONFIG, unlimited_fixture_budget(), 610)


if __name__ == '__main__':unittest.main()
