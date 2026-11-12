"""Pure CPU tensor acceptance: no model, optimizer, download or market prediction."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from grouped_loss import TrainingTargets, validate_batch, full_mse, chunked_mse, synthetic_fixture

torch.set_num_threads(2)


class GroupedLossTests(unittest.TestCase):
    def setUp(self):
        self.batch, self.targets, self.rows = synthetic_fixture()
        self.ids = self.batch['sample_ids']
        self.n = self.batch['group_offsets'][-1]

    def predictions(self, dtype=torch.float64):
        return torch.linspace(-.8, 1.2, self.n, dtype=dtype).requires_grad_()

    def pieces(self, p, size=8):
        return [(i, p[i:i+size]) for i in range(0, len(p), size)]

    def test_day_means_then_mse_equal_day_not_headline_weight(self):
        p = self.predictions();loss, means = full_mse(p, self.batch, self.targets, self.ids)
        manual = torch.stack([p[0], p[1:4].mean(), p[4:].mean()])
        y = p.new_tensor([(r['target'] - self.targets.mean)/self.targets.scale for r in self.rows])
        torch.testing.assert_close(means, manual)
        torch.testing.assert_close(loss, ((manual-y)**2).sum()/3)
        replicated = torch.repeat_interleave(y, torch.tensor([1, 3, 11]))
        self.assertGreater(abs(float(loss - ((p-replicated)**2).mean())), .01)
        loss.backward()
        for g, (a, b) in enumerate(zip([0, 1, 4], [1, 4, 15])):
            expected = 2*(manual[g]-y[g])/3/(b-a)
            torch.testing.assert_close(p.grad[a:b], expected.expand(b-a))

    def test_full_vs_uneven_chunks_leaf_loss_and_gradients(self):
        p = self.predictions();ref, means = full_mse(p, self.batch, self.targets, self.ids)
        ref.backward();grad = p.grad.clone()
        for size in [8, 4, 1, 15, 20]:
            q = self.predictions();loss, current = chunked_mse(self.pieces(q, size), self.batch, self.targets, self.ids)
            loss.backward()
            with self.subTest(size=size):
                torch.testing.assert_close(loss, ref, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(current, means, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(q.grad, grad, rtol=1e-12, atol=1e-12)

    def test_shared_parameter_gradients_float64_and_float32(self):
        for dtype, tol in [(torch.float64, 1e-12), (torch.float32, 2e-6)]:
            results = []
            for size in [None, 8, 4, 1]:
                x = self.predictions(dtype).detach();w = torch.tensor([.3, -.2], dtype=dtype, requires_grad=True)
                pred = x*w[0] + w[1]
                loss, _ = (full_mse(pred, self.batch, self.targets, self.ids) if size is None else
                    chunked_mse(self.pieces(pred, size), self.batch, self.targets, self.ids))
                loss.backward();results.append((loss.detach(), w.grad.clone()))
            for loss, grad in results[1:]:
                torch.testing.assert_close(loss, results[0][0], rtol=tol, atol=tol)
                torch.testing.assert_close(grad, results[0][1], rtol=tol, atol=tol)

    def test_train_only_population_scale_independent_of_source_order(self):
        targets = TrainingTargets(list(reversed(self.rows)), {'synthetic:heldout'})
        self.assertAlmostEqual(targets.mean, 0.)
        self.assertAlmostEqual(targets.scale, ((.01**2+.02**2+.03**2)/3)**.5)
        a = full_mse(self.predictions(), self.batch, targets, self.ids)[0]
        b = full_mse(self.predictions(), self.batch, self.targets, self.ids)[0]
        torch.testing.assert_close(a, b)

    def test_constant_training_targets_use_existing_scale_one_policy(self):
        rows = deepcopy(self.rows)
        for row in rows:row['target'] = .05
        t = TrainingTargets(rows, set())
        self.assertEqual(t.scale, 1.);self.assertAlmostEqual(t.mean, .05)

    def test_duplicate_missing_extra_or_reordered_ids_rejected(self):
        bad = [self.ids[:-1], self.ids + ['synthetic:extra'], list(reversed(self.ids)), [self.ids[0]]*3]
        for ids in bad:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                full_mse(self.predictions(), self.batch, self.targets, ids)
        b = deepcopy(self.batch);b['sample_ids'][1] = b['sample_ids'][0]
        with self.assertRaises(ValueError):validate_batch(b, self.targets, b['sample_ids'])

    def test_original_training_label_values_and_label_order_required(self):
        for field, value in [('labels', list(reversed(self.batch['labels']))),
                             ('label_sample_ids', list(reversed(self.ids))), ('labels', [True, .02, -.03])]:
            b = deepcopy(self.batch);b[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):validate_batch(b, self.targets, self.ids)

    def test_missing_duplicate_or_empty_group_boundaries_rejected(self):
        for offsets in [[0, 1, 15], [0, 1, 1, 15], [0, 1, 4, 14], [1, 1, 4, 15], [0, 4, 1, 15]]:
            b = deepcopy(self.batch);b['group_offsets'] = offsets
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):validate_batch(b, self.targets, self.ids)

    def test_wrong_group_indices_weights_or_headline_id_rejected(self):
        for field, value in [('headline_to_group', 2), ('headline_weights', .5), ('headline_hashes', '0'*64)]:
            b = deepcopy(self.batch);b[field][0] = value
            with self.subTest(field=field), self.assertRaises(ValueError):validate_batch(b, self.targets, self.ids)

    def test_missing_duplicate_reordered_empty_and_mixed_dtype_chunks_rejected(self):
        p = self.predictions();cases = [[], [(1,p)], [(0,p[:8])], [(0,p[:8]),(7,p[8:])],
            [(0,p[:8]),(8,p[8:].float())], [(0,p[:0]),(0,p)], [(0,p),(15,p[:1])]]
        for pieces in cases:
            with self.subTest(starts=[x[0] for x in pieces]), self.assertRaises(ValueError):
                chunked_mse(pieces, self.batch, self.targets, self.ids)

    def test_prediction_shape_nonfinite_and_coverage_rejected(self):
        p = self.predictions()
        for q in [p[:-1], p[:,None], torch.tensor([]), p.long(), p*float('nan'), p*float('inf')]:
            with self.subTest(shape=tuple(q.shape)), self.assertRaises(ValueError):
                full_mse(q, self.batch, self.targets, self.ids)
        rows = deepcopy(self.rows);rows[0]['target'] = float('nan')
        with self.assertRaises(ValueError):TrainingTargets(rows, set())

    def test_right_padding_invariance_of_artificial_masked_features_and_gradients(self):
        batches = [deepcopy(self.batch), deepcopy(self.batch)]
        for key in ['input_ids', 'attention_mask', 'token_type_ids']:
            for row in batches[1][key]:row.extend([0]*5)
        batches[1]['padded_length'] += 5
        results = []
        for b in batches:
            mask = torch.tensor(b['attention_mask'], dtype=torch.float64)
            tokens = torch.tensor(b['input_ids'], dtype=torch.float64)
            x = (tokens*mask).sum(1)/mask.sum(1)/100
            w = torch.tensor([.3,-.2], dtype=torch.float64, requires_grad=True)
            loss, _ = full_mse(x*w[0]+w[1], b, self.targets, self.ids)
            loss.backward();results.append((loss.detach(), w.grad.clone()))
        for a,b in zip(*results):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)

    def test_padding_values_mask_holes_and_truncation_rejected(self):
        for field, index, value in [('input_ids', -1, 9), ('attention_mask', 1, 0), ('token_type_ids', 0, 1)]:
            b = deepcopy(self.batch);b[field][0][index] = value
            with self.subTest(field=field), self.assertRaises(ValueError):validate_batch(b, self.targets, self.ids)
        b = deepcopy(self.batch);b['headline_lengths'][0] = 65
        with self.assertRaises(ValueError):validate_batch(b, self.targets, self.ids)

    def test_gradient_reset_replays_while_no_reset_accumulates(self):
        x = self.predictions().detach();w = torch.tensor([.3,-.2], dtype=torch.float64, requires_grad=True)
        def backward(size):
            p=x*w[0]+w[1]
            loss, _ = (full_mse(p,self.batch,self.targets,self.ids) if size is None else
                chunked_mse(self.pieces(p,size),self.batch,self.targets,self.ids))
            loss.backward()
        backward(None);first=w.grad.clone()
        w.grad=None;self.assertIsNone(w.grad);backward(8)
        torch.testing.assert_close(w.grad,first,rtol=1e-12,atol=1e-12)
        backward(8);torch.testing.assert_close(w.grad,first*2,rtol=1e-12,atol=1e-12)

    def test_heldout_unknown_and_feature_only_batches_cannot_train(self):
        with self.assertRaises(ValueError):TrainingTargets(self.rows, {self.ids[0]})
        for sid in ['synthetic:heldout','synthetic:unknown']:
            b = deepcopy(self.batch);b['sample_ids'][0] = sid
            with self.subTest(sid=sid), self.assertRaises(ValueError):validate_batch(b,self.targets,b['sample_ids'])
        b = deepcopy(self.batch);b.pop('labels');b.pop('label_sample_ids')
        with self.assertRaises(ValueError):validate_batch(b,self.targets,self.ids)

    def test_loss_does_not_mutate_sources_targets_or_parameters(self):
        before = deepcopy(self.batch);rows = deepcopy(self.rows)
        p = self.predictions();values = p.detach().clone()
        full_mse(p,self.batch,self.targets,self.ids)[0].backward()
        self.assertEqual(self.batch,before);self.assertEqual(self.rows,rows)
        torch.testing.assert_close(p.detach(),values)


if __name__ == '__main__':unittest.main()
