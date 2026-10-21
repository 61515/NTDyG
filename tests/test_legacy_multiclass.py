import sys
from pathlib import Path
_TEST_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_TEST_CODE))
sys.path.insert(0, str(_TEST_CODE / "ntdyg"))
"""Check fine-label training, ranking gradients, and declared metric meanings."""
import unittest

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from run_legacy_multiclass import draws, loss_function, metrics_from_logits
from run_legacy_temporal_baseline import MODELS, LegacyEventClassifier
from test_legacy_temporal_baseline import tiny_data


class MulticlassChecks(unittest.TestCase):
    def test_two_class_ranking_reduces_to_cross_entropy(self):
        # With two classes every negative is the other class; BPR equals CE.
        logits = torch.tensor([[.4, -.7], [-.2, .6], [.1, .3]], requires_grad=True)
        labels = torch.tensor([0, 1, 0])
        actual = loss_function(logits, labels)
        expected = torch.nn.functional.cross_entropy(logits, labels)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        gradient = logits.grad.clone()
        logits.grad = None
        expected.backward()
        torch.testing.assert_close(gradient, logits.grad)

    def test_metrics_are_macro_not_micro_and_support_raw_logits(self):
        labels = np.array([0, 0, 0, 1, 1, 2])
        logits = np.array([[3, 2, 1], [1, 2, 0], [2, 1, 0], [0, 3, 2], [2, 1, 0], [1, 2, 3.]])
        actual = metrics_from_logits(labels, logits)
        per_auc = [roc_auc_score(labels == i, logits[:, i]) for i in range(3)]
        per_ap = [average_precision_score(labels == i, logits[:, i]) for i in range(3)]
        self.assertAlmostEqual(actual["roc_auc"], np.mean(per_auc))
        self.assertAlmostEqual(actual["average_precision"], np.mean(per_ap))
        self.assertNotAlmostEqual(actual["roc_auc"], actual["micro_ovr_auc"])

    def test_binary_AP_is_positive_class_not_macro(self):
        labels = np.array([0, 0, 0, 1])
        logits = np.array([[5, 1], [1, 3], [2, 2], [3, 4.]])
        actual = metrics_from_logits(labels, logits)
        self.assertEqual(actual["average_precision"], average_precision_score(labels, logits[:, 1]))
        self.assertNotEqual(actual["average_precision"], actual["macro_ovr_ap"])

    def test_draws_nested_reproducible_unique_and_fixed_size(self):
        for seed in (95545, 95546, 95547, 95548):
            first, second = draws(140987, seed), draws(140987, seed)
            for rate in (1, 5):
                np.testing.assert_array_equal(first[rate], second[rate])
                self.assertEqual(len(first[rate]), int(140987 * rate / 100))
                self.assertEqual(len(np.unique(first[rate])), len(first[rate]))
            np.testing.assert_array_equal(first[1], first[5][:len(first[1])])

    def test_all_encoders_learn_fine_labels_above_binary_range(self):
        for name in MODELS:
            for classes in (8, 26):
                with self.subTest(model=name, classes=classes):
                    torch.manual_seed(51)
                    data = tiny_data()
                    data["num_classes"] = classes
                    model = LegacyEventClassifier(name, data, cawn_indexed_position=(name == "CAWN"))
                    logits = model.forward_events(data, np.array([3, 4, 5, 6]))
                    self.assertEqual(tuple(logits.shape), (4, classes))
                    labels = torch.tensor([classes - 1, classes - 2, 1, 0])
                    loss = loss_function(logits, labels)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
                    self.assertTrue(grads)
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                    self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
