import sys
from pathlib import Path
_TEST_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_TEST_CODE))
sys.path.insert(0, str(_TEST_CODE / "ntdyg"))
"""Check legacy alignment and shuffled-query support before queued training."""
import copy
import unittest

import numpy as np
import torch

from run_legacy_temporal_baseline import (
    MODELS, LegacyEventClassifier, NeighborSampler, import_legacy, loss_function,
)


def tiny_data():
    rng = np.random.default_rng(341)
    src = np.array([1, 2, 1, 3, 2, 1, 3, 1], dtype=np.int64)
    dst = np.array([2, 3, 3, 1, 1, 2, 2, 3], dtype=np.int64)
    times = np.arange(1., 9.)
    features = rng.normal(size=(8, 6)).astype(np.float32)
    adjacency = [[] for _ in range(4)]
    for i, (s, d, t) in enumerate(zip(src, dst, times)):
        adjacency[s].append((int(d), i+1, t))
        adjacency[d].append((int(s), i+1, t))
    sampler = NeighborSampler(adjacency, "recent")
    return dict(dataset="IIoTAPT", src=src, dst=dst, times=times,
                node_features=rng.normal(size=(4, 16)).astype(np.float32),
                edge_features=np.vstack([np.zeros((1, 6), np.float32), features]),
                features=features, columns=list(range(6)), samplers=dict(train=sampler, test=sampler))


class LegacyAlignmentTests(unittest.TestCase):
    def test_same_original_normal_only_loss_and_gradients(self):
        logits = torch.tensor([[.1, .2], [.3, -.4], [-.2, .5], [1., -.1], [.2, .1]], requires_grad=True)
        labels = torch.zeros(5, dtype=torch.long)
        original = import_legacy("utils")
        expected = .5 * torch.nn.functional.cross_entropy(logits, labels) + .5 * original.bpr_loss_binary_normal_only(
            logits, labels, torch.arange(5), intra_margin=.05, pseudo_margin=.10, lambda_pseudo=.20, hard_ratio=.20)
        torch.testing.assert_close(loss_function(logits, labels), expected, rtol=0, atol=0)
        loss_function(logits, labels).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_shuffled_queries_preserve_eval_scores_and_train_all_encoders(self):
        for name in MODELS:
            with self.subTest(model=name):
                torch.manual_seed(31)
                data = tiny_data()
                model = LegacyEventClassifier(name, data)
                order = np.array([6, 2, 5, 3])
                model.eval()
                with torch.no_grad():
                    ordered = model.forward_events(data, np.sort(order))
                    shuffled = model.forward_events(data, order)
                torch.testing.assert_close(shuffled[np.argsort(order)], ordered)
                model.train()
                loss = loss_function(model.forward_events(data, order), torch.zeros(4, dtype=torch.long))
                loss.backward()
                encoder_grad = [p.grad for p in model.encoder.parameters() if p.requires_grad and p.grad is not None]
                self.assertTrue(encoder_grad)
                self.assertTrue(all(torch.isfinite(g).all() for g in encoder_grad))

    def test_future_feature_changes_do_not_change_earlier_event(self):
        for name in MODELS:
            with self.subTest(model=name):
                data = tiny_data()
                changed = copy.deepcopy(data)
                changed["features"][5:] += 1000
                changed["edge_features"][6:] += 1000
                torch.manual_seed(31)
                model = LegacyEventClassifier(name, data).eval()
                torch.manual_seed(31)
                other = LegacyEventClassifier(name, changed).eval()
                with torch.no_grad():
                    expected = model.forward_events(data, np.array([4]))
                    actual = other.forward_events(changed, np.array([4]))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
