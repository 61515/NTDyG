import sys
from pathlib import Path
_TEST_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_TEST_CODE))
sys.path.insert(0, str(_TEST_CODE / "ntdyg"))
"""Check that faster NumPy indexing preserves CAWN outputs and gradients."""
import copy
import unittest
import numpy as np
import torch

from run_legacy_temporal_baseline import LegacyEventClassifier
from cawn_position_encoder import IndexedPositionEncoder
from test_legacy_temporal_baseline import tiny_data


class PositionIndexingTests(unittest.TestCase):
    def test_original_outputs_and_gradients_with_repeated_nodes_and_padding(self):
        torch.manual_seed(31)
        original = LegacyEventClassifier("CAWN", tiny_data()).eval()
        indexed = copy.deepcopy(original)
        indexed.encoder.position_encoder.__class__ = IndexedPositionEncoder
        indexed.eval()
        for order in (np.array([6, 2, 5, 3]), np.array([0, 1, 7])):
            data = tiny_data()
            expected = original.forward_events(data, order)
            actual = indexed.forward_events(data, order)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            original.zero_grad()
            indexed.zero_grad()
            expected.sum().backward()
            actual.sum().backward()
            for (name, p), (other_name, q) in zip(original.named_parameters(), indexed.named_parameters()):
                self.assertEqual(name, other_name)
                if p.grad is not None:
                    torch.testing.assert_close(q.grad, p.grad, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
