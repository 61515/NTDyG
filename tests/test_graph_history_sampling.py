import sys
from pathlib import Path
_TEST_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_TEST_CODE))
sys.path.insert(0, str(_TEST_CODE / "ntdyg"))
import unittest
import sys

sys.argv = [sys.argv[0]]
from rawdatasets_pre import TemporalBatchGraphSample


class GraphHistorySamplingTests(unittest.TestCase):
    def make_sampler(self, max_neighbors):
        sampler = object.__new__(TemporalBatchGraphSample)
        sampler.k_hop = 1
        sampler.n_degree = max_neighbors
        sampler.reverse_edge_rows = {9: [0, 1, 2, 3, 4, 5]}
        sampler.reverse_edge_srcs = {9: [1, 2, 3, 4, 5, 6]}
        return sampler

    def test_uses_only_most_recent_causal_neighbors(self):
        sampler = self.make_sampler(2)
        edges = sampler._get_k_hop_edge_indices(global_id=4, target_src=9)
        self.assertEqual(edges, {3, 4})

    def test_unbounded_mode_keeps_all_causal_neighbors(self):
        sampler = self.make_sampler(None)
        edges = sampler._get_k_hop_edge_indices(global_id=3, target_src=9)
        self.assertEqual(edges, {0, 1, 2, 3})

    def test_current_edge_aggregates_only_earlier_causal_edges(self):
        sampler = object.__new__(TemporalBatchGraphSample)
        adjacency = sampler._create_edge_adj([(0, 1), (1, 2), (2, 3)]).toarray()

        self.assertEqual(adjacency[2, 1], 1.0)
        self.assertEqual(adjacency[2, 0], 0.0)
        self.assertEqual(adjacency[0, 2], 0.0)
        self.assertEqual(adjacency[1, 0], 1.0)
        self.assertEqual(adjacency[0, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
