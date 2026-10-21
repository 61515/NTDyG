import sys
from pathlib import Path
_TEST_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_TEST_CODE))
sys.path.insert(0, str(_TEST_CODE / "ntdyg"))
import sys
import unittest

import torch

sys.argv = [sys.argv[0]]
from memory import NodeMemoryModel


class MemoryVectorStateTests(unittest.TestCase):
    def test_vector_update_matches_full_matrix_equations(self):
        torch.manual_seed(17)
        model = NodeMemoryModel(
            num_nodes=3,
            feature_dim=2,
            memory_dim=4,
            message_dim=3,
            hidden_dim=5,
            classifier_hidden_dim=6,
            device="cpu",
        )
        memory, last_update = model.init_memory()
        memory[0] = torch.tensor([0.1, -0.2, 0.3, 0.4])
        memory[1] = torch.tensor([-0.3, 0.5, 0.2, -0.1])
        last_update[0] = 1.25
        last_update[1] = 2.5
        event = torch.tensor([0.7, -0.6])
        timestamp = 3.75

        actual = model.compute_updated_states_from_vectors(
            memory[0], memory[1], timestamp, event, last_update[0], last_update[1]
        )

        src_delta = torch.tensor([timestamp]) - last_update[0].unsqueeze(0)
        dst_delta = torch.tensor([timestamp]) - last_update[1].unsqueeze(0)
        src_msg = model.build_message(memory[0], memory[1], event, src_delta)
        dst_msg = model.build_message(memory[1], memory[0], event, dst_delta)
        src_new = model.memory_updater(src_msg.unsqueeze(0), memory[0].unsqueeze(0)).squeeze(0)
        dst_new = model.memory_updater(dst_msg.unsqueeze(0), memory[1].unsqueeze(0)).squeeze(0)
        logit = model.score_event(src_new, dst_new, event)
        expected = (src_new, dst_new, src_delta, dst_delta, logit)

        for actual_tensor, expected_tensor in zip(actual, expected):
            self.assertTrue(torch.allclose(actual_tensor, expected_tensor))


if __name__ == "__main__":
    unittest.main()
