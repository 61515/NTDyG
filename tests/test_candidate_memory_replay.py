import sys
from pathlib import Path
_TEST_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_TEST_CODE))
sys.path.insert(0, str(_TEST_CODE / "ntdyg"))
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR / "ntdyg"))
sys.path.insert(0, str(CODE_DIR))
import run_candidate_experiment as candidate
from memory import NodeMemoryModel


class CandidateMemoryReplayTests(unittest.TestCase):
    def test_validation_replay_scores_match_classifier_feature_replay(self):
        torch.manual_seed(31)
        cfg = SimpleNamespace(device="cpu", memory_dim=2, time_col="Timestamp_Label")
        model = NodeMemoryModel(
            num_nodes=3,
            feature_dim=1,
            memory_dim=2,
            message_dim=2,
            hidden_dim=3,
            classifier_hidden_dim=3,
            device="cpu",
        )
        events = pd.DataFrame(
            {
                "event_index": [0, 1, 2],
                "label": [0, 1, 0],
                "Label_encoded": [0, 1, 0],
                "order_key": [1.0, 2.0, 3.0],
                "Timestamp_Label": [1.0, 2.0, 3.0],
                "src_id": [0, 1, 0],
                "dst_id": [1, 2, 2],
                "feature": [0.1, 0.2, 0.3],
            }
        )

        feature_events, feature_state = candidate.replay_split(
            cfg, model, events, ["feature"]
        )
        validation_scores, validation_state = candidate.replay_memory_scores(
            cfg, model, events, ["feature"]
        )

        np.testing.assert_allclose(
            validation_scores,
            feature_events["gru_anomaly_score"].to_numpy(),
            rtol=1e-6,
            atol=1e-7,
        )
        self.assertTrue(torch.allclose(feature_state[0], validation_state[0]))
        self.assertTrue(torch.equal(feature_state[1], validation_state[1]))


if __name__ == "__main__":
    unittest.main()
