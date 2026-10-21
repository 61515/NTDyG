import sys
from pathlib import Path
_TEST_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_TEST_CODE))
sys.path.insert(0, str(_TEST_CODE / "ntdyg"))
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "ntdyg"))
from memory import train_model


class MemoryValidationSelectionTests(unittest.TestCase):
    def test_restores_best_validation_checkpoint_and_stops_after_patience(self):
        torch.manual_seed(23)
        cfg = SimpleNamespace(
            device="cpu",
            memory_dim=2,
            message_dim=2,
            hidden_dim=3,
            classifier_hidden_dim=3,
            lr=0.01,
            weight_decay=0.0,
            num_negatives=1,
            epochs=5,
            time_col="order_key",
            label_col="label",
        )
        events = pd.DataFrame(
            {
                "src_id": [0, 1, 0],
                "dst_id": [1, 2, 2],
                "order_key": [1, 2, 3],
                "label": [0, 0, 0],
                "feature": [0.1, 0.2, 0.3],
            }
        )
        validation_scores = [0.3, 0.7, 0.5, 0.4]
        checkpoints = {}

        def validate(model, epoch):
            checkpoints[epoch] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            return validation_scores[epoch - 1]

        model = train_model(
            cfg,
            events,
            ["feature"],
            num_nodes=3,
            validation_callback=validate,
            validation_patience=2,
        )

        self.assertEqual(model.best_epoch, 2)
        self.assertEqual(len(model.validation_history), 4)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value.cpu(), checkpoints[2][name]))


if __name__ == "__main__":
    unittest.main()
