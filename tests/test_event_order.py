import sys
from pathlib import Path
_TEST_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_TEST_CODE))
sys.path.insert(0, str(_TEST_CODE / "ntdyg"))
import pandas as pd
import unittest

from ntdyg.event_order import select_normal_events, sort_event_frame


class EventOrderTests(unittest.TestCase):
    def test_sort_event_frame_uses_source_row_for_order_key_ties(self):
        events = pd.DataFrame(
            {
                "time": [2, 1, 1, 1],
                "__source_row": [4, 8, 2, 5],
                "event": ["later", "row-8", "row-2", "row-5"],
            }
        )

        ordered = sort_event_frame(events, "time")

        self.assertEqual(ordered["event"].tolist(), ["row-2", "row-5", "row-8", "later"])

    def test_sort_event_frame_preserves_input_order_without_source_rows(self):
        events = pd.DataFrame({"time": [1, 1, 1], "event": ["a", "b", "c"]})

        ordered = sort_event_frame(events, "time")

        self.assertEqual(ordered["event"].tolist(), ["a", "b", "c"])

    def test_select_normal_events_excludes_anomalous_training_rows(self):
        events = pd.DataFrame({"label": [0, 1, 0], "event": ["n1", "a1", "n2"]})

        selected = select_normal_events(events, "label")

        self.assertEqual(selected["event"].tolist(), ["n1", "n2"])


if __name__ == "__main__":
    unittest.main()
