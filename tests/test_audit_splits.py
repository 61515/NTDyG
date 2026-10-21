import sys
from pathlib import Path
_TEST_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_TEST_CODE))
sys.path.insert(0, str(_TEST_CODE / "ntdyg"))
import csv
import tempfile
import unittest
from pathlib import Path
from typing import List

from audit_splits import audit_splits


def write_csv(path: Path, rows: List[List[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


class AuditSplitsTests(unittest.TestCase):
    def test_disjoint_ordered_splits_report_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "events.csv"
            write_csv(
                csv_path,
                [
                    ["time", "label"],
                    ["1", "0"],
                    ["2", "1"],
                    ["3", "0"],
                    ["4", "0"],
                    ["5", "1"],
                    ["6", "0"],
                ],
            )
            split_paths = {}
            for name, indices in {
                "train": [0, 1],
                "validation": [2, 3],
                "test": [4, 5],
            }.items():
                path = root / f"{name}.csv"
                write_csv(path, [[str(index)] for index in indices])
                split_paths[name] = path

            report = audit_splits(csv_path, split_paths, "time", "label")

            self.assertEqual(report["splits"]["train"]["anomalous"], 1)
            self.assertEqual(report["splits"]["test"]["events"], 2)
            self.assertTrue(report["strict_order_key_order"])
            self.assertEqual(sum(report["pairwise_overlap_counts"].values()), 0)
            self.assertEqual(report["unindexed_source_rows"], 0)

    def test_overlap_is_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "events.csv"
            write_csv(
                csv_path,
                [["time", "label"], ["1", "0"], ["2", "1"], ["3", "0"]],
            )
            split_paths = {}
            for name, indices in {"train": [0, 1], "test": [1, 2]}.items():
                path = root / f"{name}.csv"
                write_csv(path, [[str(index)] for index in indices])
                split_paths[name] = path

            report = audit_splits(csv_path, split_paths, "time", "label")

            self.assertEqual(report["pairwise_overlap_counts"]["train__test"], 1)
            self.assertFalse(report["strict_order_key_order"])


if __name__ == "__main__":
    unittest.main()
