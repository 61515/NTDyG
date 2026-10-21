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

from make_chronological_splits import make_ordered_splits


def write_csv(path: Path, rows: List[List[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def read_indices(path: Path) -> List[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [int(row[0]) for row in csv.reader(handle)]


class MakeOrderedSplitsTests(unittest.TestCase):
    def test_splits_are_disjoint_ordered_and_keep_equal_keys_together(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "events.csv"
            write_csv(
                source,
                [
                    ["time", "label"],
                    ["1", "0"],
                    ["2", "0"],
                    ["2.0", "1"],
                    ["3", "0"],
                    ["4", "1"],
                    ["5", "0"],
                    ["6", "0"],
                ],
            )

            output = root / "splits"
            report = make_ordered_splits(source, output, "time", "label", 0.5, 0.25)
            train = read_indices(output / "train_indices.csv")
            validation = read_indices(output / "validation_indices.csv")
            test = read_indices(output / "test_indices.csv")

            self.assertEqual(set(train) | set(validation) | set(test), set(range(7)))
            self.assertFalse(set(train) & set(validation))
            self.assertFalse(set(train) & set(test))
            self.assertFalse(set(validation) & set(test))
            self.assertEqual(len(set(train) | set(validation) | set(test)), 7)
            self.assertTrue(report["strict_order_key_order"])
            self.assertEqual(report["splits"]["train"]["order_key_max"], "2")
            split_for_index = {
                index: split_name
                for split_name, indices in {
                    "train": train,
                    "validation": validation,
                    "test": test,
                }.items()
                for index in indices
            }
            self.assertEqual(split_for_index[1], split_for_index[2])
            self.assertIn("physical-time meaning", report["order_key_semantics"])
            self.assertTrue((output / "split_summary.json").is_file())

    def test_rejects_input_not_sorted_by_order_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "events.csv"
            write_csv(source, [["time", "label"], ["1", "0"], ["3", "1"], ["2", "0"]])

            with self.assertRaisesRegex(ValueError, "not ordered by time"):
                make_ordered_splits(source, root / "splits", "time", "label")


if __name__ == "__main__":
    unittest.main()
