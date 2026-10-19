"""Temporary metadata fixtures only: no PCB images, labels or experiment results."""

import csv
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "prepare_manifest.py"
SPEC = importlib.util.spec_from_file_location("prepare_manifest", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_rows():
    return [{
        "source_id": "unit-only", "group_id": f"group-{i // 2}", "image_id": f"row-{i}",
        "relative_path": f"metadata-only/{i}.jpg",
        "image_sha256": hashlib.sha256(f"unit-image-{i}".encode()).hexdigest(),
        "annotation_sha256": hashlib.sha256(f"unit-annotation-{i}".encode()).hexdigest(),
        "label_status": "complete_ic_bbox", "license_id": "unit-test-no-images",
        "license_review": "local_use_verified",
    } for i in range(120)]


class ManifestTests(unittest.TestCase):
    def test_order_invariant_and_group_disjoint(self):
        rows = fixture_rows()
        actual = MODULE.prepare(rows)
        self.assertEqual(actual, MODULE.prepare(list(reversed(rows))))
        groups = {}
        for row in actual:
            groups.setdefault(row["group_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual({row["split"] for row in actual}, {"train", "val", "test"})

    def test_duplicate_payload_hash_rejected(self):
        rows = fixture_rows()
        rows[1]["image_sha256"] = rows[0]["image_sha256"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            MODULE.prepare(rows)

    def test_invalid_or_unreviewed_metadata_rejected(self):
        for field, bad in (("relative_path", "../private.jpg"), ("relative_path", "C:/private.jpg"),
                           ("image_sha256", "bad"), ("label_status", "partial"),
                           ("license_review", "unknown"), ("group_id", "")):
            with self.subTest(field=field, value=bad):
                rows = fixture_rows()
                rows[0][field] = bad
                with self.assertRaises(ValueError):
                    MODULE.prepare(rows)

    def test_empty_or_missing_split_rejected(self):
        for rows in ([], fixture_rows()[:1]):
            with self.assertRaises(ValueError):
                MODULE.prepare(rows)

    def test_existing_manifest_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.csv", Path(directory) / "output.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=MODULE.COLUMNS)
                writer.writeheader()
                writer.writerows(fixture_rows())
            MODULE.write_manifest(source, output)
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                MODULE.write_manifest(source, output)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
