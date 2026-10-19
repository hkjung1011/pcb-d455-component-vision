"""Validate metadata and split groups; never open image or annotation payloads."""

import argparse
import csv
import hashlib
from pathlib import Path, PurePosixPath
import re


COLUMNS = (
    "source_id", "group_id", "image_id", "relative_path", "image_sha256",
    "annotation_sha256", "label_status", "license_id", "license_review",
)
SPLIT_SEED = "pcb-ic-ts-v01"


def split_group(group_id):
    digest = hashlib.sha256(f"{SPLIT_SEED}|{group_id}".encode("utf-8")).digest()
    score = int.from_bytes(digest, "big") * 100
    denominator = 1 << 256
    return "train" if score < 70 * denominator else "val" if score < 85 * denominator else "test"


def prepare(rows):
    if not rows:
        raise ValueError("Manifest has no rows; a header-only template is not data.")
    ids, paths, hashes = set(), set(), set()
    result = []
    for row in rows:
        if set(row) != set(COLUMNS) or any(not isinstance(row[k], str) or not row[k].strip() for k in COLUMNS):
            raise ValueError("Every required column must contain a non-empty string.")
        for key in ("source_id", "group_id", "image_id", "license_id"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", row[key]):
                raise ValueError("IDs must use non-identifying ASCII letters, digits, dots, colons, underscores or hyphens.")
        path = PurePosixPath(row["relative_path"])
        if path.is_absolute() or ".." in path.parts or "\\" in row["relative_path"] or ":" in row["relative_path"] or str(path) != row["relative_path"] or str(path) == ".":
            raise ValueError("relative_path must be a normalized relative POSIX path without traversal.")
        for key in ("image_sha256", "annotation_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise ValueError("SHA256 values must contain 64 lowercase hexadecimal characters.")
        if row["label_status"] != "complete_ic_bbox" or row["license_review"] != "local_use_verified":
            raise ValueError("Complete IC labels and verified local-use rights are required before splitting.")
        identity = (row["source_id"], row["image_id"])
        source_path = (row["source_id"], row["relative_path"])
        if identity in ids or source_path in paths or row["image_sha256"] in hashes:
            raise ValueError("Duplicate image ID, source path or image hash; deduplicate and reconcile groups first.")
        ids.add(identity)
        paths.add(source_path)
        hashes.add(row["image_sha256"])
        result.append({**row, "split": split_group(row["group_id"])})
    if {row["split"] for row in result} != {"train", "val", "test"}:
        raise ValueError("At least one split is empty; revise the data collection plan, not the seed after viewing test performance.")
    return sorted(result, key=lambda row: (row["source_id"], row["image_id"]))


def write_manifest(input_path, output_path):
    with Path(input_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError("CSV header must exactly match the published template.")
        rows = prepare(list(reader))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves prior manifests and their split assignments.
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(*COLUMNS, "split"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        rows = write_manifest(args.input, args.output)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(2, f"Manifest rejected: {error}\n")
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
    print(f"Metadata only: rows={len(rows)}, split_counts={counts}. Payloads and rights were not independently verified.")


if __name__ == "__main__":
    main()
