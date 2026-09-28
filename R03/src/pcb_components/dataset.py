"""Validate PCB instance labels and materialize a standard YOLO directory.

This module validates data contracts; it does not certify annotation accuracy.
Public proxy evaluation and D455 evaluation remain separate domains.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

SCHEMA = "pcb-instance-manifest-v1"
SPLITS = ("train", "val", "test")
DOMAINS = {"public_proxy", "d455"}
POLICIES = {"native_polygon", "weak_semantic_cc"}
STATUSES = {"reviewed", "source_provided", "weak"}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("must be a nonempty path string")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _key(path: Path) -> str:
    return os.path.normcase(str(path))


def _empty_report(path: Path) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "manifest": str(path),
        "valid": False,
        "errors": [],
        "warnings": [],
        "names": [],
        "domain": None,
        "annotation_policy": None,
        "provenance": {},
        "counts": {split: {"images": 0, "instances": 0} for split in SPLITS},
        "class_instance_counts_by_split": {split: {} for split in SPLITS},
        "class_image_counts_by_split": {split: {} for split in SPLITS},
        "split_presence": {split: False for split in SPLITS},
        "evaluation_eligible": False,
        "partial_evaluation": True,
        "evaluation_scope": "NOT_ELIGIBLE",
        "d455_evaluation_eligible": False,
        "image_dimensions_checked": False,
    }


def _polygon_class(line: str, class_count: int) -> int:
    parts = line.split()
    if len(parts) == 5:
        raise ValueError("5-token bbox label is not an instance polygon")
    if len(parts) < 7 or (len(parts) - 1) % 2:
        raise ValueError("polygon needs class ID and at least 3 x/y points")
    # Class IDs are text integers, not floats rounded or truncated to integers.
    if not parts[0].isascii() or not parts[0].isdigit():
        raise ValueError("class ID must be an exact nonnegative integer")
    class_id = int(parts[0])
    if not 0 <= class_id < class_count:
        raise ValueError(f"class ID {class_id} is outside names")
    try:
        values = [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError("polygon coordinates must be numeric") from exc
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        raise ValueError("polygon coordinates must be finite and normalized to [0,1]")
    points = list(zip(values[::2], values[1::2]))
    if len(set(points)) < 3:
        raise ValueError("polygon needs at least 3 distinct points")
    doubled_area = sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )
    if abs(doubled_area) <= 0:
        raise ValueError("polygon has zero area")
    return class_id


def validate_manifest(path: str | Path) -> dict[str, Any]:
    """Return a JSON-safe report, including errors instead of raising for bad data.

    Required provenance is ``provenance.source_url`` and
    ``provenance.class_mapping``. Root-level keys are accepted for compatibility.
    ``evaluation_eligible`` means a valid native-label validation split is present;
    test absence sets ``partial_evaluation``. It never promotes a public proxy to
    a D455 result. Class counts use canonical class names as dictionary keys.
    """
    manifest_path = Path(path).expanduser().resolve()
    report = _empty_report(manifest_path)
    errors: list[str] = report["errors"]
    warnings: list[str] = report["warnings"]
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f"Cannot read manifest: {exc}")
        return report
    if not isinstance(value, dict):
        errors.append("Manifest root must be an object")
        return report
    if value.get("schema") != SCHEMA:
        errors.append(f"schema must equal {SCHEMA}")
    names = value.get("names")
    if (not isinstance(names, list) or not names or
            not all(isinstance(name, str) and name.strip() == name and name for name in names) or
            len(set(names)) != len(names)):
        errors.append("names must be a nonempty ordered list of unique nonempty class names")
        names = []
    report["names"] = names
    for field, accepted in (("domain", DOMAINS), ("annotation_policy", POLICIES)):
        report[field] = value.get(field)
        if not isinstance(value.get(field), str) or value[field] not in accepted:
            errors.append(f"{field} must be one of {sorted(accepted)}")
    provenance = value.get("provenance", {})
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
        provenance = {}
    provenance = dict(provenance)
    for field in ("source_url", "class_mapping"):
        if field not in provenance and field in value:
            provenance[field] = value[field]
    if not isinstance(provenance.get("source_url"), str) or not provenance["source_url"].strip():
        errors.append("provenance.source_url must record the data source")
    if not isinstance(provenance.get("class_mapping"), dict) or not provenance["class_mapping"]:
        errors.append("provenance.class_mapping must record the source-to-canonical class mapping")
    report["provenance"] = provenance
    for split in SPLITS:
        report["class_instance_counts_by_split"][split] = {name: 0 for name in names}
        report["class_image_counts_by_split"][split] = {name: 0 for name in names}
    records = value.get("records")
    if not isinstance(records, list) or not records:
        errors.append("records must be a nonempty list")
        return report

    try:
        from PIL import Image
    except ImportError:
        Image = None
        warnings.append("Pillow unavailable: image decoding/dimensions were not checked")
    report["image_dimensions_checked"] = Image is not None
    seen_images: dict[str, int] = {}
    seen_labels: dict[str, int] = {}
    seen_hashes: dict[str, tuple[str, int]] = {}
    group_splits: dict[str, str] = {}
    weak_evaluation = False
    for index, record in enumerate(records):
        prefix = f"record {index}"
        if not isinstance(record, dict):
            errors.append(f"{prefix}: record must be an object")
            continue
        split = record.get("split")
        if not isinstance(split, str) or split not in SPLITS:
            errors.append(f"{prefix}: split must be train, val or test")
            continue
        report["counts"][split]["images"] += 1
        report["split_presence"][split] = True
        group = record.get("group_id")
        if not isinstance(group, str) or not group.strip():
            errors.append(f"{prefix}: group_id must identify the original physical board/group")
        elif group in group_splits and group_splits[group] != split:
            errors.append(f"{prefix}: group leakage for {group!r}: {group_splits[group]} and {split}")
        else:
            group_splits[group] = split
        status = record.get("annotation_status")
        if not isinstance(status, str) or status not in STATUSES:
            errors.append(f"{prefix}: annotation_status must be reviewed, source_provided or weak")
        is_weak = status == "weak" or value.get("annotation_policy") == "weak_semantic_cc"
        if is_weak and split != "train":
            weak_evaluation = True
            errors.append(f"{prefix}: weak labels are train-only; cannot use in {split}")
        paths: dict[str, Path] = {}
        for field, seen in (("image", seen_images), ("label", seen_labels)):
            try:
                resolved = _resolve(manifest_path.parent, record.get(field))
            except (ValueError, OSError) as exc:
                errors.append(f"{prefix}: {field} {exc}")
                continue
            key = _key(resolved)
            if key in seen:
                errors.append(f"{prefix}: duplicate {field} entry also used by record {seen[key]}")
            else:
                seen[key] = index
            if not resolved.is_file():
                errors.append(f"{prefix}: {field} is not a readable file: {resolved}")
            else:
                paths[field] = resolved
        image_path = paths.get("image")
        if image_path is not None:
            try:
                digest = _hash_file(image_path)
                if digest in seen_hashes:
                    previous_split, previous_index = seen_hashes[digest]
                    if previous_split != split:
                        errors.append(f"{prefix}: image SHA-256 leakage across {previous_split}/{split} (record {previous_index})")
                    else:
                        warnings.append(f"{prefix}: duplicate image content within {split} (record {previous_index})")
                else:
                    seen_hashes[digest] = (split, index)
                if Image is not None:
                    with Image.open(image_path) as decoded:
                        if decoded.width <= 0 or decoded.height <= 0:
                            raise ValueError("image dimensions must be positive")
                        decoded.verify()
            except (OSError, ValueError, SyntaxError) as exc:
                errors.append(f"{prefix}: image cannot be decoded/read: {exc}")
        label_path = paths.get("label")
        if label_path is not None:
            try:
                lines = label_path.read_text(encoding="utf-8-sig").splitlines()
            except (OSError, UnicodeError) as exc:
                errors.append(f"{prefix}: label cannot be read: {exc}")
                continue
            present_classes = set()
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    class_id = _polygon_class(line, len(names))
                except ValueError as exc:
                    errors.append(f"{prefix}, label line {line_number}: {exc}")
                    continue
                present_classes.add(class_id)
                report["counts"][split]["instances"] += 1
                report["class_instance_counts_by_split"][split][names[class_id]] += 1
            for class_id in present_classes:
                report["class_image_counts_by_split"][split][names[class_id]] += 1
    for split in ("train", "val"):
        if not report["split_presence"][split]:
            errors.append(f"Required {split} split is absent")
        elif not report["counts"][split]["instances"]:
            errors.append(f"Required {split} split contains no valid target instances")
    if not report["split_presence"]["test"]:
        warnings.append("Test split absent: validation-only evaluation; no held-out test result")
    for split in SPLITS:
        if report["split_presence"][split]:
            missing = [name for name, count in report["class_instance_counts_by_split"][split].items() if count == 0]
            if missing:
                warnings.append(f"{split}: no instances for classes {missing}; do not report their AP as measured")
    report["counts"]["total"] = {
        key: sum(report["counts"][split][key] for split in SPLITS)
        for key in ("images", "instances")
    }
    report["valid"] = not errors
    report["partial_evaluation"] = not report["split_presence"]["test"]
    report["evaluation_eligible"] = report["valid"] and not weak_evaluation
    report["d455_evaluation_eligible"] = report["evaluation_eligible"] and report["domain"] == "d455"
    if report["evaluation_eligible"]:
        scope = "validation_only" if report["partial_evaluation"] else "validation_and_test"
        report["evaluation_scope"] = f"{report['domain']}_{scope}"
    return report


def materialize_yolo(path: str | Path, out_dir: str | Path) -> Path:
    """Create images/<split>, labels/<split>, dataset.yaml and provenance files.

    Returns the absolute dataset.yaml path. Refuses invalid manifests and existing
    nonempty outputs. Images are hard-linked when possible, copied otherwise;
    labels are copied. Source data is never edited. A temporary sibling directory
    keeps partial materialization from becoming the requested completed output.
    """
    manifest_path = Path(path).expanduser().resolve()
    report = validate_manifest(manifest_path)
    if not report["valid"]:
        raise ValueError("Invalid dataset manifest:\n" + "\n".join(report["errors"]))
    target = Path(out_dir).expanduser().resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    materialized_records = []
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.building-", dir=target.parent) as temporary:
        stage = Path(temporary)
        for record in manifest["records"]:
            split = record["split"]
            source_image = _resolve(manifest_path.parent, record["image"])
            source_label = _resolve(manifest_path.parent, record["label"])
            source_digest = _hash_file(source_image)
            path_digest = hashlib.sha256(_key(source_image).encode("utf-8")).hexdigest()[:16]
            stable_id = f"{path_digest}_{source_digest[:12]}"
            relative_image = Path("images") / split / f"{stable_id}{source_image.suffix.lower()}"
            relative_label = Path("labels") / split / f"{stable_id}.txt"
            destination_image = stage / relative_image
            destination_label = stage / relative_label
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            if destination_image.exists() or destination_label.exists():
                raise ValueError(f"Materialized filename collision for {source_image}")
            try:
                os.link(source_image, destination_image)
            except OSError:
                shutil.copy2(source_image, destination_image)
            shutil.copy2(source_label, destination_label)
            materialized_records.append({
                **record,
                "image": relative_image.as_posix(),
                "label": relative_label.as_posix(),
                "source_image": str(source_image),
                "source_label": str(source_label),
                "image_sha256": source_digest,
            })
        yaml_lines = [f"path: {json.dumps(target.as_posix(), ensure_ascii=False)}"]
        for split in SPLITS:
            if report["split_presence"][split]:
                yaml_lines.append(f"{split}: images/{split}")
        yaml_lines.append("names: " + json.dumps(report["names"], ensure_ascii=False))
        (stage / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
        materialized_manifest = {
            **manifest,
            "provenance": report["provenance"],
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _hash_file(manifest_path),
            "records": materialized_records,
        }
        (stage / "manifest.json").write_text(json.dumps(materialized_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (stage / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if target.exists():
            target.rmdir()  # Only the previously checked empty directory.
        stage.rename(target)
    return target / "dataset.yaml"
