#!/usr/bin/env python3
"""Validate exported files, split provenance and training readiness without training."""
from collections import Counter, defaultdict
import hashlib
import json
import math

import numpy as np
import yaml

from common import CLASS_MAP, DATASETS, HOLDOUT_IMAGES, MODEL_CLASSES, ROOT, hamming, phash_variants, read_image, sha256, write_json

SPLITS = ('train', 'val', 'test')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def dataset_fingerprint(root):
    """Bind verification to every exported file, including unexpected/orphan files."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        # Ultralytics creates cache files after validation; they are not input data.
        if path.suffix == '.cache':
            continue
        digest.update(path.relative_to(root).as_posix().encode('utf-8'))
        digest.update(b'\0')
        digest.update(sha256(path).encode('ascii'))
        digest.update(b'\n')
    return digest.hexdigest()


def holdout_inventory():
    return [{'name': p.name, 'sha256': sha256(p)}
            for p in sorted(HOLDOUT_IMAGES.glob('*')) if p.is_file()]


def check(root):
    from assemble import popcounts

    problems, readiness = [], []
    counts, class_counts = Counter(), {s: Counter() for s in SPLITS}
    hashes, image_entries = {}, {}
    model = root.name.removesuffix('_v1')
    expected_classes = MODEL_CLASSES.get(model, [])
    try:
        meta = yaml.safe_load((root / 'data.yaml').read_text(encoding='utf-8'))
        names = meta.get('names', [])
        if isinstance(names, dict):
            if sorted(names) != list(range(len(names))):
                problems.append('class ids must be contiguous integers beginning at zero')
            names = [names[k] for k in sorted(names)]
        if names != expected_classes or not names:
            problems.append('data.yaml names do not match declared model classes')
        if str(meta.get('path', '')).replace('\\', '/').casefold() != root.as_posix().casefold():
            problems.append('data.yaml path does not match the inspected dataset')
        for split in SPLITS:
            if meta.get(split) != f'images/{split}':
                problems.append(f'wrong data.yaml image path for {split}')
    except Exception as exc:
        problems.append(f'invalid data.yaml: {exc}')
        names = expected_classes

    index = {}
    try:
        index = json.loads((root / 'dataset_index.json').read_text(encoding='utf-8'))
        if index.get('schema') != 'boards_ports_dataset_index_v2' or index.get('model') != model:
            problems.append('invalid dataset index schema/model')
        if index.get('classes') != names:
            problems.append('dataset index classes differ from data.yaml')
        if index.get('class_map_sha256') != sha256(ROOT / 'class_map.json'):
            problems.append('class_map.json changed after assembly; rebuild required')
    except Exception as exc:
        problems.append(f'missing/invalid dataset index: {exc}')

    family_splits, group_splits = defaultdict(set), defaultdict(set)
    for rec in index.get('records', []):
        rel = rec.get('image')
        if rel in image_entries:
            problems.append(f'duplicate image in index: {rel}')
        image_entries[rel] = rec
        family_splits[rec.get('source_group')].add(rec.get('split'))
        group_splits[rec.get('group')].add(rec.get('split'))
        if rec.get('pending_count', 0) or str(rec.get('annotation_status', '')).startswith('draft'):
            problems.append(f'unconfirmed annotations included: {rel}')
        if model == 'ports' and rec.get('annotation_completeness', {}).get('ports') != 'verified_complete':
            problems.append(f'incomplete port annotations included: {rel}')
        if rec.get('source') not in CLASS_MAP['sources']:
            problems.append(f'unknown source in index: {rel}')
    for kind, assignments in (('source family', family_splits), ('duplicate group', group_splits)):
        for group, splits in assignments.items():
            if group is None or len(splits) != 1:
                problems.append(f'{kind} spans splits or is missing: {group}: {sorted(str(s) for s in splits)}')

    disk_images = set()
    for split in SPLITS:
        image_dir, label_dir = root / 'images' / split, root / 'labels' / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            problems.append(f'missing image/label directory: {split}')
        stems = set()
        for image_path in sorted(p for p in image_dir.glob('*') if p.is_file()):
            rel = image_path.relative_to(root).as_posix()
            disk_images.add(rel)
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                problems.append(f'unexpected image file: {rel}')
                continue
            if image_path.stem.casefold() in stems:
                problems.append(f'duplicate image stem: {rel}')
            stems.add(image_path.stem.casefold())
            counts[split] += 1
            label = label_dir / (image_path.stem + '.txt')
            entry = image_entries.get(rel, {})
            if not entry:
                problems.append(f'image absent from index: {rel}')
            elif entry.get('split') != split or entry.get('label') != label.relative_to(root).as_posix():
                problems.append(f'index path/split mismatch: {rel}')
            if entry and entry.get('image_sha256') != sha256(image_path):
                problems.append(f'image changed after assembly: {rel}')
            image = read_image(image_path)
            if image is None:
                problems.append(f'unreadable image: {rel}')
            else:
                hashes[rel] = (split, phash_variants(image))
                if entry and (entry.get('height'), entry.get('width')) != image.shape[:2]:
                    problems.append(f'image dimensions differ from index: {rel}')
            if not label.is_file():
                problems.append(f'missing label: {rel}')
                continue
            if entry and entry.get('label_sha256') != sha256(label):
                problems.append(f'label changed after assembly: {rel}')
            lines = [line.strip() for line in label.read_text(encoding='utf-8').splitlines() if line.strip()]
            if not lines:
                problems.append(f'empty label in positive-only dataset: {rel}')
            for n, line in enumerate(lines, 1):
                try:
                    parts = line.split()
                    cls, values = int(parts[0]), [float(v) for v in parts[1:]]
                    if len(values) != 4 or not 0 <= cls < len(names):
                        raise ValueError('wrong field count/class id')
                    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                        raise ValueError('nonfinite/out-of-range coordinate')
                    cx, cy, w, h = values
                    if w <= 0 or h <= 0 or cx - w / 2 < -1e-6 or cy - h / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 or cy + h / 2 > 1 + 1e-6:
                        raise ValueError('box is empty or extends outside image')
                    class_counts[split][names[cls]] += 1
                except (ValueError, IndexError) as exc:
                    problems.append(f'invalid label {split}/{label.name}:{n}: {exc}')
        for label in sorted(p for p in label_dir.glob('*') if p.is_file()):
            if label.suffix != '.txt' or label.stem.casefold() not in stems:
                problems.append(f'orphan/unexpected label file: {label.relative_to(root).as_posix()}')
        if not counts[split]:
            problems.append(f'empty split: {split}')
    for rel in set(image_entries) - disk_images:
        problems.append(f'indexed image missing from disk: {rel}')

    # Independently use the same distance as assembly, including all 8 orientations.
    threshold = 6
    leaks, leaks_count = [], 0
    items = list(hashes.items())
    variant_hashes = np.array([v[1] for _, v in items], dtype=np.uint64)
    normal = variant_hashes[:, 0] if len(items) else np.array([], dtype=np.uint64)
    splits = np.array([v[0] for _, v in items])
    for i, (name, (split, variants)) in enumerate(items):
        candidate = np.flatnonzero((np.arange(len(items)) > i) & (splits != split))
        if not len(candidate):
            continue
        distances = popcounts(np.array(variants, dtype=np.uint64)[:, None] ^ normal[None, candidate]).min(axis=0)
        distances = np.minimum(distances, popcounts(variant_hashes[candidate] ^ normal[i]).min(axis=1))
        for pos in np.flatnonzero(distances <= threshold):
            leaks_count += 1
            if len(leaks) < 50:
                leaks.append(f'{name} ~ {items[int(candidate[pos])][0]}')

    inventory, holdout_hashes = holdout_inventory(), []
    if not inventory:
        problems.append('holdout inventory missing/empty')
    if index.get('holdout_inventory') != inventory:
        problems.append('holdout inventory differs from assembly')
    for item in inventory:
        image = read_image(HOLDOUT_IMAGES / item['name'])
        if image is None:
            problems.append(f'unreadable holdout: {item["name"]}')
        else:
            holdout_hashes.extend(phash_variants(image))
    holdout_hits = [name for name, (_, variants) in hashes.items()
                    if any(hamming(a, b) <= 8 for a in variants for b in holdout_hashes)]
    missing = {split: [c for c in names if not class_counts[split][c]] for split in SPLITS}
    for split, classes in missing.items():
        if classes:
            readiness.append(f'{split} has zero labels for: {", ".join(classes)}')
    structure_pass = not problems and not leaks_count and not holdout_hits
    return {
        'schema': 'boards_ports_verification_v2', 'images': {s: counts[s] for s in SPLITS},
        'boxes': {s: {c: class_counts[s][c] for c in names} for s in SPLITS},
        'label_problems': problems[:50], 'label_problem_count': len(problems),
        'cross_split_near_duplicates': leaks, 'cross_split_near_duplicate_count': leaks_count,
        'duplicate_distance': threshold, 'rotation_flip_screening': True,
        'holdout_near_duplicates': holdout_hits,
        'missing_classes_by_split': missing, 'readiness_problems': readiness,
        'structure_pass': structure_pass, 'pass': structure_pass,
        'pass_meaning': 'file and leakage checks only; training requires training_ready',
        'training_ready': structure_pass and not readiness,
        'dataset_fingerprint': dataset_fingerprint(root),
        'class_map_sha256': sha256(ROOT / 'class_map.json'), 'holdout_inventory': inventory,
    }


def main():
    result = {f'{name}_v1': check(DATASETS / f'{name}_v1') for name in MODEL_CLASSES
              if (DATASETS / f'{name}_v1').is_dir()}
    for name in MODEL_CLASSES:
        result.setdefault(f'{name}_v1', {'structure_pass': False, 'pass': False, 'training_ready': False,
                                       'readiness_problems': ['dataset not assembled']})
    write_json(DATASETS / 'verification.json', result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
