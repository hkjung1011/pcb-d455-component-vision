#!/usr/bin/env python3
"""Merge staging sources into YOLO datasets: datasets/boards_v1 and datasets/ports_v1.

Rules
- Commons holdout: an entire source/duplicate family is excluded when any member is
  within pHash distance 8 of a holdout image, including rotation/flip variants.
- Near-duplicates (pHash distance <= 6, across all sources) are merged into one group,
  on top of the source groups, so they never straddle train/val/test.
- Pending boxes exclude their explicit target model; an unknown target excludes both.
- Ports require an explicit verified_complete annotation-completeness status.
- Splits are made per group, stratified by the rarest class in the group: 75/15/10.
"""
from collections import Counter, defaultdict
import argparse
import json
import shutil

import numpy as np

from common import CLASS_MAP, DATASETS, HOLDOUT_IMAGES, MODEL_CLASSES, ROOT, STAGING, hamming, phash_variants, read_image, sha256, write_json

SPLITS = (('train', .75), ('val', .15), ('test', .10))
SEED = 20260929
HOLDOUT_DISTANCE = 8
DUPLICATE_DISTANCE = 6


def popcounts(values):
    """Vectorized uint64 population count without an NxNx64 temporary."""
    x = values.copy()
    x -= (x >> np.uint64(1)) & np.uint64(0x5555555555555555)
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)


def port_annotations_complete(record):
    source = CLASS_MAP['sources'].get(record['source'], {})
    status = record.get('annotation_completeness', {}).get('ports')
    if status is None:
        status = source.get('annotation_completeness', {}).get('ports')
    return status == 'verified_complete' and not str(record.get('annotation_status', '')).startswith('draft')


def pending_for_model(record, model):
    pending = []
    for box in record.get('pending') or []:
        prefix = str(box.get('cls', '')).partition(':')[0]
        if prefix not in MODEL_CLASSES or prefix == model:
            pending.append(box)
    return pending


def clear_dataset(root):
    resolved = root.resolve()
    if resolved.parent != DATASETS.resolve() or resolved.name not in {f'{n}_v1' for n in MODEL_CLASSES}:
        raise ValueError(f'unsafe dataset cleanup target: {resolved}')
    if resolved.exists():
        shutil.rmtree(resolved)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def load_records():
    records = []
    for path in sorted(STAGING.glob('*/records.json')):
        records.extend(json.loads(path.read_text(encoding='utf-8')))
    return records


def split_groups(groups, classes, rng):
    """groups: {gid: [records]} -> {gid: split}, stratified by the rarest class present."""
    totals = Counter(c for recs in groups.values() for r in recs for c in r['_classes'])
    rank = {c: totals[c] for c in classes}
    by_primary = defaultdict(list)
    for gid, recs in groups.items():
        present = {c for r in recs for c in r['_classes']}
        by_primary[min(present, key=lambda c: (rank[c], c))].append(gid)
    assignment = {}
    for cls in sorted(by_primary):
        gids = sorted(by_primary[cls])
        gids = [gids[i] for i in rng.permutation(len(gids))]
        total = sum(len(groups[g]) for g in gids)
        filled = Counter()
        # largest groups first, each to the split furthest below its target share
        for gid in sorted(gids, key=lambda g: -len(groups[g])):
            split = max(SPLITS, key=lambda sr: sr[1] * total - filled[sr[0]])[0]
            assignment[gid] = split
            filled[split] += len(groups[gid])
        # every class with >= 3 groups gets at least one val and one test group
        for need in ('val', 'test'):
            if len(gids) >= 3 and not filled[need]:
                donor = min((g for g in gids if assignment[g] == 'train'), key=lambda g: len(groups[g]))
                assignment[donor] = need
                filled[need] += len(groups[donor])
                filled['train'] -= len(groups[donor])
    return assignment


def write_dataset(name, records, assignment, classes):
    root = DATASETS / f'{name}_v1'
    clear_dataset(root)
    index = {c: i for i, c in enumerate(classes)}
    counts = {s: Counter() for s, _ in SPLITS}
    images = Counter()
    sources = defaultdict(Counter)
    exported = []
    used_stems = set()
    for split, _ in SPLITS:
        (root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    for r in records:
        split = assignment[r['_group']]
        img_dir, lbl_dir = root / 'images' / split, root / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        stem = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in r['id'])
        if stem.casefold() in used_stems:
            raise ValueError(f'output filename collision: {r["id"]}')
        used_stems.add(stem.casefold())
        suffix = '.' + r['image'].rsplit('.', 1)[-1].lower()
        shutil.copyfile(r['image'], img_dir / (stem + suffix))
        lines = []
        for b in r[name]:
            x1, y1, x2, y2 = b['xyxy']
            w, h = r['width'], r['height']
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            lines.append(f"{index[b['cls']]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            counts[split][b['cls']] += 1
        (lbl_dir / (stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
        images[split] += 1
        sources[split][r['source']] += 1
        image_file = img_dir / (stem + suffix)
        label_file = lbl_dir / (stem + '.txt')
        exported.append({
            'id': r['id'], 'source': r['source'], 'source_group': r['group'],
            'group': r['_group'], 'split': split,
            'image': image_file.relative_to(root).as_posix(),
            'label': label_file.relative_to(root).as_posix(),
            'image_sha256': sha256(image_file), 'label_sha256': sha256(label_file),
            'width': r['width'], 'height': r['height'],
            'pending_count': len(pending_for_model(r, name)),
            'other_model_pending_count': len(r.get('pending') or []) - len(pending_for_model(r, name)),
            'annotation_status': r.get('annotation_status'),
            'annotation_completeness': {'ports': 'verified_complete' if port_annotations_complete(r) else 'unverified'},
        })
    (root / 'data.yaml').write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        + 'names:\n' + ''.join(f'  {i}: {c}\n' for i, c in enumerate(classes)), encoding='utf-8')
    write_json(root / 'dataset_index.json', {
        'schema': 'boards_ports_dataset_index_v2', 'model': name, 'classes': classes,
        'class_map_sha256': sha256(ROOT / 'class_map.json'),
        'duplicate_distance': DUPLICATE_DISTANCE, 'rotation_flip_screening': True,
        'holdout_inventory': [{'name': p.name, 'sha256': sha256(p)} for p in sorted(HOLDOUT_IMAGES.glob('*')) if p.is_file()],
        'records': exported,
    })
    return {'root': str(root), 'images': dict(images), 'boxes': {s: dict(c) for s, c in counts.items()},
            'sources': {s: dict(c) for s, c in sources.items()},
            'groups': dict(Counter(assignment[g] for g in {r['_group'] for r in records}))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=sorted(MODEL_CLASSES), default=['boards', 'ports'])
    args = parser.parse_args()
    records = load_records()
    if not records:
        raise SystemExit('no staged records; run build_iotkits.py / import_roboflow.py first')

    holdout = []
    for p in sorted(HOLDOUT_IMAGES.glob('*')):
        image = read_image(p)
        if image is not None:
            holdout.append((p.name, phash_variants(image)))
    if not holdout:
        raise SystemExit('holdout images are missing or unreadable; refusing an unscreened build')

    uf = UnionFind(len(records))
    first_of_group = {}
    for i, r in enumerate(records):
        if r['group'] in first_of_group:
            uf.union(i, first_of_group[r['group']])
        else:
            first_of_group[r['group']] = i
    # Compare rotation/flip variants too; source family names alone miss renamed copies.
    variants = []
    for r in records:
        image = read_image(r['image'])
        if image is None:
            raise ValueError(f'unreadable staged image: {r["image"]}')
        variants.append(phash_variants(image))
    variant_hashes = np.array(variants, dtype=np.uint64)
    normal_hashes = variant_hashes[:, 0]
    near_count, cross_source = 0, 0
    for i, hashes in enumerate(variants):
        if i + 1 == len(records):
            break
        distances = popcounts(np.array(hashes, dtype=np.uint64)[:, None] ^ normal_hashes[None, i + 1:]).min(axis=0)
        distances = np.minimum(distances, popcounts(variant_hashes[i + 1:] ^ normal_hashes[i]).min(axis=1))
        for offset in np.flatnonzero(distances <= DUPLICATE_DISTANCE):
            j = i + 1 + int(offset)
            uf.union(i, j)
            near_count += 1
            cross_source += records[i]['source'] != records[j]['source']
    holdout_groups = {}
    for i, r in enumerate(records):
        hit = next((name for name, hv in holdout
                    if min(hamming(a, b) for a in variants[i] for b in hv) <= HOLDOUT_DISTANCE), None)
        if hit:
            holdout_groups[uf.find(i)] = hit
    excluded, kept = [], []
    for i, r in enumerate(records):
        r['_group'] = f'g{uf.find(i)}'
        if uf.find(i) in holdout_groups:
            excluded.append({'id': r['id'], 'group': r['_group'],
                             'reason': f'family of near holdout image {holdout_groups[uf.find(i)]}'})
        else:
            kept.append(r)
    records = kept

    rng = np.random.default_rng(SEED)
    manifest = {'seed': SEED, 'splits': dict(SPLITS), 'holdout_distance': HOLDOUT_DISTANCE,
                'duplicate_distance': DUPLICATE_DISTANCE, 'holdout_images': len(holdout),
                'excluded_near_holdout': excluded, 'near_duplicate_pairs': near_count,
                'rotation_flip_screening': True,
                'near_duplicate_pairs_across_sources': int(cross_source), 'models': {}}
    for name in args.models:
        classes = MODEL_CLASSES[name]
        eligible, skipped = [], Counter()
        for r in records:
            boxes = r.get(name)
            if pending_for_model(r, name):
                skipped['pending_mapping'] += 1
            elif str(r.get('annotation_status', '')).startswith('draft'):
                skipped['draft_annotations'] += 1
            elif name == 'ports' and not port_annotations_complete(r):
                skipped['incomplete_port_annotations'] += 1
            elif not boxes:
                skipped['no_complete_labels_for_model'] += 1
            else:
                r['_classes'] = sorted({b['cls'] for b in boxes})
                eligible.append(r)
        if not eligible:
            root = DATASETS / f'{name}_v1'
            clear_dataset(root)
            manifest['models'][name] = {'status': 'no eligible images', 'skipped': dict(skipped)}
            continue
        groups = defaultdict(list)
        for r in eligible:
            groups[r['_group']].append(r)
        assignment = split_groups(groups, classes, rng)
        info = write_dataset(name, eligible, assignment, classes)
        info['skipped'] = dict(skipped)
        manifest['models'][name] = info
    write_json(DATASETS / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'excluded_near_holdout'}, indent=2, ensure_ascii=False))
    print('excluded near holdout:', len(excluded))


if __name__ == '__main__':
    main()
