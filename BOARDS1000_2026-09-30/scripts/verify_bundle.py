"""Verify the downloaded snapshot without inference or training (stdlib only)."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(root):
    root = root.resolve()
    manifest = json.loads((root / 'BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        path = (root / entry['path']).resolve()
        assert path.is_relative_to(root), entry['path']
        assert path.is_file(), f'Missing file; download the complete Release ZIP: {entry["path"]}'
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], entry['path']
    dataset = root / 'datasets/boards_v1'
    index = json.loads((dataset / 'dataset_index.json').read_text(encoding='utf-8'))
    evidence = json.loads((root / 'evidence/dataset_verification.json').read_text(encoding='utf-8'))['boards_v1']
    counts = Counter()
    groups, hashes = defaultdict(set), defaultdict(set)
    boxes = {s: Counter() for s in ('train', 'val', 'test')}
    expected = set()
    for record in index['records']:
        split = record['split']
        counts[split] += 1
        groups[record['group']].add(split)
        hashes[record['image_sha256']].add(split)
        for field in ('image', 'label'):
            path = (dataset / record[field]).resolve()
            assert path.is_relative_to(dataset.resolve())
            assert sha(path) == record[field + '_sha256'], record[field]
            expected.add(record[field])
        for line in (dataset / record['label']).read_text().splitlines():
            tokens = line.split()
            assert len(tokens) == 5
            cid = int(tokens[0]); x, y, w, h = map(float, tokens[1:])
            assert 0 <= cid < len(index['classes'])
            assert all(math.isfinite(v) for v in (x, y, w, h))
            assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
            assert x-w/2 >= -1e-5 and y-h/2 >= -1e-5 and x+w/2 <= 1+1e-5 and y+h/2 <= 1+1e-5
            boxes[split][index['classes'][cid]] += 1
    assert dict(counts) == {'train': 650, 'val': 150, 'test': 200}
    assert all(len(s) == 1 for s in groups.values())
    assert all(len(s) == 1 for s in hashes.values())
    actual = {p.relative_to(dataset).as_posix() for d in ('images', 'labels') for p in (dataset/d).rglob('*') if p.is_file() and p.suffix != '.cache'}
    assert actual == expected
    assert {s: dict(c) for s, c in boxes.items()} == evidence['boxes']
    fingerprint = hashlib.sha256()
    for p in sorted(p for p in dataset.rglob('*') if p.is_file() and p.suffix != '.cache'):
        fingerprint.update(p.relative_to(dataset).as_posix().encode('utf-8') + b'\0' + sha(p).encode('ascii') + b'\n')
    assert fingerprint.hexdigest() == evidence['dataset_fingerprint']
    return {'status': 'PASS', 'files_hashed': len(manifest['files']), 'images': dict(counts),
            'dataset_fingerprint': fingerprint.hexdigest(), 'inference_executed': False,
            'scope': 'Bundle bytes, frozen labels and split integrity. Original pHash/Commons checks are archived evidence; no new Commons screening or physical independence claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
