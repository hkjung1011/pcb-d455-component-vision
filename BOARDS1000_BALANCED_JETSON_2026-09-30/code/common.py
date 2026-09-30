"""Shared helpers for the boards/ports dataset build."""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
STAGING = ROOT / 'staging'
RAW = ROOT / 'raw'
DATASETS = ROOT / 'datasets'
IOTKITS_RECORDS = WORK / 'r03' / 'assets' / 'board_records.json'
HOLDOUT_IMAGES = Path('C:\\Users\\hkjun\\Documents\\Codex\\2026-09-27\\new-chat\\work\\r03\\assets\\commons_holdout\\images')
CLASS_MAP = json.loads((ROOT / 'class_map.json').read_text(encoding='utf-8'))
MODEL_CLASSES = {name: spec['classes'] for name, spec in CLASS_MAP['models'].items()}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_image(path):
    data = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def phash(image):
    """64-bit DCT perceptual hash."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return int(''.join('1' if b else '0' for b in bits), 2)


def phash_variants(image):
    """Hashes of the 8 rotation/flip variants (holdout screening)."""
    out = []
    for flip in (False, True):
        im = cv2.flip(image, 1) if flip else image
        for k in range(4):
            out.append(phash(np.rot90(im, k).copy()))
    return out


def hamming(a, b):
    return bin(a ^ b).count('1')


def rf_family(filename):
    """Roboflow exports name augmented copies '<stem>.rf.<hash>.jpg'; the stem is the original photo."""
    stem = Path(filename).name
    return stem.split('.rf.')[0] if '.rf.' in stem else Path(stem).stem


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
