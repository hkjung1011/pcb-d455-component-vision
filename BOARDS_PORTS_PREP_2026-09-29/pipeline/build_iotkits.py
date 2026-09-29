#!/usr/bin/env python3
"""Select a balanced ~1,000-image board subset from the local IoTKITs records.

Selection is by export-filename group so augmented copies of one photo stay together.
Writes staging/iotkits_v1/records.json (pixel xyxy boxes, mapped board classes).
"""
from collections import defaultdict
import json

import numpy as np

from common import CLASS_MAP, IOTKITS_RECORDS, STAGING, read_image, phash, write_json

SEED = 20260929
# target image counts per mapped board class (primary object of the image)
QUOTA = {
    'raspberry_pi_5': 100, 'raspberry_pi_4': 100, 'raspberry_pi_other': 250,
    'stm32_other': 100, 'arduino': 200, 'esp_board': 150, 'jetson': 100, 'other_board': 60,
}
# multi-board photos are assigned to their rarest board class (IoTKITs STM32 photos also show an ESP32)
PRIORITY = ['stm32_other', 'raspberry_pi_5', 'raspberry_pi_4', 'other_board', 'jetson',
            'raspberry_pi_other', 'esp_board', 'arduino']


def main():
    mapping = CLASS_MAP['sources']['iotkits_v1']['map']
    records = json.loads(IOTKITS_RECORDS.read_text(encoding='utf-8'))
    groups = defaultdict(list)
    unmapped = set()
    for r in records:
        objs = r.get('objects') or []
        if not objs:
            continue
        boxes = []
        for o in objs:
            target = mapping.get(o['class_name'])
            if target is None:
                unmapped.add(o['class_name'])
                break
            boxes.append({'cls': target[0], 'native': o['class_name'], 'xyxy': [float(v) for v in o['bbox_xyxy']]})
        else:
            # IoTKITs file stems like '36_png_jpg' repeat across board classes, so the
            # export family alone merges unrelated photos. Near-duplicates across classes
            # are merged later by pHash in assemble.py.
            primary = min(boxes, key=lambda b: PRIORITY.index(b['cls']))
            groups[f"{r['group_id']}|{primary['native']}"].append({'r': r, 'boards': boxes, 'primary': primary})
    if unmapped:
        raise SystemExit(f'unmapped IoTKITs classes: {sorted(unmapped)}')

    # native-class balancing inside the merged classes (e.g. 10 Pi models -> raspberry_pi_other)
    by_class = defaultdict(lambda: defaultdict(list))
    for gid, items in groups.items():
        primary = items[0]['primary']
        by_class[primary['cls']][primary['native']].append(gid)

    rng = np.random.default_rng(SEED)
    chosen = []
    for cls, quota in QUOTA.items():
        natives = {n: list(rng.permutation(g)) for n, g in sorted(by_class[cls].items())}
        count = 0
        while count < quota and any(natives.values()):
            for n in list(natives):
                if not natives[n] or count >= quota:
                    continue
                gid = natives[n].pop()
                chosen.append(gid)
                count += len(groups[gid])

    out = []
    for gid in chosen:
        for item in groups[gid]:
            r = item['r']
            image = read_image(r['image'])
            out.append({
                'id': r['id'], 'source': 'iotkits_v1', 'image': r['image'], 'group': gid,
                'license': r['license'], 'source_url': r['source_url'],
                'width': r['width'], 'height': r['height'], 'sha256': r['sha256'],
                'phash': str(phash(image)), 'boards': item['boards'], 'ports': None,
                'ports_note': 'ports not annotated in this source; do not use for the ports model',
            })
    counts = defaultdict(int)
    for rec in out:
        for cls in {b['cls'] for b in rec['boards']}:
            counts[cls] += 1
    write_json(STAGING / 'iotkits_v1' / 'records.json', out)
    summary = {'images': len(out), 'groups': len(chosen), 'images_containing_class': dict(counts), 'seed': SEED, 'quota': QUOTA}
    write_json(STAGING / 'iotkits_v1' / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
