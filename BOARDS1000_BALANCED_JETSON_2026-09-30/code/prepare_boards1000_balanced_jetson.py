"""Prepare an isolated train-only Nano support correction; no model inference.

Run --preview, inspect all contact sheets, write visual_review.json, then --build.
Existing experiment images/labels are never modified. All evaluation bytes are frozen.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
PARENT = HERE / 'boards1000_s650_v150_t200_e50_20260929'
ORIGINAL = PARENT / 'datasets/boards_v1'
TRIAL = HERE / 'boards1000_balanced_jetson_20260930'
ROOT = TRIAL / 'datasets/boards_v1'
INVENTORY = HERE / 'next_training_review/nano_candidate_inventory.json'
SOURCE = Path(r'C:\Users\hkjun\Documents\Codex\2026-09-27\new-chat\work\r03\assets\board_records.json')
SEED = 20260930
KEEP_TX2 = {'g915': 13, 'g950': 8, 'g987': 6}
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def utc():
    return datetime.now(timezone.utc).isoformat()


def preview(refresh=False):
    if TRIAL.exists():
        if not refresh or ROOT.exists() or (TRIAL / 'visual_review.json').exists():
            raise SystemExit(f'Preserving existing experiment: {TRIAL}')
        archive = TRIAL / 'review_initial'
        revision = 2
        while archive.exists():
            archive = TRIAL / f'review_revision_{revision}'
            revision += 1
        archive.mkdir(exist_ok=False)
        for path in [TRIAL / 'selection_proposal.json', TRIAL / 'source_annotations.json',
                     *TRIAL.glob('nano_native_gt_sheet_*.jpg')]:
            shutil.copyfile(path, archive / path.name)
    index, inventory = read(ORIGINAL / 'dataset_index.json'), read(INVENTORY)
    assert index['classes'][6] == 'jetson'
    assert inventory['screening']['performed'] and inventory['screening']['exact_metadata_hashes_match']
    source_records = {r['id']: r for r in read(SOURCE)}
    components = defaultdict(list)
    for fam in inventory['families']:
        if fam['eligibility'] != 'CANDIDATE_REQUIRES_VISUAL_AND_LABEL_REVIEW':
            continue
        assert len(fam['candidate_components']) == 1
        assert not fam['current_connected_splits'] and not fam['holdout_hits']
        components[fam['candidate_components'][0]].extend(fam['records'])
    comp_keys = sorted(components)
    rng = random.Random(SEED)
    rng.shuffle(comp_keys)
    exclusions_path = TRIAL / 'candidate_exclusions.json'
    exclusions = read(exclusions_path) if exclusions_path.exists() else []
    excluded_ids = {r['id'] for r in exclusions}
    comp_keys = [c for c in comp_keys if not any(r['id'] in excluded_ids for r in components[c])]
    assert len(comp_keys) >= 40
    selected = []
    for comp in comp_keys[:40]:
        candidate = sorted(components[comp], key=lambda r: r['id'])[0]
        raw = source_records[candidate['id']]
        assert sha(Path(raw['image'])) == raw['sha256'] == candidate['actual_sha256']
        assert all(o['class_name'] == 'Jetson Nano' for o in raw['objects'])
        selected.append({'id': raw['id'], 'candidate_component': comp,
                         'source_family': raw['group_id'], 'source_image': raw['image'],
                         'source_sha256': raw['sha256']})
    tx2 = defaultdict(list)
    for r in index['records']:
        if r['split'] == 'train' and 'Jetson TX2' in r['source_group']:
            tx2[r['group']].append(r)
    assert {g: len(rs) for g, rs in tx2.items()} == {'g915': 33, 'g950': 19, 'g987': 15}
    retained, removed = [], []
    for group, retain_count in KEEP_TX2.items():
        rs = sorted(tx2[group], key=lambda r: r['id'])
        rng.shuffle(rs)
        retained.extend(r['id'] for r in rs[:retain_count])
        removed.extend(r['id'] for r in rs[retain_count:])
    assert len(retained) == 27 and len(removed) == 40
    for r in index['records']:
        if r['id'] in removed:
            assert {int(l.split()[0]) for l in (ORIGINAL / r['label']).read_text().splitlines() if l.strip()} == {6}
    TRIAL.mkdir(exist_ok=refresh)
    proposal = {'created_at': utc(), 'seed': SEED, 'parent': str(ORIGINAL),
                'parent_index_sha256': sha(ORIGINAL / 'dataset_index.json'),
                'inventory_sha256': sha(INVENTORY), 'source_metadata_sha256': sha(SOURCE),
                'selection_basis': 'No model predictions/AP used. Seeded random choice of 40 distinct unused Nano pHash components; one deterministic native record per component. Visually incomplete source components are rejected.',
                'visual_candidate_exclusions': exclusions,
                'nano_additions': selected, 'retained_tx2_ids': retained,
                'removed_tx2_ids': removed, 'retained_tx2_images_per_group': KEEP_TX2,
                'scope': 'Train-only subtype support correction identified before subsequent test evaluation. No evaluation-image selection or label changes.',
                'limitations': 'Filename/pHash component diversity does not establish physical specimen or capture-scene independence.'}
    write(TRIAL / 'selection_proposal.json', proposal)
    write(TRIAL / 'source_annotations.json', {'source_metadata': str(SOURCE),
                                            'source_metadata_sha256': sha(SOURCE),
                                            'records': [source_records[r['id']] for r in selected]})
    for page in range(4):
        sheet = Image.new('RGB', (1600, 700), 'white')
        draw = ImageDraw.Draw(sheet)
        for pos, selected_rec in enumerate(selected[page * 10:(page + 1) * 10]):
            raw = source_records[selected_rec['id']]
            image = Image.open(raw['image']).convert('RGB')
            assert image.size == (raw['width'], raw['height'])
            overlay = ImageDraw.Draw(image)
            for obj in raw['objects']:
                overlay.rectangle(tuple(obj['bbox_xyxy']), outline='red', width=4)
            image.thumbnail((320, 320))
            x, y = (pos % 5) * 320, (pos // 5) * 350
            sheet.paste(image, (x, y))
            draw.text((x + 3, y + 320), selected_rec['id'], fill='black')
            draw.text((x + 3, y + 334), raw['source_parent_filename'] + f" boxes={len(raw['objects'])}", fill='black')
        sheet.save(TRIAL / f'nano_native_gt_sheet_{page + 1}.jpg', quality=95)
    print(json.dumps({'mode': 'preview', 'selected_nano': len(selected),
                      'components': len({r['candidate_component'] for r in selected}),
                      'removed_tx2': len(removed), 'retained_tx2': len(retained),
                      'trial': str(TRIAL)}, indent=2))


def build():
    if ROOT.exists():
        raise SystemExit(f'Preserving existing dataset: {ROOT}')
    proposal, visual = read(TRIAL / 'selection_proposal.json'), read(TRIAL / 'visual_review.json')
    assert visual['reviewed_ids'] == [r['id'] for r in proposal['nano_additions']]
    assert visual['accepted'] and visual['label_edits'] == []
    assert sha(ORIGINAL / 'dataset_index.json') == proposal['parent_index_sha256']
    assert sha(INVENTORY) == proposal['inventory_sha256']
    assert sha(SOURCE) == proposal['source_metadata_sha256']
    index = read(ORIGINAL / 'dataset_index.json')
    names = index['classes']
    for script in ['common.py', 'assemble.py', 'verify_datasets.py']:
        shutil.copyfile(PARENT / script, TRIAL / script)
    mapping = read(PARENT / 'class_map.json')
    mapping['models']['boards']['purpose'] = '8-class 1000-image development experiment; Nano support corrected in train only'
    mapping['experiment_scope']['training_data_correction'] = {
        'parent_dataset': str(ORIGINAL), 'remove_tx2_images': 40, 'add_nano_images': 40,
        'evaluation_splits_frozen': True, 'selection_uses_model_predictions': False,
        'physical_scene_independence': 'NOT_VERIFIED'}
    write(TRIAL / 'class_map.json', mapping)
    for kind in ['images', 'labels']:
        for split in ['train', 'val', 'test']:
            (ROOT / kind / split).mkdir(parents=True)
    removed = set(proposal['removed_tx2_ids'])
    new_records = []
    copied_eval = []
    for r in index['records']:
        if r['id'] in removed:
            assert r['split'] == 'train'
            continue
        shutil.copyfile(ORIGINAL / r['image'], ROOT / r['image'])
        shutil.copyfile(ORIGINAL / r['label'], ROOT / r['label'])
        assert sha(ROOT / r['image']) == r['image_sha256']
        assert sha(ROOT / r['label']) == r['label_sha256']
        new_records.append(r)
        if r['split'] in ['val', 'test']:
            copied_eval.append(r['id'])
    raw_records = {r['id']: r for r in read(TRIAL / 'source_annotations.json')['records']}
    boxes_per_new_image = {}
    for item in proposal['nano_additions']:
        raw = raw_records[item['id']]
        image = f'images/train/{raw["id"]}.jpg'
        label = f'labels/train/{raw["id"]}.txt'
        assert raw['id'] not in {r['id'] for r in new_records}
        shutil.copyfile(Path(raw['image']), ROOT / image)
        assert sha(ROOT / image) == raw['sha256']
        width, height = raw['width'], raw['height']
        lines = []
        for o in raw['objects']:
            mapped = mapping['sources']['iotkits_v1']['map'][o['class_name']][0]
            assert mapped == 'jetson'
            x1, y1, x2, y2 = map(float, o['bbox_xyxy'])
            assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
            coords = [(x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height]
            lines.append(' '.join([str(names.index(mapped))] + [f'{v:.8f}' for v in coords]))
        (ROOT / label).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        boxes_per_new_image[raw['id']] = len(lines)
        new_records.append({'id': raw['id'], 'source': 'iotkits_v1',
            'source_group': raw['group_id'] + '|Jetson Nano',
            'group': f'nano_unused_c{item["candidate_component"]}', 'split': 'train',
            'image': image, 'label': label, 'image_sha256': sha(ROOT / image),
            'label_sha256': sha(ROOT / label), 'width': width, 'height': height,
            'pending_count': 0, 'other_model_pending_count': 0, 'annotation_status': 'publisher_native_visual_reviewed',
            'annotation_completeness': {'ports': 'unverified'}, 'prior_split': raw['originalsplit'],
            'source_native_image': raw['image'], 'source_metadata': str(TRIAL / 'source_annotations.json'),
            'source_annotation_sha256': hashlib.sha256(json.dumps(raw, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            'source_license': raw.get('license'), 'source_url': raw.get('source_url'),
            'source_independence': 'NOT_VERIFIED', 'visual_review': str(TRIAL / 'visual_review.json')})
    write(ROOT / 'dataset_index.json', {**index, 'class_map_sha256': sha(TRIAL / 'class_map.json'), 'records': new_records})
    (ROOT / 'data.yaml').write_text(f'path: {ROOT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(names)), encoding='utf-8')
    assert Counter(r['split'] for r in new_records) == {'train':650, 'val':150, 'test':200}
    old_eval = [r for r in index['records'] if r['split'] in ['val', 'test']]
    new_eval = [r for r in new_records if r['split'] in ['val', 'test']]
    assert new_eval == old_eval
    assert len(copied_eval) == 350
    counts, groups = {}, {}
    for split in ['train', 'val', 'test']:
        box_counts, class_groups = Counter(), defaultdict(set)
        for r in new_records:
            if r['split'] != split: continue
            for line in (ROOT / r['label']).read_text().splitlines():
                if not line.strip(): continue
                cls = names[int(line.split()[0])]
                box_counts[cls] += 1
                class_groups[cls].add(r['group'])
        counts[split] = {n: box_counts[n] for n in names}
        groups[split] = {n: len(class_groups[n]) for n in names}
    sys.path.insert(0, str(TRIAL))
    from verify_datasets import check
    result = check(ROOT)
    write(TRIAL / 'datasets/verification.json', {'boards_v1':result})
    derivation = {'created_at':utc(), 'parent_dataset':str(ORIGINAL),
        'parent_dataset_fingerprint':read(PARENT / 'datasets/verification.json')['boards_v1']['dataset_fingerprint'],
        'dataset_fingerprint':result['dataset_fingerprint'], 'images':result['images'],
        'boxes':counts, 'class_group_counts':groups, 'evaluation_bytes_and_records_unchanged':True,
        'evaluation_images_verified':len(copied_eval), 'retained_tx2_ids':proposal['retained_tx2_ids'],
        'removed_tx2_ids':proposal['removed_tx2_ids'], 'nano_additions':proposal['nano_additions'],
        'nano_added_boxes_per_image':boxes_per_new_image, 'labels_modified':[],
        'rationale':proposal['scope'], 'selection_uses_model_predictions':False,
        'physical_scene_independence':'NOT_VERIFIED', 'visual_review_sha256':sha(TRIAL / 'visual_review.json'),
        'source_annotations_sha256':sha(TRIAL / 'source_annotations.json'),
        'selection_proposal_sha256':sha(TRIAL / 'selection_proposal.json'),
        'source_metadata_sha256':proposal['source_metadata_sha256']}
    write(TRIAL / 'derivation.json', derivation)
    print(json.dumps({'dataset':str(ROOT), 'images':result['images'], 'boxes':counts,
        'class_group_counts':groups, 'dataset_fingerprint':result['dataset_fingerprint'],
        'structure_pass':result['structure_pass'], 'training_ready':result['training_ready'],
        'label_problem_count':result['label_problem_count'],
        'cross_split_near_duplicate_count':result['cross_split_near_duplicate_count'],
        'holdout_near_duplicates':result['holdout_near_duplicates'],
        'evaluation_images_verified_unchanged':350}, indent=2))
    assert result['structure_pass'] and result['training_ready']


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--preview', action='store_true')
    args.add_argument('--build', action='store_true')
    args.add_argument('--refresh-preview', action='store_true')
    opt = args.parse_args()
    assert opt.preview != opt.build, 'Choose one of --preview or --build'
    assert not opt.refresh_preview or opt.preview
    preview(refresh=opt.refresh_preview) if opt.preview else build()
