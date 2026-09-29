#!/usr/bin/env python3
"""Audited Roboflow YOLO import; native annotations are never discarded.

Invalid labels and conflicting exact duplicates are quarantined, never staged.
Review includes representative and full paginated crops. Optional override schema:
{"records": {"source:train/images/example.jpg": {
  "annotations": {"0": {"target": "boards:stm32_nucleo",
    "status": "confirmed_by_visual_review", "reason": "review evidence"}},
  "annotation_completeness": {"ports": "incomplete"}}}}
Indices are zero-based nonempty label-line indices. Mapping overrides do not fill
missing boxes. Ports remain incomplete unless explicitly verified complete.
"""
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import zipfile

import cv2
import numpy as np
import yaml

from common import CLASS_MAP, MODEL_CLASSES, RAW, ROOT, STAGING, phash, read_image, rf_family, sha256, write_json

SLUG_TO_SOURCE = {
    'raspberrypi-vniye': 'rf_raspberrypi_vniye',
    'raspberry-pi-dpiwf': 'rf_raspberry_pi_dpiwf',
    'stm32': 'rf_stm32_nucleo',
    'stm32-3eajy': 'rf_stm32_nucleo',
    'embedded-hardware': 'rf_embedded_hardware',
}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def safe_extract(zip_path, digest):
    """Digest-scoped extraction; identical duplicate members are safe and counted."""
    parent = RAW / 'extracted'
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / digest
    marker = destination / '.archive_complete.json'
    if destination.exists():
        if marker.is_file() and json.loads(marker.read_text(encoding='utf-8')).get('zip_sha256') == digest:
            return destination
        raise ValueError(f'incomplete extraction exists; inspect it before retrying: {destination}')
    temporary = Path(tempfile.mkdtemp(prefix=f'{digest[:12]}-', dir=parent))
    with zipfile.ZipFile(zip_path) as archive:
        seen, identical_duplicates = {}, []
        reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
        for info in archive.infolist():
            name = info.filename.replace('\\', '/')
            path = PurePosixPath(name)
            if (path.is_absolute() or not path.parts or '..' in path.parts
                    or any(':' in part or part.endswith(('.', ' ')) for part in path.parts)
                    or any(part.split('.')[0].upper() in reserved for part in path.parts)):
                raise ValueError(f'unsafe archive path: {info.filename!r}')
            key = path.as_posix().casefold()
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f'symbolic link archive member: {info.filename!r}')
            output = temporary.joinpath(*path.parts)
            if not output.resolve().is_relative_to(temporary.resolve()):
                raise ValueError(f'archive member escapes extraction directory: {info.filename!r}')
            if key in seen:
                # Roboflow sometimes repeats data.yaml once per split. Accept only
                # exactly identical bytes and spelling, never last-member-wins.
                if (seen[key] != path.as_posix() or info.is_dir() != output.is_dir()
                        or (not info.is_dir() and archive.read(info) != output.read_bytes())):
                    raise ValueError(f'conflicting duplicate archive member: {info.filename!r}')
                identical_duplicates.append(info.filename)
                continue
            seen[key] = path.as_posix()
            if info.is_dir():
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, output.open('wb') as target:
                    shutil.copyfileobj(source, target)
    write_json(temporary / '.archive_complete.json', {'zip_sha256': digest, 'zip': zip_path.name,
                                                      'identical_duplicate_members': identical_duplicates})
    os.replace(temporary, destination)
    return destination


def identify(extract_dir, zip_name):
    candidates = []
    for data_yaml in sorted(extract_dir.rglob('data.yaml')):
        meta = yaml.safe_load(data_yaml.read_text(encoding='utf-8-sig')) or {}
        project = str((meta.get('roboflow') or {}).get('project', ''))
        if project in SLUG_TO_SOURCE:
            candidates.append((SLUG_TO_SOURCE[project], meta, data_yaml))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f'multiple recognized data.yaml files in {zip_name}')
    yamls = sorted(extract_dir.rglob('data.yaml'))
    if len(yamls) == 1:
        for slug, source in sorted(SLUG_TO_SOURCE.items(), key=lambda item: -len(item[0])):
            if slug in zip_name.lower():
                return source, yaml.safe_load(yamls[0].read_text(encoding='utf-8-sig')), yamls[0]
    raise ValueError(f'cannot unambiguously identify Roboflow project for {zip_name}')


def class_names(meta):
    names = meta.get('names')
    if isinstance(names, dict):
        names = {int(key): value for key, value in names.items()}
        if sorted(names) != list(range(len(names))):
            raise ValueError('native class ids must be contiguous and zero-based')
        names = [names[index] for index in range(len(names))]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError('data.yaml names must be a string list or class-id map')
    if len(set(names)) != len(names):
        raise ValueError('duplicate native class names in data.yaml')
    if meta.get('nc') is not None and int(meta['nc']) != len(names):
        raise ValueError('data.yaml nc and names disagree')
    return names


def target_of(entry, default_model):
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        raise ValueError(f'mapping must be [target, status]: {entry!r}')
    value, status = entry
    if not isinstance(value, str) or not isinstance(status, str):
        raise ValueError('mapping target and status must be strings')
    if value == 'drop':
        return 'dropped', None, 'drop'
    if value.startswith('needs_') or status.startswith('needs_'):
        return 'pending', value, status
    model, cls = value.split(':', 1) if ':' in value else (default_model, value)
    if model not in MODEL_CLASSES or cls not in MODEL_CLASSES[model]:
        raise ValueError(f'unknown target mapping: {value!r} (default model {default_model!r})')
    return model, cls, status


def parse_native(line, index, line_number, names, width, height):
    parts = line.split()
    if len(parts) < 5 or not parts[0].isdigit():
        raise ValueError(f'line {line_number}: invalid YOLO class/field count')
    class_id = int(parts[0])
    if class_id >= len(names):
        raise ValueError(f'line {line_number}: unknown native class id {class_id}')
    values = [float(value) for value in parts[1:]]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        raise ValueError(f'line {line_number}: nonfinite or out-of-range normalized coordinates')
    if len(values) == 4:
        cx, cy, bw, bh = values
        if bw <= 0 or bh <= 0:
            raise ValueError(f'line {line_number}: nonpositive box size')
        xyxy = [(cx - bw / 2) * width, (cy - bh / 2) * height,
                (cx + bw / 2) * width, (cy + bh / 2) * height]
        native_format = 'yolo_bbox'
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs, ys = values[0::2], values[1::2]
        xyxy = [min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height]
        native_format = 'yolo_polygon'
    else:
        raise ValueError(f'line {line_number}: invalid polygon coordinate count')
    clipped = [max(0.0, min(value, limit)) for value, limit in zip(xyxy, (width, height, width, height))]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f'line {line_number}: zero area box after clipping')
    return {'native_index': index, 'line_number': line_number, 'native_class_id': class_id,
            'native': names[class_id], 'native_class': names[class_id],
            'native_format': native_format, 'native_coordinates': values, 'native_line': line,
            'xyxy': [round(value, 4) for value in clipped], 'box_clipped': xyxy != clipped}


def provider_split(relative):
    return next((part for part in relative.parts[:-1] if part.lower() in {'train', 'val', 'valid', 'validation', 'test'}), 'unknown')


def grouping_for(source, filename):
    """Conservative capture grouping; augmentation family remains separate evidence.

    These rules are source-specific findings from visual review, not a claim that
    arbitrary filenames or pHash matches prove independent acquisition sessions.
    """
    family = rf_family(filename)
    result = {'group': f'{source}:{family}', 'augmentation_family': f'{source}:{family}',
              'group_basis': 'roboflow_augmentation_family'}
    if source == 'rf_embedded_hardware':
        match = re.fullmatch(r'(?P<video>(?:AURIX|Raspberry_Pi|STM32)_\d+_MOV)-(?P<frame>\d+)_(?:jpg|jpeg|png)', family)
        if match:
            result.update(group=f'{source}:video:{match["video"]}',
                          group_basis='source_video_filename_prefix',
                          video_family=match['video'], frame_index=int(match['frame']))
    elif source == 'rf_raspberry_pi_dpiwf':
        # Visual audit: every exported angle depicts the same six-Pi cluster.
        result.update(group=f'{source}:scene:six_pi_cluster',
                      group_basis='visual_audit_single_scene_all_camera_angles',
                      scene_id='six_pi_cluster')
    elif source == 'rf_raspberrypi_vniye':
        match = re.fullmatch(r'(?P<date>\d{8})_(?P<time>\d{6})(?:[-_].*)?', family)
        if match:
            try:
                captured = datetime.strptime(match['date'] + match['time'], '%Y%m%d%H%M%S')
            except ValueError:
                return result
            day = captured.date().isoformat()
            result.update(group=f'{source}:capture_day:{day}',
                          group_basis='capture_day_conservative_same_scene', capture_day=day,
                          filename_capture_time=captured.isoformat())
    return result


@lru_cache(maxsize=8)
def review_image(path):
    return read_image(path)


def make_sheet(items, output):
    columns, tw, th = 6, 240, 196
    rows = (len(items) + columns - 1) // columns
    sheet = np.full((rows * th, columns * tw, 3), 245, np.uint8)
    for tile, item in enumerate(items):
        image = review_image(item['image'])
        x1, y1, x2, y2 = item['xyxy']
        crop = image[max(0, math.floor(y1)):math.ceil(y2), max(0, math.floor(x1)):math.ceil(x2)]
        if crop.size:
            scale = min(232 / crop.shape[1], 144 / crop.shape[0])
            crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))))
            oy, ox = (tile // columns) * th + 4, (tile % columns) * tw + 4
            sheet[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        oy, ox = (tile // columns) * th + 160, (tile % columns) * tw + 4
        label = f"#{item['native_index']} {item['provider_split']} {item['image_sha256'][:8]}"
        filename = Path(item['relative_image']).name
        for offset, line in enumerate((label, filename[:33])):
            cv2.putText(sheet, line, (ox, oy + 17 * offset), cv2.FONT_HERSHEY_SIMPLEX, .37, (10, 10, 10), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError(f'cannot encode crop sheet {output}')
    data.tofile(str(output))


def make_review(records, review, page_size):
    crops = defaultdict(list)
    for record in records:
        for annotation in record['native_annotations']:
            crops[annotation['native']].append({
                'record_id': record['id'], 'native_index': annotation['native_index'],
                'native': annotation['native'], 'xyxy': annotation['xyxy'],
                'image': record['image'], 'relative_image': record['provenance']['relative_image'],
                'image_sha256': record['sha256'], 'provider_split': record['provider_split'],
                'group': record['group'], 'target_model': annotation['target_model'],
                'target_class': annotation['target_class'], 'mapping_status': annotation['mapping_status']})
    index = {'schema': 'roboflow_crop_review_v2', 'page_size': page_size, 'classes': {}}
    for native, items in sorted(crops.items()):
        families = defaultdict(list)
        for item in items:
            families[item['group']].append(item)
        for family in families.values():
            family.sort(key=lambda item: json_digest([item['record_id'], item['native_index']]))
        # Deterministic hash rank spread across the whole archive and round-robin
        # families prevent early files or one augmentation sequence dominating review.
        ordered = []
        family_names = sorted(families, key=lambda group: json_digest(group))
        for depth in range(max(map(len, families.values()))):
            ordered.extend(families[group][depth] for group in family_names if depth < len(families[group]))
        safe = ''.join(char if char.isalnum() else '_' for char in native)[:60] + '_' + json_digest(native)[:8]
        representative = f'{safe}/representative.jpg'
        make_sheet(ordered[:min(48, len(ordered))], review / representative)
        pages = []
        for start in range(0, len(ordered), page_size):
            filename = f'{safe}/page_{start // page_size + 1:04d}.jpg'
            page = ordered[start:start + page_size]
            make_sheet(page, review / filename)
            pages.append({'file': filename, 'items': [{**item, 'tile': tile} for tile, item in enumerate(page)]})
        index['classes'][native] = {'count': len(items), 'families': len(families), 'representative': representative,
                                    'representative_items': ordered[:min(48, len(ordered))], 'pages': pages}
    write_json(review / 'index.json', index)
    review_image.cache_clear()
    return {native: len(items) for native, items in crops.items()}


def import_zip(zip_path, overrides=None, page_size=48):
    overrides = overrides or {}
    zip_digest = sha256(zip_path)
    extract_dir = safe_extract(zip_path, zip_digest)
    source, meta, yaml_path = identify(extract_dir, zip_path.name)
    spec = CLASS_MAP['sources'][source]
    names = class_names(meta)
    unknown = sorted(set(names) - set(spec['map']))
    if unknown:
        write_json(STAGING / source / 'records.json', [])
        write_json(STAGING / source / 'summary.json', {'source': source, 'status': 'failed',
                   'zip_sha256': zip_digest, 'unknown_native_classes': unknown})
        raise ValueError(f'{source}: classes missing from class_map.json: {unknown}')
    default_model = spec['model'] if spec['model'] in MODEL_CLASSES else None
    metadata = meta.get('roboflow') or {}
    records, quarantined, by_digest, applied = [], [], defaultdict(list), []
    image_paths = sorted(path for path in extract_dir.rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    yaml_digest = sha256(yaml_path)
    for image_path in image_paths:
        relative = image_path.relative_to(extract_dir)
        record_id = f'{source}:{relative.as_posix()}'
        label_path = image_path.parent.parent / 'labels' / (image_path.stem + '.txt')
        provenance = {'zip': zip_path.name, 'zip_sha256': zip_digest, 'roboflow': metadata,
                      'version': metadata.get('version'), 'relative_image': relative.as_posix(),
                      'relative_label': label_path.relative_to(extract_dir).as_posix(),
                      'provider_split': provider_split(relative), 'data_yaml_sha256': yaml_digest}
        grouping = grouping_for(source, image_path.name)
        provenance['grouping'] = dict(grouping)
        rec = {'id': record_id, 'source': source, 'image': str(image_path),
               **grouping, 'license': spec['license'],
               'source_url': spec['url'], 'source_version': metadata.get('version'),
               'zip_sha256': zip_digest, 'provider_split': provider_split(relative),
               'provenance': provenance, 'boards': [], 'ports': [], 'pending': [], 'dropped': [],
               'native_annotations': [], 'annotation_status': 'imported_native_mapping',
               'annotation_completeness': {'ports': 'incomplete', **spec.get('annotation_completeness', {})}}
        try:
            image = read_image(image_path)
            if image is None:
                raise ValueError('unreadable image')
            height, width = image.shape[:2]
            rec.update(width=width, height=height, sha256=sha256(image_path), phash=str(phash(image)))
            if not label_path.is_file():
                raise ValueError('missing label file (not an explicit empty negative label)')
            text = label_path.read_text(encoding='utf-8-sig')
            provenance['label_sha256'] = sha256(label_path)
            rec['native_label_lines'] = text.splitlines()
            record_override = overrides.get('records', {}).get(record_id, {})
            annotations_override = record_override.get('annotations', {})
            used_indices = set()
            for line_number, raw in enumerate(text.splitlines(), 1):
                line = raw.strip()
                if not line:
                    continue
                native = parse_native(line, len(rec['native_annotations']), line_number, names, width, height)
                override = annotations_override.get(str(native['native_index']))
                entry = spec['map'][native['native']]
                if override is not None:
                    if not isinstance(override, dict) or not override.get('reason'):
                        raise ValueError('annotation override requires target, status, and review reason')
                    entry = [override['target'], override['status']]
                    used_indices.add(str(native['native_index']))
                    native['mapping_override'] = override
                model, cls, status = target_of(entry, default_model)
                native.update(target_model=model, target_class=cls, mapping_status=status)
                rec['native_annotations'].append(native)
                rec[model].append({'cls': cls, 'native': native['native'], 'native_index': native['native_index'],
                                   'xyxy': native['xyxy'], 'status': status})
            unused = set(annotations_override) - used_indices
            if unused:
                raise ValueError(f'override references nonexistent native annotation indices: {sorted(unused)}')
            if record_override:
                rec['annotation_completeness'].update(record_override.get('annotation_completeness', {}))
                rec['annotation_status'] = record_override.get('annotation_status', 'reviewed_mapping_override')
                rec['review_override'] = record_override
                applied.append(record_id)
            records.append(rec)
            by_digest[rec['sha256']].append(rec)
        except (ValueError, KeyError, TypeError, OSError, cv2.error) as error:
            quarantined.append({'id': record_id, 'reason': str(error), 'record': rec})
            if rec.get('sha256'):
                by_digest[rec['sha256']].append(rec)
    conflicts, duplicates, discarded = [], [], set()
    invalid_ids = {item['id'] for item in quarantined}
    for digest, copies in by_digest.items():
        if len(copies) < 2:
            continue
        signatures = {json_digest(sorted([(ann['native'], ann['native_format'], ann['native_coordinates'],
                                           ann['target_model'], ann['target_class'])
                                          for ann in record['native_annotations']], key=repr)) for record in copies}
        if len(signatures) != 1 or any(record['id'] in invalid_ids for record in copies):
            conflicts.append({'sha256': digest, 'records': [record['id'] for record in copies],
                              'reason': 'exact image duplicate has conflicting native/target labels or invalid labels'})
            for record in copies:
                discarded.add(record['id'])
                if record['id'] not in invalid_ids:
                    quarantined.append({'id': record['id'], 'reason': 'exact_duplicate_annotation_conflict', 'record': record})
        else:
            keep = copies[0]
            keep['duplicate_provenance'] = [{'id': record['id'], **record['provenance']} for record in copies[1:]]
            for record in copies[1:]:
                discarded.add(record['id'])
                duplicates.append({'id': record['id'], 'kept_id': keep['id'], 'sha256': digest})
    records = [record for record in records if record['id'] not in discarded]
    review = STAGING / source / 'review' / zip_digest[:16] / json_digest([spec, overrides])[:16]
    native_counts = make_review(records, review, page_size) if records else {}
    summary = {'schema': 'roboflow_import_v2', 'source': source, 'status': 'imported_with_quarantine' if quarantined else 'imported',
               'zip': zip_path.name, 'zip_sha256': zip_digest, 'roboflow': metadata,
               'source_version': metadata.get('version'), 'native_class_names': names,
               'provider_split_counts': dict(Counter(record['provider_split'] for record in records)),
               'input_images': len(image_paths), 'files_after_exact_dedup': len(records),
               'groups': len({record['group'] for record in records}), 'native_box_counts': native_counts,
               'group_basis_counts': dict(Counter(record['group_basis'] for record in records)),
               'augmentation_families': len({record['augmentation_family'] for record in records}),
               'video_groups': len({record['video_family'] for record in records if 'video_family' in record}),
               'quarantined_images': len(quarantined), 'exact_duplicate_conflicts': len(conflicts),
               'exact_duplicates_removed': len(duplicates), 'mapping_override_records_applied': applied,
               'class_map_sha256': sha256(ROOT / 'class_map.json'), 'overrides_sha256': json_digest(overrides),
               'images_with_pending_boxes': sum(bool(record['pending']) for record in records),
               'ports_completeness': dict(Counter(record['annotation_completeness']['ports'] for record in records)),
               'review_index': str(review / 'index.json')}
    write_json(STAGING / source / 'quarantine.json', quarantined)
    write_json(STAGING / source / 'duplicate_conflicts.json', conflicts)
    write_json(STAGING / source / 'exact_duplicates.json', duplicates)
    write_json(STAGING / source / 'records.json', records)
    write_json(STAGING / source / 'summary.json', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('zips', nargs='*', type=Path, help='default: all raw/*.zip')
    parser.add_argument('--overrides', type=Path, default=ROOT / 'annotation_overrides.json')
    parser.add_argument('--page-size', type=int, default=48)
    args = parser.parse_args()
    if args.page_size < 1 or args.page_size > 120:
        parser.error('--page-size must be between 1 and 120')
    zips = args.zips or sorted(RAW.glob('*.zip'))
    if not zips:
        raise SystemExit(f'no zip files in {RAW}')
    overrides = json.loads(args.overrides.read_text(encoding='utf-8-sig')) if args.overrides.is_file() else {}
    if not isinstance(overrides, dict) or not isinstance(overrides.get('records', {}), dict):
        raise SystemExit('annotation overrides must contain a records object')
    # Avoid silently replacing one dataset version by a different ZIP from raw/.
    imports, seen = [], {}
    for path in zips:
        digest = sha256(path)
        source, _, _ = identify(safe_extract(path, digest), path.name)
        if source in seen:
            if seen[source] == digest:
                continue
            raise SystemExit(f'multiple distinct ZIPs for {source}; explicitly pass the desired ZIP version')
        seen[source] = digest
        imports.append(path)
    for path in imports:
        import_zip(path, overrides, args.page_size)


if __name__ == '__main__':
    main()
