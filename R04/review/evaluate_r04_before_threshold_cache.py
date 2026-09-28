"""R04 four-class native-coordinate bbox evaluation (no mask or camera claim).

COCO/AP and confidence matching follow R03. Native and dual crop inference both
restore YOLO result.xyxy by translation only: those boxes already use crop pixels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np

NAMES = ['resistor', 'capacitor', 'ic', 'connector']
BIN_NAMES = ['lt8', '8to16', '16to32', 'ge32']
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def utc():
    return datetime.now(timezone.utc).isoformat()


def size_bin(box):
    short = min(box[2]-box[0], box[3]-box[1])
    return 'lt8' if short < 8 else '8to16' if short < 16 else '16to32' if short < 32 else 'ge32'


def bbox_iou(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    overlap = np.maximum(np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]), 0).prod()
    union = np.maximum(a[2:]-a[:2], 0).prod()+np.maximum(b[2:]-b[:2], 0).prod()-overlap
    return float(overlap/union) if union > 0 else 0.


def tile_bounds(height, width, requested_size, overlap=.2):
    """Pixel-edge xyxy bounds. Last tile touches each image edge, matching R03."""
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in [height, width, requested_size]):
        raise ValueError('Positive integer image/tile dimensions required')
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError('Overlap must be finite and in [0,1)')
    def starts(length):
        size = min(length, requested_size)
        end = length-size
        if not end:
            return [0]
        stride = max(1, math.floor(size*(1-overlap)))
        values = list(range(0, end+1, stride))
        if values[-1] != end:
            values.append(end)
        return values
    tw, th = min(width, requested_size), min(height, requested_size)
    return [(x, y, x+tw, y+th) for y in starts(height) for x in starts(width)]


def inference_plan(height, width, pipeline):
    if pipeline not in ['native', 'dual']:
        raise ValueError('Unknown pipeline')
    sizes = [1024] if pipeline == 'native' else [1024, 2048]
    return [{'requested_tile_size': size, 'bounds': bounds}
            for size in sizes for bounds in tile_bounds(height, width, size)]


def restore_prediction(box, bounds, class_id, score, class_count=4):
    """Translate crop-native xyxy, including 2048 crops; never multiply by two."""
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or not math.isfinite(float(score)):
        raise ValueError('Non-finite or malformed prediction')
    if int(class_id) != class_id or not 0 <= int(class_id) < class_count or not 0 <= score <= 1:
        raise ValueError('Invalid prediction class or confidence')
    ox, oy, ex, ey = bounds
    if ex <= ox or ey <= oy:
        raise ValueError('Invalid tile bounds')
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    box[[0, 2]] = np.clip(box[[0, 2]], 0, ex-ox)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, ey-oy)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    box += np.array([ox, oy, ox, oy], dtype=float)
    return {'class_id': int(class_id), 'score': float(score), 'bbox_xyxy': box.tolist()}


def deduplicate(predictions, iou=.5, max_det=1000):
    """One class-aware global NMS across all crops/scales; CPU torch float32."""
    if not predictions:
        return []
    import torch
    from torchvision.ops import batched_nms
    boxes = torch.tensor([p['bbox_xyxy'] for p in predictions], dtype=torch.float32)
    scores = torch.tensor([p['score'] for p in predictions], dtype=torch.float32)
    classes = torch.tensor([p['class_id'] for p in predictions], dtype=torch.int64)
    indices = batched_nms(boxes, scores, classes, iou).tolist()[:max_det]
    return [copy.deepcopy(predictions[i]) for i in indices]


def operate(samples, threshold, class_count=4):
    """Score ordered, same-class greedy IoU>=.5 matching, identical to R03."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Confidence must be finite and in [0,1]')
    counts = np.zeros((class_count, 3), dtype=np.int64)
    bins = {name: [0, 0] for name in BIN_NAMES}
    cross = [{name: [0, 0] for name in BIN_NAMES} for _ in range(class_count)]
    by_image = []
    negatives = negative_fp = 0
    for sample in samples:
        truths = sample['truth']
        found = set()
        local = np.zeros_like(counts)
        predictions = sorted((p for p in sample['predictions'] if p['score'] >= threshold), key=lambda p: -p['score'])
        if not truths:
            negatives += 1
            negative_fp += bool(predictions)
        for pred in predictions:
            candidates = [(bbox_iou(pred['bbox_xyxy'], gt['bbox_xyxy']), j)
                          for j, gt in enumerate(truths)
                          if j not in found and gt['class_id'] == pred['class_id']]
            quality, j = max(candidates, default=(0, -1))
            if quality >= .5:
                found.add(j)
                local[pred['class_id'], 0] += 1
            else:
                local[pred['class_id'], 1] += 1
        for j, gt in enumerate(truths):
            cls = gt['class_id']
            if j not in found:
                local[cls, 2] += 1
            bucket = size_bin(gt['bbox_xyxy'])
            bins[bucket][0] += int(j in found)
            bins[bucket][1] += 1
            cross[cls][bucket][0] += int(j in found)
            cross[cls][bucket][1] += 1
        counts += local
        by_image.append({'id': sample['id'], 'group_id': sample['group_id'], 'source': sample.get('source'),
                         'counts_tp_fp_fn': local.tolist(), 'ground_truth_count': len(truths),
                         'prediction_count': len(predictions)})
    def recall_table(values):
        return {k: {'detected': v[0], 'total': v[1], 'recall': v[0]/v[1] if v[1] else None} for k, v in values.items()}
    tp, fp, fn = map(int, counts.sum(axis=0))
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else None
    return {'confidence': float(threshold), 'match_box_iou': .5, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall,
            'f1': 2*precision*recall/(precision+recall) if recall is not None and precision+recall else 0.,
            'per_class_counts_tp_fp_fn': counts.tolist(), 'size_recall': recall_table(bins),
            'class_by_native_shortside_recall': {NAMES[i]: recall_table(v) for i, v in enumerate(cross)},
            'negative_images': negatives, 'negative_images_with_false_positive': negative_fp,
            'negative_false_positive_image_rate': negative_fp/negatives if negatives else None,
            'per_image': by_image}


def coco_score(ground_truth, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = copy.deepcopy(ground_truth)
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(copy.deepcopy(predictions))
        else:
            dt = COCO()
            dt.dataset = {'images': ground_truth['images'], 'categories': ground_truth['categories'], 'annotations': []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, 'bbox')
        evaluator.params.maxDets = [1, 100, 300]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval['precision']
    def avg(values):
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else None
    def stats(values):
        return {'ap50_95_max100': avg(values[:, :, 0, 1]), 'ap50_max100': avg(values[0, :, 0, 1]),
                'ap75_max100': avg(values[5, :, 0, 1]), 'ap50_95_max300': avg(values[:, :, 0, 2]),
                'ap50_max300': avg(values[0, :, 0, 2]), 'ap75_max300': avg(values[5, :, 0, 2])}
    result = {'iou_type': 'bbox',
              'ap50_95_max100': avg(precision[:, :, :, 0, 1]), 'ap50_max100': avg(precision[0, :, :, 0, 1]),
              'ap75_max100': avg(precision[5, :, :, 0, 1]),
              'ap50_95_max300': avg(precision[:, :, :, 0, 2]), 'ap50_max300': avg(precision[0, :, :, 0, 2]),
              'ap75_max300': avg(precision[5, :, :, 0, 2]),
              'per_class': {category['name']: stats(precision[:, :, evaluator.params.catIds.index(category['id']), :, :])
                            for category in ground_truth['categories']},
              'max_detections_scope': 'Per image and category, before standard COCO accumulation',
              'area_ap50_95_max100': {name: avg(precision[:, :, :, i, 1]) for i, name in enumerate(['all', 'small', 'medium', 'large'])},
              'area_ap50_95_max300': {name: avg(precision[:, :, :, i, 2]) for i, name in enumerate(['all', 'small', 'medium', 'large'])}}
    return result


def threshold_curves(samples):
    """Validation diagnostics; per-class optimal thresholds are not applied."""
    curves = []
    class_curves = {name: [] for name in NAMES}
    for step in range(1, 20):
        threshold = step/20.
        result = operate(samples, threshold)
        curves.append({k: v for k, v in result.items() if k != 'per_image'})
        for name, (tp, fp, fn) in zip(NAMES, result['per_class_counts_tp_fp_fn']):
            p = tp/(tp+fp) if tp+fp else 0.
            r = tp/(tp+fn) if tp+fn else None
            class_curves[name].append({'confidence': threshold, 'tp': tp, 'fp': fp, 'fn': fn, 'precision': p,
                                      'recall': r, 'f1': 2*p*r/(p+r) if r is not None and p+r else 0.})
    selected = max(curves, key=lambda r: (r['f1'], r['confidence']))['confidence']
    return {'selected_global_confidence': selected, 'global_curve': curves, 'per_class_curves': class_curves,
            'per_class_thresholds_applied': False,
            'selection_rule': 'Validation-only box IoU>=0.5 micro-F1 on confidence .05:.95 by .05, highest confidence breaks ties'}


def bootstrap_recall(operating, resamples, split, cohort):
    groups = defaultdict(list)
    for row in operating['per_image']:
        groups[row['group_id']].append(np.array(row['counts_tp_fp_fn']).sum(axis=0))
    reason = None
    if len(groups) < 2:
        reason = 'Only one source board group (or none); no meaningful between-board confidence interval'
    elif split != 'test' or cohort != 'general_test':
        reason = 'Bootstrap prespecified only for general_test development holdout'
    elif not resamples:
        reason = 'Bootstrap not requested'
    if reason:
        return None, reason
    matrix = np.array([np.sum(v, axis=0) for v in groups.values()])
    rng = np.random.default_rng(42)
    values = []
    for _ in range(resamples):
        tp, fp, fn = matrix[rng.integers(0, len(matrix), len(matrix))].sum(axis=0)
        if tp+fn:
            values.append(float(tp/(tp+fn)))
    if not values:
        return None, 'No positive ground truth in bootstrap samples'
    return {'resamples': resamples, 'groups': len(groups), 'lower': float(np.percentile(values, 2.5)),
            'upper': float(np.percentile(values, 97.5)), 'seed': 42,
            'scope': 'Fixed-global-threshold micro-recall group bootstrap, not AP CI; descriptive development holdout'}, None


def resolve_image(row, manifest_path):
    path = Path(row.get('source_image', row['image']))
    return path if path.is_absolute() else manifest_path.parent/path


def make_truth(row, image_id, start_annotation_id, width, height):
    annotations, truths = [], []
    for index, obj in enumerate(row['objects'], start_annotation_id+1):
        cls = obj['class_id']
        box = np.asarray(obj['bbox_xyxy'], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or int(cls) != cls or not 0 <= cls < 4:
            raise ValueError(f'Invalid ground truth: {row["id"]}')
        x1, y1, x2, y2 = map(float, box)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f'Ground truth outside native image: {row["id"]}')
        annotations.append({'id': index, 'image_id': image_id, 'category_id': int(cls)+1,
                            'bbox': [x1, y1, x2-x1, y2-y1], 'area': (x2-x1)*(y2-y1), 'iscrowd': 0})
        truths.append({'class_id': int(cls), 'bbox_xyxy': [x1, y1, x2, y2]})
    return annotations, truths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--cohort')
    parser.add_argument('--pipeline', choices=['native', 'dual'], default='native')
    parser.add_argument('--operating-conf', type=float)
    parser.add_argument('--device', default='0')
    parser.add_argument('--bootstrap', type=int, default=0)
    args = parser.parse_args(argv)
    if args.split != 'val' and args.operating_conf is None:
        parser.error('Test/development holdout requires confidence frozen on validation')
    if args.operating_conf is not None and (not math.isfinite(args.operating_conf) or not 0 <= args.operating_conf <= 1):
        parser.error('Confidence must be finite and in [0,1]')
    if args.bootstrap < 0:
        parser.error('Bootstrap count must be nonnegative')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('A new output directory is required')
    return args


def main(argv=None):
    args = parse_args(argv)
    import os
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'runtime_config'))
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    started = utc()
    manifest_sha, checkpoint_sha = sha(args.manifest), sha(args.weights)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    if manifest['names'] != NAMES:
        raise ValueError('Expected fixed target4 class order')
    rows = [r for r in manifest['records'] if r['split'] == args.split and (not args.cohort or r.get('cohort') == args.cohort)]
    if not rows:
        raise ValueError('No matching native images')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate native image IDs')
    model = YOLO(str(args.weights))
    if model.task != 'detect' or [model.names[i].lower() for i in range(len(model.names))] != NAMES:
        raise ValueError('Checkpoint task or classes differ from target4 bbox ontology')
    args.output.mkdir(parents=True, exist_ok=True)
    ground_truth = {'info': {}, 'images': [], 'annotations': [], 'categories': [{'id': i+1, 'name': n} for i, n in enumerate(NAMES)]}
    samples, predictions, timings, inputs, image_profiles = [], [], [], [], []
    def synchronize():
        if str(args.device).lower() != 'cpu' and torch.cuda.is_available():
            torch.cuda.synchronize(next(model.model.parameters()).device)
    predict_kwargs = dict(imgsz=1024, conf=.001, iou=.6, max_det=1000, verbose=False, device=args.device, half=False)
    for image_id, row in enumerate(rows, 1):
        path = resolve_image(row, args.manifest)
        source_sha = sha(path)
        if row.get('sha256') and source_sha != row['sha256']:
            raise ValueError(f'Native image SHA differs from manifest: {row["id"]}')
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot decode native image: {path}')
        height, width = image.shape[:2]
        if ('width' in row and row['width'] != width) or ('height' in row and row['height'] != height):
            raise ValueError(f'Native dimensions differ: {row["id"]}')
        inputs.append({'id': row['id'], 'path': str(path.resolve()), 'sha256': source_sha, 'width': width, 'height': height})
        ground_truth['images'].append({'id': image_id, 'width': width, 'height': height, 'file_name': str(path)})
        annotations, truths = make_truth(row, image_id, len(ground_truth['annotations']), width, height)
        ground_truth['annotations'].extend(annotations)
        plan = inference_plan(height, width, args.pipeline)
        if image_id == 1:
            ox, oy, ex, ey = plan[0]['bounds']
            model.predict(image[oy:ey, ox:ex], **predict_kwargs)
        synchronize()
        began = time.perf_counter()
        raw = []
        for tile in plan:
            ox, oy, ex, ey = tile['bounds']
            result = model.predict(image[oy:ey, ox:ex], **predict_kwargs)[0]
            if tuple(result.orig_shape) != (ey-oy, ex-ox):
                raise ValueError('Predictor orig_shape differs from actual crop; coordinate audit failed')
            if result.boxes is None:
                continue
            for box, cls, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                restored = restore_prediction(box, tile['bounds'], cls, score)
                if restored is not None:
                    raw.append(restored)
        local = deduplicate(raw)
        synchronize()
        elapsed = (time.perf_counter()-began)*1000
        timings.append(elapsed)
        image_profiles.append({'id': row['id'], 'tiles': len(plan), 'requested_tile_sizes': [1024] if args.pipeline == 'native' else [1024, 2048],
                               'pre_global_nms_predictions': len(raw), 'post_global_nms_predictions': len(local),
                               'offline_predict_and_postprocess_ms': elapsed})
        for item in local:
            x1, y1, x2, y2 = item['bbox_xyxy']
            predictions.append({'image_id': image_id, 'category_id': item['class_id']+1,
                                'bbox': [x1, y1, x2-x1, y2-y1], 'score': item['score']})
        samples.append({'id': row['id'], 'image_id': image_id, 'group_id': row['group_id'], 'source': row.get('source'),
                        'cohort': row.get('cohort'), 'truth': truths, 'predictions': local})
        print(f'EVALUATED {image_id}/{len(rows)} {row["id"]} tiles={len(plan)}', flush=True)
    curve = threshold_curves(samples) if args.split == 'val' else None
    threshold = args.operating_conf if args.operating_conf is not None else curve['selected_global_confidence']
    operating = operate(samples, threshold)
    interval, interval_reason = bootstrap_recall(operating, args.bootstrap, args.split, args.cohort)
    if sha(args.manifest) != manifest_sha or sha(args.weights) != checkpoint_sha:
        raise ValueError('Manifest or checkpoint changed during evaluation')
    report = {'schema': 'r04-native-bbox-evaluation-v1', 'started_utc': started, 'finished_utc': utc(),
              'scope': manifest.get('scope'), 'split': args.split, 'cohort': args.cohort, 'pipeline': args.pipeline,
              'evaluation_role': 'validation_selection' if args.split == 'val' else 'development_holdout_not_pristine_final',
              'test_reuse_notice': 'R03 general/Pi test diagnostics informed R04; this is not a new untouched final test',
              'task': 'detect', 'class_names': NAMES, 'checkpoint_sha256': checkpoint_sha, 'manifest_sha256': manifest_sha,
              'evaluator_sha256': sha(__file__), 'weights_path': str(args.weights.resolve()), 'manifest_path': str(args.manifest.resolve()),
              'ultralytics': ultralytics.__version__, 'torch': torch.__version__, 'device': args.device,
              'annotation_limitations': manifest.get('annotation_limitations', manifest.get('limitations')),
              'ground_truth_policy': 'Unmodified source target4 bbox GT; no unknown or DNP relabel/removal; no evaluation ignore regions added',
              'images': len(rows), 'groups': len({r['group_id'] for r in rows}), 'native_gt_instances': len(ground_truth['annotations']),
              'imgsz': 1024, 'original_crop_sizes': [1024] if args.pipeline == 'native' else [1024, 2048],
              'tile_overlap': .2, 'tile_nms_iou': .5, 'raw_confidence_floor': .001, 'prediction_nms_iou': .6,
              'per_crop_max_det': 1000, 'global_max_det': 1000, 'coordinates': 'crop-native xyxy plus crop offset; no manual scale factor',
              'bbox': coco_score(ground_truth, predictions), 'mask': None, 'operating': operating,
              'threshold_source': 'supplied_frozen_confidence' if args.operating_conf is not None else 'validation_micro_f1_grid',
              'per_class_thresholds_applied': False, 'group_bootstrap_recall95': interval, 'group_bootstrap_exclusion_reason': interval_reason,
              'latency': {'offline_predict_and_postprocess_ms_p50': float(np.percentile(timings, 50)),
                          'offline_predict_and_postprocess_ms_p95': float(np.percentile(timings, 95)), 'samples': len(timings),
                          'warmup_excluded': True, 'scope': 'Offline crop predict plus native coordinate merge/global NMS only; NOT camera-to-result latency',
                          'excludes': ['model initialization', 'file read/hash/decode', 'camera capture', 'visualization', 'metrics', 'disk write']},
              'd455_verified': False, 'physical_specimen_independence_verified': False}
    save(args.output/'native_ground_truth.json', ground_truth)
    save(args.output/'bbox_predictions.json', predictions)
    save(args.output/'samples.json', samples)
    save(args.output/'native_input_inventory.json', inputs)
    save(args.output/'inference_profiles.json', image_profiles)
    if curve:
        save(args.output/'validation_threshold_curve.json', curve['global_curve'])
        save(args.output/'validation_per_class_pr.json', {'per_class_curves': curve['per_class_curves'], 'per_class_thresholds_applied': False})
    save(args.output/'metrics.json', report)
    print(json.dumps({key: report[key] for key in ['split', 'cohort', 'pipeline', 'images', 'native_gt_instances', 'bbox']}, indent=2))


if __name__ == '__main__':
    main()
