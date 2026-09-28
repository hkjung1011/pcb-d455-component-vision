"""CPU-only evidence audit; never runs model inference or modifies source metrics."""
from __future__ import annotations
import copy
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'reports/evaluation_suite'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def independent_operating(samples, threshold, class_count):
    """Independent vectorized-IoU implementation of specified greedy matching."""
    totals = np.zeros((class_count, 3), dtype=np.int64)
    sizes = np.zeros((4, 2), dtype=np.int64)
    negatives = negative_fp = 0
    for row in samples:
        truth = row['truth']
        predictions = sorted((x for x in row['predictions'] if x['score'] >= threshold), key=lambda x: -x['score'])
        boxes = np.array([x['bbox_xyxy'] for x in truth], dtype=float).reshape(-1, 4)
        labels = np.array([x['class_id'] for x in truth])
        hit = np.zeros(len(truth), bool)
        if not truth:
            negatives += 1
            negative_fp += bool(predictions)
        for pred in predictions:
            candidates = np.flatnonzero((labels == pred['class_id']) & ~hit)
            chosen = -1
            if len(candidates):
                box = np.array(pred['bbox_xyxy'])
                others = boxes[candidates]
                inter = np.clip(np.minimum(others[:, 2:], box[2:]) - np.maximum(others[:, :2], box[:2]), 0, None).prod(1)
                union = np.clip(others[:, 2:] - others[:, :2], 0, None).prod(1) + np.clip(box[2:] - box[:2], 0, None).prod() - inter
                iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
                position = len(iou) - 1 - int(np.argmax(iou[::-1]))
                if iou[position] >= .5:
                    chosen = int(candidates[position])
            if chosen >= 0:
                hit[chosen] = True
                totals[pred['class_id'], 0] += 1
            else:
                totals[pred['class_id'], 1] += 1
        for idx, obj in enumerate(truth):
            totals[obj['class_id'], 2] += int(not hit[idx])
            short = float(min(boxes[idx, 2:] - boxes[idx, :2]))
            bucket = 0 if short < 8 else 1 if short < 16 else 2 if short < 32 else 3
            sizes[bucket, 0] += int(hit[idx])
            sizes[bucket, 1] += 1
    return totals.tolist(), sizes.tolist(), negatives, negative_fp

def main():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    aggregate = read(SUITE / 'aggregate.json')
    protocol = read(SUITE / 'protocol.json')
    selection = read(SUITE / 'selection.json')
    check_count = 0
    def require(condition, label):
        nonlocal check_count
        check_count += 1
        if not condition:
            raise AssertionError(label)
    require(fingerprint({'plan': protocol['plan'], 'evidence': protocol['evidence']}) == protocol['protocol_sha256'] == selection['protocol_sha256'] == aggregate['protocol_sha256'], 'protocol fingerprint')
    require(sha(SUITE / 'selection.json') == aggregate['selection_sha256'], 'selection hash')
    models = {x['model_id']: x for x in protocol['plan']['models']}
    jobs = {x.stem: read(x) for x in (SUITE / 'jobs').glob('*.json')}
    vals = [x for x in jobs.values() if x['binding']['split'] == 'val']
    tests = [x for x in jobs.values() if x['binding']['split'] == 'test']
    require(len(jobs) == 20 and len(vals) == 10 and len(tests) == 10, 'ten val and ten test jobs')
    require(all(x['status'] == 'completed' and x['return_code'] == 0 for x in jobs.values()), 'jobs complete')
    parse = datetime.fromisoformat
    require(max(parse(x['finished_utc']) for x in vals) <= parse(selection['frozen_utc']) < min(parse(x['started_utc']) for x in tests), 'validation then frozen selection then all tests')
    cache = {}
    manifest_summary = {}
    for model in models.values():
        path = Path(model['manifest'])
        if str(path) in cache:
            continue
        data = cache[str(path)] = read(path)
        groups, hashes = defaultdict(set), defaultdict(set)
        for row in data['records']:
            groups[row['group_id']].add(row['split'])
            hashes[row['sha256']].add(row['split'])
        require(all(len(v) == 1 for v in groups.values()), 'group split overlap: ' + str(path))
        require(all(len(v) == 1 for v in hashes.values()), 'exact image SHA split overlap: ' + str(path))
        splits = {}
        for split in ['train', 'val', 'test']:
            rows = [r for r in data['records'] if r['split'] == split]
            splits[split] = {'images': len(rows), 'groups': len({r['group_id'] for r in rows}), 'gt_instances': sum(len(r['objects']) for r in rows), 'class_counts': dict(Counter(data['names'][o['class_id']] for r in rows for o in r['objects']))}
        manifest_summary[str(path.relative_to(ROOT))] = splits
    pi = read(ROOT / 'component_assets/native_manifest.json')
    pi_rows = [x for x in pi['records'] if x.get('cohort') == 'pi_test']
    require(len(pi_rows) == 2 and len({r['group_id'] for r in pi_rows}) == 1 and sum(len(r['objects']) for r in pi_rows) == 189, 'Pi native 2 images 1 group 189 boxes')
    require(all(r['split'] == 'test' for r in pi_rows), 'Pi absent from fine-tuning train and val')
    unknown = sum(o['source_type'] == 'unknown' for r in pi_rows for o in r['excluded_source_objects'])
    require(unknown == 67, 'Pi unknown annotations preserved and excluded')
    job_summary, compact_by_job = {}, {}
    coordinate_pairs = 0
    evaluator_sha = sha(protocol['plan']['evaluator'])
    for model_id, model in aggregate['models'].items():
        locked = selection['selections'][model_id]
        require(model['selection'] == locked, 'aggregate selection: ' + model_id)
        candidates = model['validation_candidates']
        ordered = sorted(candidates, key=lambda c: (-c['bbox']['ap50_95_max100'], -c['bbox']['ap50_95_max300'], c['latency']['offline_predict_and_postprocess_ms_p50'], c['candidate_id']))
        require(ordered[0]['candidate_id'] == locked['candidate']['candidate_id'], 'validation-only ranking: ' + model_id)
        curve = read(Path(locked['validation_metrics_path']).parent / 'validation_threshold_curve.json')
        require(max(curve, key=lambda c: (c['f1'], c['confidence']))['confidence'] == locked['operating_confidence'], 'validation F1 confidence: ' + model_id)
        require(sha(locked['candidate']['weights']) == locked['checkpoint_sha256'], 'checkpoint hash: ' + model_id)
        for compact in candidates + list(model['tests'].values()):
            metric_path = Path(compact['metrics_path'])
            job_id = metric_path.parent.name
            job, metric = jobs[job_id], read(metric_path)
            binding = job['binding']
            compact_by_job[job_id] = compact
            require(sha(metric_path) == compact['metrics_sha256'] == job['metrics_sha256'], 'metric hashes: ' + job_id)
            require(evaluator_sha == binding['evaluator_sha256'], 'frozen evaluator: ' + job_id)
            require(fingerprint(binding) == job['binding_sha256'], 'job binding hash: ' + job_id)
            require(metric['checkpoint_sha256'] == binding['checkpoint_sha256'] and metric['manifest_sha256'] == binding['manifest_sha256'], 'metric binding: ' + job_id)
            for key in ['split', 'cohort', 'task', 'class_names', 'images', 'groups', 'native_gt_instances', 'bbox', 'mask', 'latency', 'group_bootstrap_recall95', 'annotation_limitations']:
                require(metric.get(key) == compact.get(key), 'aggregate ' + key + ': ' + job_id)
            require({k: v for k, v in metric['operating'].items() if k != 'per_image'} == compact['operating'], 'aggregate operating: ' + job_id)
            if binding['split'] == 'test':
                require(binding['candidate_id'] == locked['candidate']['candidate_id'] and binding['checkpoint_sha256'] == locked['checkpoint_sha256'] and binding['confidence'] == metric['operating']['confidence'] == locked['operating_confidence'], 'frozen test checkpoint/confidence: ' + job_id)
                require(binding['tile_size'] == locked['candidate']['tile_size'] and binding['imgsz'] == locked['imgsz'], 'frozen test geometry: ' + job_id)
                require(not (metric_path.parent / 'validation_threshold_curve.json').exists(), 'no test threshold curve: ' + job_id)
            command = job['command']
            manifest_path = Path(command[command.index('--manifest') + 1])
            if str(manifest_path) not in cache:
                cache[str(manifest_path)] = read(manifest_path)
            manifest = cache[str(manifest_path)]
            require(sha(manifest_path) == binding['manifest_sha256'], 'manifest hash: ' + job_id)
            rows = [r for r in manifest['records'] if r['split'] == binding['split'] and (binding['cohort'] is None or r.get('cohort') == binding['cohort'])]
            ground = read(metric_path.parent / 'native_ground_truth.json')
            samples = read(metric_path.parent / 'samples.json')
            predictions = read(metric_path.parent / 'bbox_predictions.json')
            require(len(rows) == len(samples) == len(ground['images']) == metric['images'] and sum(len(r['objects']) for r in rows) == len(ground['annotations']) == metric['native_gt_instances'], 'native ground truth counts: ' + job_id)
            require(len({r['group_id'] for r in rows}) == metric['groups'], 'native groups: ' + job_id)
            require(Counter(o['class_id'] + 1 for r in rows for o in r['objects']) == Counter(o['category_id'] for o in ground['annotations']), 'native per-class counts: ' + job_id)
            sample_map = {r['id']: r for r in samples}
            annotations = defaultdict(list)
            for obj in ground['annotations']:
                annotations[obj['image_id']].append(obj)
            for row in rows:
                sample = sample_map[row['id']]
                require(len(sample['truth']) == len(row['objects']), 'sample truth counts: ' + job_id)
                for truth, obj, annotation in zip(sample['truth'], row['objects'], annotations[sample['image_id']]):
                    require(truth['class_id'] == obj['class_id'] == annotation['category_id'] - 1 and np.allclose(truth['bbox_xyxy'], obj['bbox_xyxy'], atol=1e-8, rtol=0), 'source/sample coordinates: ' + job_id)
                    box = truth['bbox_xyxy']
                    xywh = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                    require(np.allclose(xywh, annotation['bbox'], atol=1e-8, rtol=0), 'native COCO coordinates: ' + job_id)
                    coordinate_pairs += 1
            require(sum(len(r['predictions']) for r in samples) == len(predictions), 'prediction count: ' + job_id)
            counts, sizes, negatives, negative_fp = independent_operating(samples, metric['operating']['confidence'], len(metric['class_names']))
            require(counts == metric['operating']['per_class_counts_tp_fp_fn'], 'independent matching: ' + job_id)
            require(sizes == [[metric['operating']['size_recall'][b]['detected'], metric['operating']['size_recall'][b]['total']] for b in ['lt8', '8to16', '16to32', 'ge32']], 'independent native size bins: ' + job_id)
            require(negatives == metric['operating']['negative_images'] and negative_fp == metric['operating']['negative_images_with_false_positive'], 'independent negative FP image rate: ' + job_id)
            if binding.get('simulation'):
                for row in rows:
                    original = next(x for x in pi_rows if x['id'] == row['simulation_original_id'])
                    require(row['objects'] == original['objects'] and row['width'] == original['width'] and row['height'] == original['height'] and row['simulation_original_sha256'] == original['sha256'], 'simulation identical GT/canvas/source: ' + job_id)
                require(compact['simulation']['camera_proxy_not_D455'] is True and compact['simulation']['native_gt_unchanged'] is True, 'simulation scope: ' + job_id)
            job_summary[job_id] = {'images': len(rows), 'groups': metric['groups'], 'gt': len(ground['annotations']), 'class_gt': dict(Counter(metric['class_names'][o['category_id'] - 1] for o in ground['annotations'])), 'confidence': metric['operating']['confidence'], 'bbox_ap50_95_max100': metric['bbox']['ap50_95_max100'], 'mask_ap50_95_max100': metric['mask']['ap50_95_max100'] if metric['mask'] else None, 'per_class_counts_tp_fp_fn': counts, 'negative_images': negatives, 'negative_images_with_fp': negative_fp}
    ap_rechecks = {}
    for job_id, kind in [('parts_yolo11n__pi_test', 'bbox'), ('parts_yolo11s__pi_test', 'bbox'), ('board_yolo11n_seg__general_test', 'segm')]:
        folder = SUITE / 'evaluations' / job_id
        ground = read(folder / 'native_ground_truth.json')
        predictions = read(folder / ('mask_predictions.json' if kind == 'segm' else 'bbox_predictions.json'))
        with redirect_stdout(io.StringIO()):
            gt = COCO()
            gt.dataset = copy.deepcopy(ground)
            gt.createIndex()
            dt = gt.loadRes(copy.deepcopy(predictions))
            evaluator = COCOeval(gt, dt, kind)
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate()
            evaluator.accumulate()
        precision = evaluator.eval['precision'][:, :, :, 0, 2]
        ap = float(precision[precision > -1].mean())
        reported = compact_by_job[job_id]['mask' if kind == 'segm' else 'bbox']['ap50_95_max100']
        require(math.isclose(ap, reported, abs_tol=1e-12), 'official COCO maxDets[1,10,100]: ' + job_id)
        ap_rechecks[job_id] = {'kind': kind, 'COCOeval_maxDets': [1, 10, 100], 'recomputed_AP50_95': ap}
    registry = read(ROOT / 'model_registry.json')
    best_parts = max(['parts_yolo11n', 'parts_yolo11s'], key=lambda mid: selection['selections'][mid]['validation_bbox']['ap50_95_max100'])
    require(registry['default_parts_model'] == best_parts == 'parts_yolo11s', 'default selected from validation')
    training = {}
    for model_id in models:
        run = read(ROOT / f'runs/{model_id}/run_summary.json')
        training[model_id] = {k: run[k] for k in ['requested_epochs', 'last_epoch', 'actual_optimizer_steps', 'weights_changed']}
        require(run['requested_epochs'] == run['last_epoch'] and run['actual_optimizer_steps'] > 0 and run['weights_changed'], 'real optimizer updates: ' + model_id)
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    require(all(x in readme for x in ['23.15%', '49.21%', '96.28%', 'unknown 67']), 'README core numbers')
    result = {
        'schema': 'r03-independent-final-audit-v1', 'audited_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'METHODOLOGY_CHECKS_PASS',
        'status_scope': 'Evidence consistency and evaluation methodology only; not approval of field accuracy or full annotation correctness',
        'gpu_used': False, 'd455_status': 'NOT_D455_VERIFIED', 'target4_component_masks': 'NOT_TRAINED_NO_HUMAN_TARGET4_MASK_GT',
        'protocol_sha256': aggregate['protocol_sha256'], 'selection_sha256': aggregate['selection_sha256'], 'aggregate_sha256': sha(SUITE / 'aggregate.json'), 'readme_sha256': sha(ROOT / 'README.md'),
        'check_count': check_count, 'failed_checks': [],
        'checks_summary': {'job_count': len(jobs), 'validation_jobs': len(vals), 'test_jobs': len(tests), 'native_gt_coordinate_triplets_compared': coordinate_pairs, 'all_20_operating_counts_independently_recomputed': True, 'all_20_native_shortside_bins_independently_recomputed': True, 'group_and_exact_sha_split_leakage_not_found': True, 'source_physical_independence_verified': False, 'pretraining_overlap_verified': False, 'model_weights_and_all_metrics_hashes_verified': True},
        'chronology': {'last_validation_finished_utc': max(x['finished_utc'] for x in vals), 'selection_frozen_utc': selection['frozen_utc'], 'first_test_started_utc': min(x['started_utc'] for x in tests), 'last_test_finished_utc': max(x['finished_utc'] for x in tests)},
        'native_manifest_counts': manifest_summary, 'training': training, 'jobs': job_summary, 'official_coco_cpu_rechecks': ap_rechecks,
        'critical_interpretation': {
            'default_parts_model': 'parts_yolo11s selected only by validation AP50-95 max100',
            'pi_scope': '2 source images, 1 filename board group; 189 image-level target4 boxes, not unique physical BOM inventory',
            'pi_target4_gt_by_class': dict(Counter(pi['names'][o['class_id']] for r in pi_rows for o in r['objects'])), 'pi_unknown_excluded_count': unknown,
            'pi_s': {'bbox_AP50_95_max100': .231528018955079, 'precision_at_val_conf035': .7209302325581395, 'recall_at_val_conf035': .49206349206349204, 'ic_detected_over_gt': [3, 16], 'connector_detected_over_gt': [0, 12], 'resistor_detected_over_gt': [39, 57], 'capacitor_detected_over_gt': [51, 104]},
            'connector_note': 'Both n and s detect 0/12 connectors at their respective frozen thresholds. Nonzero AP integrates predictions across lower scores and is not contradictory.',
            'board_mask_note': '96.28% mask AP evaluates whole-board polygons, not target4 masks. Detect and segment tests differ; their AP is not a controlled architecture comparison.',
            'size_note': 'Pi s detects 19/77 at native short side [8,16), 60/72 at[16,32), and14/38 at>=32px. Class mix matters; no monotonic hard pixel threshold is established.',
            'stress_note': 'Same189 GT on deterministic rescale-upsample or sigma1 blur is synthetic sensitivity, never D455 evidence.',
            'bootstrap_note': 'General group-bootstrap intervals estimate fixed-confidence recall only, not AP. Pi one-group intervals omitted.',
            'negative_fp_note': 'Board detect2/469 and board seg3/324 negative images contain >=1prediction at frozen confidence; image rates differ from total FP detection counts.',
            'annotation_note': 'Source labels not newly expert-adjudicated. Unknowns may cause apparent FP or omissions; no inference-based replacement GT was used.',
            'license_note': 'WACV explicit redistribution license unconfirmed; native photos not repackaged as redistributable dataset.',
            'readme_review': 'Core numbers and limitations match frozen evidence. Lead should distinguish whole-board success on source holdouts from weak IC/connector recall and unverified D455/target4 masks.'
        }
    }
    output = ROOT / 'reports/final_evaluation_audit.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'checks': check_count, 'jobs': len(jobs), 'coordinate_triplets': coordinate_pairs, 'chronology': result['chronology'], 'official_coco_cpu_rechecks': ap_rechecks, 'output': str(output)}, indent=2))

if __name__ == '__main__':
    main()
