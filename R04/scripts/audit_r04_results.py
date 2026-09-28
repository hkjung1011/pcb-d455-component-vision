"""Independent CPU audit of completed R04 training and saved evaluation results.

Never trains, predicts or opens a camera. Uses a separate native-IoU matcher and
direct pycocotools accumulation; no evaluator/orchestrator scoring code imported.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import redirect_stdout
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import numpy as np

NAMES = ['resistor', 'capacitor', 'ic', 'connector']
BINS = ['lt8', '8to16', '16to32', 'ge32']
ARMS = ['baseline', 'improved']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def stamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError('Timezone-qualified evidence timestamp required')
    return parsed


def near(a, b, tolerance=1e-10):
    if a is None or b is None:
        return a is b
    return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance


def independent_match(samples, threshold):
    """Separate vectorized-IoU implementation, stable confidence prefix matching."""
    total = np.zeros((4, 3), dtype=np.int64)
    cross = np.zeros((4, 4, 2), dtype=np.int64)  # class,size,detected/total
    per_image = []
    for sample in samples:
        truths = sample['truth']
        classes = np.array([g['class_id'] for g in truths], dtype=int)
        boxes = np.array([g['bbox_xyxy'] for g in truths], dtype=float).reshape((-1, 4))
        found = np.zeros(len(truths), dtype=bool)
        counts = np.zeros((4, 3), dtype=np.int64)
        for pred in sorted(sample['predictions'], key=lambda p: -p['score']):
            if pred['score'] < threshold:
                break
            cls = pred['class_id']
            pred_box = np.array(pred['bbox_xyxy'], dtype=float)
            intersection = np.maximum(0, np.minimum(boxes[:, 2:], pred_box[2:])-np.maximum(boxes[:, :2], pred_box[:2])).prod(axis=1)
            union = (boxes[:, 2:]-boxes[:, :2]).prod(axis=1)+(pred_box[2:]-pred_box[:2]).prod()-intersection
            ious = np.divide(intersection, union, out=np.zeros(len(truths)), where=union > 0)
            eligible = np.flatnonzero((classes == cls) & ~found & (ious >= .5))
            if len(eligible):
                maximum = ious[eligible].max()
                # R03/R04 break exact IoU ties at the highest native GT index.
                index = int(eligible[ious[eligible] == maximum][-1])
                found[index] = True
                counts[cls, 0] += 1
            else:
                counts[cls, 1] += 1
        for index, truth in enumerate(truths):
            cls = truth['class_id']
            if not found[index]:
                counts[cls, 2] += 1
            box = truth['bbox_xyxy']; short = min(box[2]-box[0], box[3]-box[1])
            bucket = 0 if short < 8 else 1 if short < 16 else 2 if short < 32 else 3
            cross[cls, bucket, 0] += int(found[index])
            cross[cls, bucket, 1] += 1
        total += counts
        per_image.append({'id': sample['id'], 'counts_tp_fp_fn': counts.tolist()})
    tp, fp, fn = map(int, total.sum(axis=0))
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else None
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1,
            'counts': total.tolist(), 'class_by_size_detected_total': cross.tolist(), 'per_image': per_image}


def independent_grid(samples):
    """Match confidence-sorted prefixes once; lower-score suffixes cannot alter prefixes."""
    thresholds = np.arange(1, 20)/20.
    counts = np.zeros((19, 4, 3), dtype=np.int64)
    gt_counts = np.zeros(4, dtype=np.int64)
    for sample in samples:
        truths = sample['truth']; classes = np.array([g['class_id'] for g in truths], dtype=int)
        boxes = np.array([g['bbox_xyxy'] for g in truths], dtype=float).reshape((-1, 4))
        gt_counts += np.bincount(classes, minlength=4)
        found = np.zeros(len(truths), dtype=bool)
        for pred in sorted(sample['predictions'], key=lambda p: -p['score']):
            if pred['score'] < .05: break
            cls = pred['class_id']; box = np.array(pred['bbox_xyxy'], dtype=float)
            overlap = np.maximum(0, np.minimum(boxes[:, 2:], box[2:])-np.maximum(boxes[:, :2], box[:2])).prod(axis=1)
            union = (boxes[:, 2:]-boxes[:, :2]).prod(axis=1)+(box[2:]-box[:2]).prod()-overlap
            ious = np.divide(overlap, union, out=np.zeros(len(truths)), where=union > 0)
            choices = np.flatnonzero((classes == cls) & ~found & (ious >= .5))
            is_tp = bool(len(choices))
            if is_tp:
                found[int(choices[ious[choices] == ious[choices].max()][-1])] = True
            counts[thresholds <= pred['score'], cls, 0 if is_tp else 1] += 1
    counts[:, :, 2] = gt_counts[np.newaxis, :]-counts[:, :, 0]
    return [row.tolist() for row in counts]


def independent_coco(gt_data, pred_data):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = copy.deepcopy(gt_data); gt.createIndex()
        if pred_data:
            pred = gt.loadRes(copy.deepcopy(pred_data))
        else:
            pred = COCO(); pred.dataset = {'images': gt_data['images'], 'categories': gt_data['categories'], 'annotations': []}; pred.createIndex()
        calc = COCOeval(gt, pred, 'bbox'); calc.params.maxDets = [1, 100, 300]
        calc.evaluate(); calc.accumulate()
    p = calc.eval['precision']
    def avg(array):
        values = array[array >= 0]
        return float(values.mean()) if values.size else None
    output = {}
    for maxdet, slot in [(100, 1), (300, 2)]:
        output[f'ap50_95_max{maxdet}'] = avg(p[:, :, :, 0, slot])
        output[f'ap50_max{maxdet}'] = avg(p[0, :, :, 0, slot])
        output[f'ap75_max{maxdet}'] = avg(p[5, :, :, 0, slot])
    output['per_class'] = {}
    for cls, name in enumerate(NAMES):
        output['per_class'][name] = {f'ap50_95_max{maxdet}': avg(p[:, :, cls, 0, slot]) for maxdet, slot in [(100, 1), (300, 2)]}
    return output


class Audit:
    def __init__(self, root):
        self.root = root
        self.report = {'schema': 'r04-independent-results-audit-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                       'status': 'RUNNING', 'scope': 'CPU saved-result audit; no inference/training/camera operation',
                       'checks_passed': [], 'failures': [], 'source_sha256': {}, 'training': {}, 'evaluations': {}}

    def require(self, condition, message):
        if not condition:
            raise ValueError(message)
        self.report['checks_passed'].append(message)

    def load(self, path):
        path = Path(path)
        self.report['source_sha256'][str(path.resolve())] = sha(path)
        return read(path)

    def check_hash(self, path, expected):
        actual = sha(path)
        self.require(actual == expected, f'SHA verified: {path}')
        self.report['source_sha256'][str(Path(path).resolve())] = actual

    def source_truth(self):
        root = self.root
        manifest = self.load(root/'data/native_manifest.json')
        originals = self.load(root/'data/original_records.json')
        old = self.load(root.parent/'r03/component_assets/native_manifest.json')
        original_by_id = {r['image_id']: r for r in originals['images']}
        old_by_id = {r['id']: r for r in old['records']}
        self.require(manifest['names'] == NAMES, 'Four-class ontology unchanged')
        self.require(len(manifest['records']) == len(original_by_id) == len(old_by_id), 'Native image inventory preserved')
        groups = {}
        for row in manifest['records']:
            original, prior = original_by_id[row['id']], old_by_id[row['id']]
            self.require(row['objects'] == original['objects'] == prior['objects'], f'Source target boxes unchanged: {row["id"]}')
            self.require(row.get('excluded_source_objects', []) == original.get('excluded_source_objects', []) == prior.get('excluded_source_objects', []),
                         f'Unknown/non-target source definitions preserved: {row["id"]}')
            self.require(row['sha256'] == original['image_sha256'] == prior['sha256'], f'Original image SHA identity: {row["id"]}')
            if prior['split'] == 'test':
                self.require(row['split'] == 'test', f'Prior holdout excluded from new training: {row["id"]}')
                self.require(row['cohort'] == ('pi_test' if prior['cohort'] == 'pi_test' else 'general_test'), f'Holdout cohort preserved: {row["id"]}')
            self.require(groups.get(row['group_id'], row['split']) == row['split'], f'No source group split leakage: {row["id"]}')
            groups[row['group_id']] = row['split']
        self.require(manifest['annotation_limitations']['pi3b_unknown_source_objects'] == 67, 'Pi unknown source count remains 67')
        self.require(manifest['annotation_limitations']['d455_verified'] is False, 'D455 remains unverified in source manifest')
        self.report['source_definition'] = {'target4_objects': sum(len(r['objects']) for r in manifest['records']),
                                            'new_manual_labels': 0, 'new_masks': 0, 'pi_unknown_source_objects': 67,
                                            'policy': 'Source target4 and excluded objects unchanged; no DNP/unknown repair of evaluation GT'}
        return manifest

    def training(self, protocol):
        root = self.root
        initial_hashes, initial_parameters, final_calls, final_draws = [], [], [], []
        os.environ.setdefault('YOLO_CONFIG_DIR', str(root/'runtime_config'))
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import torch
        for arm in ARMS:
            folder = root/'runs'/arm
            start = self.load(folder/'start.json'); summary = self.load(folder/'training_summary.json')
            epochs = self.load(folder/'epochs.json'); exposure = self.load(folder/'sampled_exposures.json')
            candidates = self.load(folder/'candidates.json')
            views = {r['id']: r for r in self.load(root/'data'/arm/'views.json')['records']}
            self.require(summary['status'] == 'COMPLETE' and not (folder/'failure.json').exists(), f'{arm}: completed training, no failure record')
            self.require(summary['actual_direct_optimizer_calls'] == protocol['max_direct_optimizer_calls_per_arm'] == 1165, f'{arm}: exactly 1165 directly observed calls')
            self.require(1 <= summary['actual_epochs_reached'] <= protocol['max_epochs_per_arm'] <= 20, f'{arm}: authorized epoch bound')
            self.require([e['epoch'] for e in epochs] == list(range(1, summary['actual_epochs_reached']+1)), f'{arm}: real epochs only, no final-evaluation pseudo epoch')
            self.require(summary['initial_parameter_sha256'] != summary['final_parameter_sha256'], f'{arm}: measured parameter change')
            self.require(start['initial_model_parameter_sha256'] == summary['initial_parameter_sha256'], f'{arm}: initial parameter fingerprints consistent')
            self.require(start['initial_checkpoint_sha256'] == protocol['initial_weights_sha256'], f'{arm}: same official pretrained checkpoint')
            self.require(start['protocol_sha256'] == summary['protocol_sha256'] == sha(root/'protocol.json'), f'{arm}: frozen training protocol binding')
            self.check_hash(root/'scripts/train_r04.py', start['training_script_sha256'])
            self.check_hash(root/'scripts/ignore_adapter.py', start['ignore_adapter_sha256'])
            self.check_hash(root/'data'/arm/'data.yaml', start['data_yaml_sha256'])
            self.check_hash(root/'data'/arm/'ignore_regions.json', start['sidecar_sha256'])
            self.check_hash(folder/'candidates.json', summary['candidates_manifest_sha256'])
            self.require(summary['class_weights'] == [1., 1., 1., 1.], f'{arm}: explicit unit class multipliers')
            nb, batch = start['loader_batches_per_epoch'], start['actual_batch']
            self.require(nb*batch == protocol['epoch_replacement_draws'] and batch == protocol['batch'], f'{arm}: fixed sampled draws/batch plan')
            expected_calls, last_step = 0, -1
            accumulate = max(round(protocol['nbs']/batch), 1)
            warmup_iterations = round(min(protocol['warmup_epochs'], protocol['max_epochs_per_arm']-1)*nb)
            for row in epochs:
                count = row['processed_batches']
                self.require(0 < count <= row['planned_batches'] == nb, f'{arm} epoch{row["epoch"]}: actual/planned batches valid')
                self.require(row['partial_epoch'] == (count != nb), f'{arm} epoch{row["epoch"]}: partial-epoch flag correct')
                for batch_index in range(count):
                    ni = (row['epoch']-1)*nb+batch_index
                    if ni < warmup_iterations:
                        accumulate = max(1, int(np.interp(ni, [0, warmup_iterations], [1, protocol['nbs']/batch]).round()))
                    if ni-last_step >= accumulate:
                        expected_calls += 1; last_step = ni
                self.require(row['optimizer_calls'] == expected_calls == row['ema_updates'], f'{arm} epoch{row["epoch"]}: direct calls, schedule and EMA counts agree')
                steps = row['parameter_steps']
                self.require(0 < steps['min'] <= steps['max'] <= row['optimizer_calls'], f'{arm} epoch{row["epoch"]}: per-parameter counters separately bounded')
            self.require(sum(e['processed_batches'] for e in epochs) == summary['actual_batches'], f'{arm}: actual batch total')
            self.require(epochs[-1]['partial_epoch'] == summary['last_epoch_partial'], f'{arm}: final partial-epoch summary')
            self.require(epochs[-1]['parameter_steps'] == summary['parameter_steps'] and epochs[-1]['optimizer_calls'] == summary['actual_direct_optimizer_calls'], f'{arm}: final counters unchanged after post-training val')
            groups, kinds, classes, ignores = Counter(), Counter(), Counter(), Counter()
            ids, complete_ids = set(), set()
            for key, repeats in exposure['view_draw_counts'].items():
                self.require(key in views and isinstance(repeats, int) and repeats > 0, f'{arm}: valid sampled view {key}')
                view = views[key]
                groups[view['group_id']] += repeats; kinds[view['kind']] += repeats
                for target in view['targets']:
                    classes[str(target['class_id'])] += repeats; ids.add(target['instance_id'])
                    if target['visible_fraction'] >= 1-1e-8: complete_ids.add(target['instance_id'])
                for region in view['ignore_regions']: ignores[region['reason']] += repeats
            draws = sum(exposure['view_draw_counts'].values())
            self.require(draws == exposure['draws'] == summary['actual_batches']*batch, f'{arm}: actual sampled draws equal processed full batches')
            for label, value in [('groups', groups), ('view_types', kinds), ('class_exposures', classes), ('ignore_exposures', ignores)]:
                self.require(dict(value) == exposure[label], f'{arm}: independently recounted {label}')
            self.require(len(ids) == exposure['unique_positive_source_instances_seen'] and len(complete_ids) == exposure['unique_positive_source_instances_seen_complete'], f'{arm}: unique source instances versus repeated exposures')
            self.require(len(exposure['view_draw_counts']) == exposure['unique_views_drawn'], f'{arm}: sampled unique-view count')
            self.require([r['epoch'] for r in candidates] == protocol['save_epoch_candidates'], f'{arm}: complete declared candidate set')
            checkpoint_audit = []
            for row in candidates:
                path = Path(row['path']); path = path if path.is_absolute() else root/path
                self.check_hash(path, row['sha256'])
                ck = torch.load(path, map_location='cpu', weights_only=False)
                saved = ck['r04']; ep = epochs[row['epoch']-1]
                self.require(ck['epoch']+1 == saved['actual_epoch'] == row['epoch'], f'{arm} candidate{row["epoch"]}: unambiguous saved epoch')
                self.require(saved['direct_optimizer_calls'] == row['optimizer_calls'] == ep['optimizer_calls'] == ck['updates'], f'{arm} candidate{row["epoch"]}: actual calls and EMA source provenance')
                self.require(saved['partial_epoch'] == row['partial_epoch'] == ep['partial_epoch'], f'{arm} candidate{row["epoch"]}: partial provenance')
                self.require(saved['protocol_sha256'] == sha(root/'protocol.json') and saved['arm'] == arm, f'{arm} candidate{row["epoch"]}: protocol/arm provenance')
                model = ck['model']
                self.require(type(model).__name__ == 'DetectionModel' and type(model).__module__.startswith('ultralytics.'), f'{arm} candidate{row["epoch"]}: standard Ultralytics detection model')
                self.require(ck.get('ema') is None and ck.get('optimizer') is None and getattr(model, 'criterion', None) is None, f'{arm} candidate{row["epoch"]}: EMA weights saved without custom loss/optimizer')
                self.require([model.names[i] for i in range(4)] == NAMES, f'{arm} candidate{row["epoch"]}: target4 names')
                self.require(all(p.device.type == 'cpu' and torch.isfinite(p).all() for p in model.parameters()), f'{arm} candidate{row["epoch"]}: finite CPU-loaded weights')
                checkpoint_audit.append({'epoch': row['epoch'], 'optimizer_calls': row['optimizer_calls'], 'sha256': row['sha256'], 'partial_epoch': row['partial_epoch'],
                                         'model_type': f'{type(model).__module__}.{type(model).__name__}', 'ema_updates': ck['updates']})
                del model, ck
            initial_hashes.append(start['initial_checkpoint_sha256']); initial_parameters.append(summary['initial_parameter_sha256'])
            final_calls.append(summary['actual_direct_optimizer_calls']); final_draws.append(draws)
            self.report['training'][arm] = {'epochs_reached': summary['actual_epochs_reached'], 'last_epoch_partial': summary['last_epoch_partial'],
                                            'direct_optimizer_calls': summary['actual_direct_optimizer_calls'], 'parameter_steps': summary['parameter_steps'],
                                            'ema_updates': summary['ema_updates'], 'processed_batches': summary['actual_batches'], 'actual_draws': draws,
                                            'actual_class_exposures': dict(classes), 'actual_ignore_exposures': dict(ignores),
                                            'checkpoints': checkpoint_audit, 'finished_utc': summary['finished_utc']}
        self.require(len(set(initial_hashes)) == len(set(initial_parameters)) == len(set(final_calls)) == len(set(final_draws)) == 1,
                     'Both arms have identical official initialization, initialized model parameters, direct calls and actual draws')

    def evaluation_job(self, item, manifest, validate_grid=False):
        metric_path = Path(item['metrics_path']); folder = metric_path.parent
        metric = self.load(metric_path)
        self.check_hash(metric_path, item['metrics_sha256'])
        job_path = Path(item['job_path']); job = self.load(job_path)
        self.check_hash(job_path, item['job_sha256'])
        self.require(job['status'] == 'completed', f'{folder.name}: completed job')
        candidate = item['candidate']
        trained = [r for r in self.report['training'][candidate['arm']]['checkpoints'] if r['epoch'] == candidate['epoch']]
        self.require(len(trained) == 1 and trained[0]['sha256'] == candidate['checkpoint_sha256'] == metric['checkpoint_sha256']
                     and trained[0]['optimizer_calls'] == candidate['optimizer_calls'], f'{folder.name}: evaluation uses audited trained candidate')
        self.require(metric['pipeline'] == candidate['pipeline'], f'{folder.name}: candidate pipeline binding')
        for filename, expected in job['output_sha256'].items(): self.check_hash(folder/filename, expected)
        for key in ['checkpoint_sha256', 'manifest_sha256', 'evaluator_sha256', 'split', 'cohort', 'pipeline']:
            self.require(metric[key] == job['binding'][key], f'{folder.name}: bound {key}')
        self.require(metric['manifest_sha256'] == sha(self.root/'data/native_manifest.json'), f'{folder.name}: current frozen native manifest')
        self.require(metric['d455_verified'] is False and metric['mask'] is None and metric['per_class_thresholds_applied'] is False, f'{folder.name}: truthful target4 bbox-only/offline scope')
        samples = self.load(folder/'samples.json'); gt = self.load(folder/'native_ground_truth.json'); preds = self.load(folder/'bbox_predictions.json')
        rows = [r for r in manifest['records'] if r['split'] == metric['split'] and (not metric['cohort'] or r['cohort'] == metric['cohort'])]
        self.require([s['id'] for s in samples] == [r['id'] for r in rows], f'{folder.name}: exact native evaluation cohort order')
        expected_annotations, expected_predictions = [], []
        class_counts = Counter()
        for image_id, (row, sample) in enumerate(zip(rows, samples), 1):
            expected_truth = [{'class_id': o['class_id'], 'bbox_xyxy': list(map(float, o['bbox_xyxy']))} for o in row['objects']]
            self.require(sample['truth'] == expected_truth and sample['group_id'] == row['group_id'], f'{folder.name}/{row["id"]}: native source truth and group unchanged')
            image = gt['images'][image_id-1]
            self.require((image['id'], image['width'], image['height']) == (image_id, row['width'], row['height']), f'{folder.name}/{row["id"]}: COCO native dimensions')
            for obj in row['objects']:
                x1,y1,x2,y2 = map(float, obj['bbox_xyxy']); class_counts[NAMES[obj['class_id']]] += 1
                expected_annotations.append({'id': len(expected_annotations)+1, 'image_id': image_id, 'category_id': obj['class_id']+1,
                                             'bbox': [x1,y1,x2-x1,y2-y1], 'area': (x2-x1)*(y2-y1), 'iscrowd': 0})
            bounded_predictions = True
            for pred in sample['predictions']:
                x1,y1,x2,y2 = pred['bbox_xyxy']
                bounded_predictions = bounded_predictions and 0 <= x1 < x2 <= row['width'] and 0 <= y1 < y2 <= row['height'] and 0 <= pred['score'] <= 1
                expected_predictions.append({'image_id': image_id, 'category_id': pred['class_id']+1, 'bbox': [x1,y1,x2-x1,y2-y1], 'score': pred['score']})
            self.require(bounded_predictions, f'{folder.name}/{row["id"]}: all predictions bounded in native coordinates')
        self.require(gt['categories'] == [{'id': i+1, 'name': n} for i,n in enumerate(NAMES)] and gt['annotations'] == expected_annotations,
                     f'{folder.name}: COCO GT exact source conversion, no ignored/DNP/unknown relabels')
        self.require(preds == expected_predictions, f'{folder.name}: COCO predictions exactly equal saved native samples')
        recomputed = independent_coco(gt, preds)
        for key, value in recomputed.items():
            if key == 'per_class':
                for cls in NAMES:
                    for name, score in value[cls].items(): self.require(near(score, metric['bbox'][key][cls][name]), f'{folder.name}: independent COCO {cls}/{name}')
            else: self.require(near(value, metric['bbox'][key]), f'{folder.name}: independent COCO {key}')
        computed = independent_match(samples, metric['operating']['confidence'])
        self.require(computed['counts'] == metric['operating']['per_class_counts_tp_fp_fn'], f'{folder.name}: independent class TP/FP/FN')
        for key in ['tp','fp','fn','precision','recall','f1']:
            self.require(near(computed[key], metric['operating'][key]), f'{folder.name}: independent operating {key}')
        self.require([x['counts_tp_fp_fn'] for x in computed['per_image']] == [x['counts_tp_fp_fn'] for x in metric['operating']['per_image']], f'{folder.name}: image-level matching parity')
        cross = np.array(computed['class_by_size_detected_total'])
        for cls,name in enumerate(NAMES):
            for size,bucket in enumerate(BINS):
                cell=metric['operating']['class_by_native_shortside_recall'][name][bucket]
                self.require(cross[cls,size].tolist() == [cell['detected'],cell['total']], f'{folder.name}: {name}/{bucket} GT/detection totals')
            self.require(int(cross[cls,:,1].sum()) == class_counts[name], f'{folder.name}: class-by-size totals equal source {name}')
        for size,bucket in enumerate(BINS):
            cell=metric['operating']['size_recall'][bucket]
            self.require(cross[:,size,:].sum(axis=0).tolist() == [cell['detected'],cell['total']], f'{folder.name}: aggregate size {bucket}')
        if validate_grid:
            stored_curve = self.load(folder/'validation_threshold_curve.json')
            stored_pr = self.load(folder/'validation_per_class_pr.json')
            self.require(len(stored_curve) == 19 and stored_pr['per_class_thresholds_applied'] is False, f'{folder.name}: complete diagnostic PR grid')
            f1s=[]
            independent_counts = independent_grid(samples)
            for index in range(19):
                threshold=(index+1)/20.; counts = independent_counts[index]
                self.require(stored_curve[index]['confidence'] == threshold and counts == stored_curve[index]['per_class_counts_tp_fp_fn'], f'{folder.name}: independently recounted threshold {threshold}')
                # Match the evaluator arithmetic order for exact tie-breaking; all counts are independently computed.
                tp,fp,fn = map(int,np.array(counts).sum(axis=0))
                p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else None
                f1=2*p*r/(p+r) if r is not None and p+r else 0.
                self.require(near(f1,stored_curve[index]['f1']),f'{folder.name}: threshold F1 {threshold}')
                f1s.append((f1,threshold))
                for cls,name in enumerate(NAMES):
                    actual=stored_pr['per_class_curves'][name][index]
                    self.require([actual['tp'],actual['fp'],actual['fn']] == counts[cls] and actual['confidence'] == threshold,
                                 f'{folder.name}: diagnostic class PR {name}/{threshold}')
            self.require(metric['operating']['confidence'] == max(f1s)[1] and metric['threshold_source'] == 'validation_micro_f1_grid',
                         f'{folder.name}: confidence is grid-best validation global micro-F1')
        else:
            self.require(metric['evaluation_role'] == 'development_holdout_not_pristine_final' and metric['threshold_source'] == 'supplied_frozen_confidence',
                         f'{folder.name}: inspected development holdout, no threshold optimization')
            if metric['cohort'] == 'pi_test':
                self.require(metric['groups'] == 1 and metric['group_bootstrap_recall95'] is None and bool(metric['group_bootstrap_exclusion_reason']),
                             f'{folder.name}: no invalid single-board CI')
            else:
                self.require(metric['group_bootstrap_recall95']['resamples'] == 200 and metric['group_bootstrap_recall95']['groups'] == metric['groups'],
                             f'{folder.name}: declared general-holdout bootstrap scope')
        self.report['evaluations'][folder.name] = {'ap100': recomputed['ap50_95_max100'], 'ap300': recomputed['ap50_95_max300'],
                                                  'ap50_max300': recomputed['ap50_max300'], 'confidence': metric['operating']['confidence'],
                                                  'class_counts': dict(class_counts), 'operating_counts': computed['counts'],
                                                  'pipeline': metric['pipeline'], 'independent_recalculation': 'PASS'}
        print(f'AUDITED {folder.name}', flush=True)
        return metric,job

    def run(self):
        root=self.root; suite=root/'reports/evaluation_suite'
        protocol=self.load(root/'protocol.json')
        manifest=self.source_truth()
        self.check_hash(root/'data/native_manifest.json',protocol['native_manifest_sha256'])
        self.check_hash(root/'data/summary.json',protocol['data_summary_sha256'])
        self.training(protocol)
        aggregate=self.load(suite/'aggregate.json'); frozen=self.load(suite/'selections_frozen.json')
        eval_protocol=self.load(suite/'protocol.json')
        self.require(aggregate['status']=='COMPLETED_DEVELOPMENT_HOLDOUT_NOT_D455_VALIDATED','Evaluation suite finished all development holdouts')
        self.require(aggregate['d455_verified'] is False and aggregate['fresh_final_test_available'] is False,'No fresh final/D455 claim')
        self.check_hash(suite/'selections_frozen.json',aggregate['selection_sha256'])
        self.require(frozen['test_results_used_for_selection'] is False and frozen['protocol_sha256']==aggregate['protocol_sha256']==eval_protocol['protocol_sha256'],
                     'Frozen selection/evaluation protocol binding')
        self.check_hash(root/'protocol.json',eval_protocol['evidence']['training_protocol_sha256'])
        frozen_time=stamp(frozen['frozen_utc']); winners=[]
        for arm in ARMS:
            entry=aggregate['arms'][arm]; candidates=entry['validation_candidates']; choices=[]
            expected={f'{arm}__epoch_{e:02d}__{p}' for e in protocol['save_epoch_candidates'] for p in ['native','dual']}
            self.require({c['candidate_id'] for c in candidates}==expected and len(candidates)==len(expected),f'{arm}: every epoch/pipeline validation candidate present')
            for candidate in candidates:
                metric,job=self.evaluation_job(candidate,manifest,True)
                self.require(stamp(job['finished_utc'])<=frozen_time,f'{candidate["candidate_id"]}: validation completed before freeze')
                self.require(all(stamp(t['finished_utc'])<=stamp(job['started_utc']) for t in self.report['training'].values()),f'{candidate["candidate_id"]}: both arms trained before evaluation')
                key=(-metric['bbox']['ap50_95_max300'],-metric['bbox']['ap50_95_max100'],metric['latency']['offline_predict_and_postprocess_ms_p50'],candidate['candidate_id'])
                choices.append((key,candidate,metric))
            key,winner,wm=min(choices,key=lambda x:x[0]); winners.append((key,arm,winner['candidate_id']))
            selected=frozen['selections'][arm]
            self.require(selected==entry['selection'],f'{arm}: aggregate selection matches frozen record')
            self.require(selected['candidate']['candidate_id']==winner['candidate_id'] and selected['validation_metrics_sha256']==winner['metrics_sha256'],f'{arm}: VAL-only AP300/AP100/latency/id selection rank')
            self.require(selected['operating_confidence']==wm['operating']['confidence'],f'{arm}: frozen global confidence from winning val')
            for cohort in ['general_test','pi_test']:
                item=entry['tests'][cohort]; metric,job=self.evaluation_job(item,manifest)
                self.require(stamp(job['started_utc'])>=frozen_time,f'{arm}/{cohort}: holdout started after selection freeze')
                self.require(job['binding']['selection_sha256']==aggregate['selection_sha256'],f'{arm}/{cohort}: holdout bound to frozen selections')
                self.require(metric['checkpoint_sha256']==selected['checkpoint_sha256'] and metric['pipeline']==selected['candidate']['pipeline'] and metric['operating']['confidence']==selected['operating_confidence'],
                             f'{arm}/{cohort}: frozen checkpoint/pipeline/threshold unchanged')
            self.report['training'][arm]['selected_epoch']=selected['candidate']['epoch']
            self.report['training'][arm]['selected_pipeline']=selected['candidate']['pipeline']
        _,recommended,candidate_id=min(winners,key=lambda x:x[0])
        self.require(recommended==frozen['recommended_arm']==aggregate['recommended_arm'] and candidate_id==frozen['recommended_candidate_id']==aggregate['recommended_candidate_id'],
                     'Overall recommended arm uses validation only')
        jobs=[self.load(path) for path in sorted((suite/'jobs').glob('*.json'))]
        test_jobs=[job for job in jobs if job['binding']['split']=='test']
        self.require(len(test_jobs)==4 and Counter((job['binding']['arm'],job['binding']['cohort']) for job in test_jobs)==Counter((a,c) for a in ARMS for c in ['general_test','pi_test']),
                     'Exactly one recorded general and Pi holdout job per selected arm; no extra threshold search')
        self.report.update(status='PASS_SAVED_RESULTS_AUDIT_NOT_D455_VALIDATED',recommended_arm=recommended,
                           limitation='Checks prove saved evidence/selection arithmetic consistency, not physical camera success or fresh-test generalization',
                           finished_utc=datetime.now(timezone.utc).isoformat())

    def write(self):
        reports=self.root/'reports';reports.mkdir(parents=True,exist_ok=True)
        (reports/'final_results_audit.json').write_text(json.dumps(self.report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        lines=['# R04 저장 결과 독립 감사','',f'상태: **{self.report["status"]}**','',
               'GPU 추론·재학습 없이 checkpoint를 CPU로 읽고, 저장 예측의 COCO와 별도 구현 매칭을 재계산했다. 실제 D455나 새로운 최종 시험을 검증한 결과는 아니다.','',
               f'통과한 검사: {len(self.report["checks_passed"])}개. 실패: {len(self.report["failures"])}개.','',
               '| arm | 도달 epoch | 마지막 partial | 직접 optimizer 호출 | 파라미터 step 최소/최대 | 실제 draw | 선택 |',
               '|---|---:|---|---:|---|---:|---|']
        for arm,t in self.report['training'].items():
            lines.append(f'| {arm} | {t["epochs_reached"]} | {t["last_epoch_partial"]} | {t["direct_optimizer_calls"]} | {t["parameter_steps"]["min"]}/{t["parameter_steps"]["max"]} | {t["actual_draws"]} | epoch{t.get("selected_epoch","pending")} / {t.get("selected_pipeline","pending")} |')
        lines+=['','검사 범위: 동일 초기 가중치와 실제 초기 파라미터, 직접 호출·EMA·파라미터 step 분리, 반복 라벨 노출, 후보 SHA·표준 모델, 전체 VAL 후보 순위와 confidence grid, 선택 동결 시각, source GT·unknown 정의, 저장 예측 AP/TP/FP/FN와 클래스×크기 합계.','',
                '일반/Pi 자료는 R03 진단에 사용한 개선용 holdout이다. unknown·미실장 footprint 해석을 임의 수정하지 않았다. 새 mask나 수작업 라벨은 추가하지 않았다.']
        if self.report['failures']: lines+=['','실패 항목:']+['- '+x for x in self.report['failures']]
        lines+=['','상세 수치·파일 SHA·개별 검사 결과는 `final_results_audit.json`에 기록했다.']
        (reports/'final_results_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def self_test():
    import unittest
    class Tests(unittest.TestCase):
        def test_matching_duplicate_and_class(self):
            s={'id':'x','truth':[{'class_id':0,'bbox_xyxy':[0,0,10,10]}],
               'predictions':[{'class_id':0,'bbox_xyxy':[0,0,10,10],'score':.9},{'class_id':0,'bbox_xyxy':[0,0,10,10],'score':.8},
                              {'class_id':1,'bbox_xyxy':[0,0,10,10],'score':.7}]}
            result=independent_match([s],.5)
            self.assertEqual((result['tp'],result['fp'],result['fn']),(1,2,0))
            self.assertEqual(result['class_by_size_detected_total'][0][1],[1,1])
        def test_exact_confidence(self):
            s={'id':'x','truth':[{'class_id':2,'bbox_xyxy':[0,0,4,4]}],
               'predictions':[{'class_id':2,'bbox_xyxy':[0,0,4,4],'score':.35}]}
            self.assertEqual(independent_match([s],.35)['tp'],1)
            self.assertEqual(independent_match([s],.350001)['tp'],0)
        def test_empty(self):
            self.assertIsNone(independent_match([{'id':'e','truth':[],'predictions':[]}],.5)['recall'])
        def test_dense_coco_independent(self):
            gt={'info':{},'images':[{'id':1,'width':2000,'height':100}],
                'categories':[{'id':i+1,'name':n} for i,n in enumerate(NAMES)],'annotations':[]};pred=[]
            for i in range(150):
                box=[i*10,0,5,5];gt['annotations'].append({'id':i+1,'image_id':1,'category_id':1,'bbox':box,'area':25,'iscrowd':0})
                pred.append({'image_id':1,'category_id':1,'bbox':box,'score':1-i/1000})
            result=independent_coco(gt,pred)
            self.assertGreater(result['ap50_95_max300'],.999)
            self.assertLess(result['ap50_95_max100'],.68)
            self.assertIsNone(result['per_class']['ic']['ap50_95_max300'])
        def test_timestamp_requires_zone(self):
            with self.assertRaises(ValueError): stamp('2026-09-28T12:00:00')
        def test_confidence_prefix_recount_matches_separate_matching(self):
            samples=[{'id':'a','truth':[{'class_id':0,'bbox_xyxy':[0,0,10,10]},{'class_id':2,'bbox_xyxy':[20,20,30,30]}],
                      'predictions':[{'class_id':0,'bbox_xyxy':[0,0,10,10],'score':.35},
                                     {'class_id':0,'bbox_xyxy':[0,0,10,10],'score':.34},
                                     {'class_id':1,'bbox_xyxy':[20,20,30,30],'score':.8},
                                     {'class_id':2,'bbox_xyxy':[20,20,30,30],'score':.15}]}]
            curves=independent_grid(samples)
            for i in range(19): self.assertEqual(curves[i],independent_match(samples,(i+1)/20.)['counts'])
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful(): raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test: return self_test()
    audit=Audit(args.root.resolve())
    try:
        audit.run()
    except BaseException as error:
        audit.report['status']='FAIL_OR_INCOMPLETE_SAVED_RESULTS_AUDIT'
        audit.report['failures'].append(f'{type(error).__name__}: {error}')
        audit.write()
        raise
    audit.write()
    print(json.dumps({'status':audit.report['status'],'checks_passed':len(audit.report['checks_passed']),
                      'report':str(audit.root/'reports/final_results_audit.json')}))


if __name__=='__main__': main()
