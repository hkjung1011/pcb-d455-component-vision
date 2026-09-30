"""Balanced Jetson training phase; frozen validation gate and conditional test."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import math
import os
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'boards1000_s650_v150_t200_e50_20260929'
TRIAL = HERE / 'boards1000_balanced_jetson_20260930'
RUN = TRIAL / 'runs/boards1000_balanced_jetson30'
STATE = TRIAL / 'execution_state.json'
BASE = SOURCE / 'runs/boards1000_yolo11s_50ep/weights/best.pt'
CURRENT = HERE / 'boards1000_finetune20_20260930/runs/boards1000_finetune20/weights/best.pt'
CURRENT_SHA = 'fa0b8fb5fc11de748e9d8051b93fc57dde5c6cc34db6167045e3fee9e4912752'
os.environ['YOLO_CONFIG_DIR'] = str(TRIAL / 'ultralytics_config')
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['COMET_MODE'] = 'DISABLED'
sys.path.insert(0, str(TRIAL))
from common import DATASETS, MODEL_CLASSES, ROOT, sha256
from verify_datasets import dataset_fingerprint, holdout_inventory
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

SETTINGS.update({k: False for k in ['sync', 'clearml', 'comet', 'dvc', 'mlflow', 'neptune', 'raytune', 'tensorboard', 'wandb'] if k in SETTINGS})
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
utc = lambda: datetime.now(timezone.utc).isoformat()
config = {**read(SOURCE / 'runs/boards1000_yolo11s_50ep/training-summary.json')['config'],
          'epochs': 30, 'lr0': 0.0003, 'lrf': 0.1, 'warmup_epochs': 3.0,
          'warmup_bias_lr': 0.0, 'mosaic': 0.5, 'close_mosaic': 10, 'patience': 10,
          'resume': False}
protocol = dict(imgsz=640, batch=8, device=0, conf=0.001, iou=0.7,
                max_det=300, half=False, plots=True, verbose=False)
policy = {'minimum_overall_ap50_95_gain': 0.005, 'maximum_per_class_ap50_95_drop': 0.05,
          'minimum_jetson_ap50_95_gain': 0.0,
          'minimum_overall_gain_vs_current_selected': 0.0,
          'selection_split': 'val', 'test_role': 'development test; reused from previous experiments',
          'test_policy': 'One candidate test invocation only if the validation promotion gate passes.',
          'interpretation': 'Fixed relative to ORIGINAL 50-epoch best. Pragmatic gate on a reused small validation set, not statistical significance. Training data and schedule changed; not a data-only or epochs-only ablation.'}
EXPECTED_FINGERPRINT = '3120af5994f7494f608b31b59b48833eebee8423408f5a0190f516f6e708ba6f'
EXPECTED_MAP_SHA = '7c6ed00977a900b18ed6efa95dab86c853680fdf716f434d1d3ad1c348a34a6c'
BASELINE_JSON = HERE / 'boards1000_finetune20_20260930/baseline_validation.json'
CURRENT_VALIDATION_JSON = HERE / 'boards1000_finetune20_20260930/candidate_validation.json'


def evaluation_inventory(dataset):
    index = read(dataset / 'dataset_index.json')
    records = sorted((r for r in index['records'] if r['split'] in {'val', 'test'}), key=lambda r: (r['split'], r['id']))
    assert len(records) == 350
    manifest = []
    for record in records:
        assert sha256(dataset / record['image']) == record['image_sha256']
        assert sha256(dataset / record['label']) == record['label_sha256']
        manifest.append({k: record[k] for k in ['id', 'split', 'image', 'label', 'image_sha256', 'label_sha256', 'width', 'height']})
    return {'classes': index['classes'], 'records': manifest}


def integrity(verified, expected_eval=None):
    assert verified['dataset_fingerprint'] == EXPECTED_FINGERPRINT == dataset_fingerprint(DATASETS / 'boards_v1')
    assert verified['class_map_sha256'] == EXPECTED_MAP_SHA == sha256(ROOT / 'class_map.json')
    assert verified['holdout_inventory'] == holdout_inventory()
    assert sha256(BASE) == '25a6253ce7520d50d362fdb276e61d25b2f77ec78583d8706cd463f74282b766'
    assert sha256(CURRENT) == CURRENT_SHA
    inventory = evaluation_inventory(DATASETS / 'boards_v1')
    assert inventory == evaluation_inventory(SOURCE / 'datasets/boards_v1'), 'Validation/test image and label bytes must stay identical'
    if expected_eval is not None:
        assert inventory == expected_eval
    return inventory


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def metrics_data(metrics, split, verified):
    indices = {int(cid): i for i, cid in enumerate(metrics.box.ap_class_index)}
    rows = []
    for cid, name in enumerate(MODEL_CLASSES['boards']):
        i = indices[cid]
        rows.append({'class_id': cid, 'class_name': name,
                     'ground_truth_boxes': verified['boxes'][split][name],
                     'map50': float(metrics.box.ap50[i]), 'map50_95': float(metrics.box.ap[i])})
    return {'overall': {k: float(v) for k, v in metrics.results_dict.items()}, 'per_class': rows}


def main():
    if STATE.exists() or RUN.exists() or (TRIAL / 'experiment_plan.json').exists():
        raise SystemExit('Existing balanced experiment preserved; do not silently rerun.')
    assert TRIAL.is_dir(), 'Verified data folder must already exist'
    verified = read(DATASETS / 'verification.json')['boards_v1']
    assert verified['training_ready'] and verified['structure_pass']
    assert verified['images'] == {'train': 650, 'val': 150, 'test': 200}
    inventory = integrity(verified)
    baseline = read(BASELINE_JSON)
    current_validation = read(CURRENT_VALIDATION_JSON)
    assert baseline['checkpoint_sha256'] == sha256(BASE)
    assert baseline['protocol'] == protocol
    assert current_validation['protocol'] == protocol
    assert current_validation['checkpoint_sha256'] == CURRENT_SHA
    assert read(HERE / 'boards1000_finetune20_20260930/selection.json')['selected_checkpoint_sha256'] == CURRENT_SHA
    assert [r['class_name'] for r in baseline['per_class']] == MODEL_CLASSES['boards']
    assert all(r['ground_truth_boxes'] == verified['boxes']['val'][r['class_name']] for r in baseline['per_class'])
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    plan = {'created_at': utc(), 'config': config, 'validation_protocol': protocol, 'promotion_policy': policy,
            'dataset': str(DATASETS / 'boards_v1'), 'dataset_fingerprint': verified['dataset_fingerprint'],
            'initial_checkpoint': str(BASE), 'initial_checkpoint_sha256': sha256(BASE),
            'script_sha256': sha256(Path(__file__)), 'classes': MODEL_CLASSES['boards'],
            'baseline_validation_source': str(BASELINE_JSON), 'baseline_validation_source_sha256': sha256(BASELINE_JSON),
            'current_selected_checkpoint': str(CURRENT), 'current_selected_checkpoint_sha256': CURRENT_SHA,
            'current_selected_validation_source': str(CURRENT_VALIDATION_JSON),
            'current_selected_validation_source_sha256': sha256(CURRENT_VALIDATION_JSON),
            'current_selected_val_ap50_95': current_validation['overall']['metrics/mAP50-95(B)'],
            'baseline_reinference': False, 'evaluation_inventory': inventory,
            'data_change': 'Train-only replacement of 40 redundant TX2 photos with 40 reviewed Nano photos; retain 27 TX2+40 Nano train images; exact650/150/200 unchanged. Validation/test IDs and image/label byte hashes unchanged.',
            'known_data_limits': 'Validation Jetson has only2 source groups (14TX2+2Nano images). Candidate pHash screening and manual label review do not prove physical specimen/capture independence.'}
    write(TRIAL / 'experiment_plan.json', plan)
    state = {'status': 'training', 'started_at': utc(), 'optimizer_steps': 0,
             'epochs': [], 'test_invocations': 0, 'config': config,
             'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
             'ultralytics': ultralytics.__version__, 'dataset_fingerprint': verified['dataset_fingerprint']}
    write(STATE, state)
    shutil.copyfile(BASELINE_JSON, TRIAL / 'baseline_validation.json')
    shutil.copyfile(CURRENT_VALIDATION_JSON, TRIAL / 'current_selected_validation.json')
    assert sha256(TRIAL / 'baseline_validation.json') == plan['baseline_validation_source_sha256']
    started = time.monotonic()
    model = YOLO(str(BASE))
    hook = None

    def count_step(optimizer, args, kwargs): state['optimizer_steps'] += 1
    def train_start(trainer):
        nonlocal hook
        hook = trainer.optimizer.register_step_post_hook(count_step)
    def losses(trainer):
        values = trainer.tloss.values() if isinstance(trainer.tloss, dict) else trainer.tloss
        return [float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for v in values]
    def check_batch(trainer):
        if trainer.tloss is not None and not all(math.isfinite(x) for x in losses(trainer)):
            raise RuntimeError('Nonfinite training loss')
    def check_weights(trainer):
        assert all(bool(torch.isfinite(t).all()) for t in trainer.model.state_dict().values() if t.is_floating_point()), 'Nonfinite model parameters'
    def epoch_end(trainer):
        epoch = trainer.epoch + 1
        path = Path(trainer.save_dir) / 'results.csv'
        logged = {int(r['epoch']) for r in csv.DictReader(path.open(encoding='utf-8-sig'))} if path.exists() else set()
        if epoch not in logged or any(r['epoch'] == epoch for r in state['epochs']): return
        row = {'epoch': epoch, 'optimizer_steps': state['optimizer_steps'], 'elapsed_seconds': time.monotonic()-started,
               'losses': losses(trainer), 'fitness': float(trainer.fitness),
               'val_metrics': {k: float(v) for k, v in trainer.metrics.items()}}
        state['epochs'].append(row)
        if epoch == 3: state['three_epoch_check'] = {'finite_losses_and_parameters': True, 'optimizer_steps': state['optimizer_steps']}
        write(STATE, state)
        print('BALANCED_EPOCH ' + json.dumps(row), flush=True)

    model.add_callback('on_train_start', train_start)
    model.add_callback('on_train_batch_end', check_batch)
    model.add_callback('on_train_epoch_end', check_weights)
    model.add_callback('on_fit_epoch_end', epoch_end)
    result = model.train(data=str(DATASETS / 'boards_v1/data.yaml'), project=str(TRIAL / 'runs'),
                         name=RUN.name, exist_ok=False, **config)
    if hook is not None: hook.remove()
    seconds = time.monotonic()-started
    best = Path(model.trainer.best)
    assert best.is_file() and model.names == dict(enumerate(MODEL_CLASSES['boards']))
    rows = list(csv.DictReader((RUN / 'results.csv').open(encoding='utf-8-sig')))
    assert len(rows) == len(state['epochs']) and 0 < len(rows) <= 30
    assert [int(r['epoch']) for r in rows] == list(range(1, len(rows)+1))
    summary = {'started_at': state['started_at'], 'finished_at': utc(), 'epochs_completed': len(rows),
               'actual_optimizer_steps': state['optimizer_steps'], 'training_seconds': seconds,
               'best_epoch': max(state['epochs'], key=lambda e: (e['fitness'], e['epoch']))['epoch'],
               'best': str(best), 'best_sha256': sha256(best), 'config': config,
               'initial_checkpoint_sha256': plan['initial_checkpoint_sha256'],
               'training_final_val': metrics_data(result, 'val', verified),
               'peak_cuda_memory_gib': torch.cuda.max_memory_allocated()/2**30}
    write(TRIAL / 'training-summary.json', summary)
    integrity(verified, inventory)
    assert sha256(best) == summary['best_sha256']
    state['status'] = 'candidate_validation'
    write(STATE, state)
    candidate_metrics = YOLO(str(best)).val(data=str(DATASETS / 'boards_v1/data.yaml'), split='val',
                                           project=str(TRIAL), name='candidate_val', exist_ok=False, **protocol)
    candidate = {**metrics_data(candidate_metrics, 'val', verified), 'checkpoint_sha256': sha256(best),
                 'completed_at': utc(), 'protocol': protocol}
    write(TRIAL / 'candidate_validation.json', candidate)
    gain = candidate['overall']['metrics/mAP50-95(B)'] - baseline['overall']['metrics/mAP50-95(B)']
    deltas = {b['class_name']: c['map50_95']-b['map50_95'] for b, c in zip(baseline['per_class'], candidate['per_class'])}
    current_gain = candidate['overall']['metrics/mAP50-95(B)'] - current_validation['overall']['metrics/mAP50-95(B)']
    promoted = (gain >= policy['minimum_overall_ap50_95_gain']
                and min(deltas.values()) >= -policy['maximum_per_class_ap50_95_drop']
                and deltas['jetson'] >= policy['minimum_jetson_ap50_95_gain']
                and current_gain >= policy['minimum_overall_gain_vs_current_selected'])
    selection = {'selection_split': 'val', 'promoted': promoted, 'overall_ap_gain': gain,
                 'per_class_ap_gains': deltas, 'policy': policy,
                 'overall_ap_gain_vs_current_selected': current_gain,
                 'selected_checkpoint': str(best if promoted else CURRENT),
                 'selected_checkpoint_sha256': sha256(best if promoted else CURRENT),
                 'candidate_checkpoint_sha256': sha256(best), 'baseline_checkpoint_sha256': sha256(BASE),
                 'current_selected_checkpoint_sha256': CURRENT_SHA,
                 'initial_checkpoint_sha256': sha256(BASE),
                 'comparison_roles': 'Initial weights and per-class/global gain baseline: original50best. Global no-downgrade and failure fallback: prior fine-tuning selected model. All selection metrics are validation-only.'}
    write(TRIAL / 'selection.json', selection)
    integrity(verified, inventory)
    assert sha256(best) == summary['best_sha256'] == candidate['checkpoint_sha256']
    if promoted:
        state['status'] = 'development_test'
        write(STATE, state)
        marker = TRIAL / 'test-evaluation-attempt.json'
        with marker.open('x', encoding='utf-8') as f:
            json.dump({'started_at': utc(), 'checkpoint_sha256': sha256(best), 'split': 'test'}, f)
        metrics = YOLO(str(best)).val(data=str(DATASETS / 'boards_v1/data.yaml'), split='test',
                                    project=str(TRIAL), name='test', exist_ok=False, **{**protocol, 'batch': 1})
        write(TRIAL / 'test-evaluation.json', {**metrics_data(metrics, 'test', verified),
              'checkpoint_sha256': sha256(best), 'images': 200, 'invocations': 1, 'completed_at': utc(),
              'scope': policy['test_role']})
        state['test_invocations'] = 1
    state.update({'status': 'complete', 'finished_at': utc(), 'promoted': promoted,
                  'selected_checkpoint_sha256': selection['selected_checkpoint_sha256']})
    write(STATE, state)
    print('BALANCED_COMPLETE ' + json.dumps(selection), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if TRIAL.exists():
            write(TRIAL / 'failure.json', {'error': repr(exc), 'failed_at': utc(), 'traceback': traceback.format_exc()})
            if STATE.exists():
                failed_state = read(STATE)
                failed_state.update({'status': 'failed', 'failed_at': utc(), 'error': repr(exc)})
                write(STATE, failed_state)
        raise
