"""Bounded fine-tuning, paired validation selection, conditional development test."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import math
import os
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'boards1000_s650_v150_t200_e50_20260929'
TRIAL = HERE / 'boards1000_finetune20_20260930'
RUN = TRIAL / 'runs/boards1000_finetune20'
STATE = TRIAL / 'execution_state.json'
BASE = SOURCE / 'runs/boards1000_yolo11s_50ep/weights/best.pt'
os.environ['YOLO_CONFIG_DIR'] = str(TRIAL / 'ultralytics_config')
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['COMET_MODE'] = 'DISABLED'
sys.path.insert(0, str(SOURCE))
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
          'epochs': 20, 'lr0': 0.0001, 'lrf': 0.1, 'warmup_epochs': 1.0,
          'warmup_bias_lr': 0.0, 'mosaic': 0.0, 'close_mosaic': 0, 'patience': 8,
          'resume': False}
protocol = dict(imgsz=640, batch=8, device=0, conf=0.001, iou=0.7,
                max_det=300, half=False, plots=True, verbose=False)
policy = {'minimum_overall_ap50_95_gain': 0.005, 'maximum_per_class_ap50_95_drop': 0.05,
          'selection_split': 'val', 'test_role': 'development test; reused from previous experiments',
          'test_policy': 'One candidate test invocation only if the validation promotion gate passes.',
          'interpretation': 'Pragmatic gate on a small reused validation set, not statistical significance. Fresh optimizer and schedule; not an epochs-only ablation.'}


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
    if TRIAL.exists():
        raise SystemExit('Existing fine-tuning experiment preserved; do not silently rerun.')
    verified = read(DATASETS / 'verification.json')['boards_v1']
    assert verified['training_ready'] and verified['structure_pass']
    assert verified['images'] == {'train': 650, 'val': 150, 'test': 200}
    assert verified['dataset_fingerprint'] == dataset_fingerprint(DATASETS / 'boards_v1')
    assert verified['class_map_sha256'] == sha256(ROOT / 'class_map.json')
    assert verified['holdout_inventory'] == holdout_inventory()
    assert sha256(BASE) == '25a6253ce7520d50d362fdb276e61d25b2f77ec78583d8706cd463f74282b766'
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    TRIAL.mkdir()
    plan = {'created_at': utc(), 'config': config, 'validation_protocol': protocol, 'promotion_policy': policy,
            'dataset': str(DATASETS / 'boards_v1'), 'dataset_fingerprint': verified['dataset_fingerprint'],
            'initial_checkpoint': str(BASE), 'initial_checkpoint_sha256': sha256(BASE),
            'script_sha256': sha256(Path(__file__)), 'classes': MODEL_CLASSES['boards'],
            'known_data_limits': 'Jetson training contains 3 TX2 source groups and no Nano group; validation contains only 2 Jetson groups. More epochs cannot supply missing scene/subtype diversity.'}
    write(TRIAL / 'experiment_plan.json', plan)
    state = {'status': 'baseline_validation', 'started_at': utc(), 'optimizer_steps': 0,
             'epochs': [], 'test_invocations': 0, 'config': config,
             'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
             'ultralytics': ultralytics.__version__, 'dataset_fingerprint': verified['dataset_fingerprint']}
    write(STATE, state)
    baseline_model = YOLO(str(BASE))
    assert baseline_model.names == dict(enumerate(MODEL_CLASSES['boards']))
    baseline_metrics = baseline_model.val(data=str(DATASETS / 'boards_v1/data.yaml'), split='val',
                                         project=str(TRIAL), name='baseline_val', exist_ok=False, **protocol)
    baseline = {**metrics_data(baseline_metrics, 'val', verified), 'checkpoint_sha256': sha256(BASE),
                'completed_at': utc(), 'protocol': protocol}
    write(TRIAL / 'baseline_validation.json', baseline)
    del baseline_model, baseline_metrics
    torch.cuda.empty_cache()
    state['status'] = 'training'
    write(STATE, state)
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
        print('FINETUNE_EPOCH ' + json.dumps(row), flush=True)

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
    assert len(rows) == len(state['epochs']) and 0 < len(rows) <= 20
    summary = {'started_at': state['started_at'], 'finished_at': utc(), 'epochs_completed': len(rows),
               'actual_optimizer_steps': state['optimizer_steps'], 'training_seconds': seconds,
               'best_epoch': max(state['epochs'], key=lambda e: (e['fitness'], e['epoch']))['epoch'],
               'best': str(best), 'best_sha256': sha256(best), 'config': config,
               'initial_checkpoint_sha256': plan['initial_checkpoint_sha256'],
               'training_final_val': metrics_data(result, 'val', verified),
               'peak_cuda_memory_gib': torch.cuda.max_memory_allocated()/2**30}
    write(TRIAL / 'training-summary.json', summary)
    state['status'] = 'candidate_validation'
    write(STATE, state)
    candidate_metrics = YOLO(str(best)).val(data=str(DATASETS / 'boards_v1/data.yaml'), split='val',
                                           project=str(TRIAL), name='candidate_val', exist_ok=False, **protocol)
    candidate = {**metrics_data(candidate_metrics, 'val', verified), 'checkpoint_sha256': sha256(best),
                 'completed_at': utc(), 'protocol': protocol}
    write(TRIAL / 'candidate_validation.json', candidate)
    gain = candidate['overall']['metrics/mAP50-95(B)'] - baseline['overall']['metrics/mAP50-95(B)']
    deltas = {b['class_name']: c['map50_95']-b['map50_95'] for b, c in zip(baseline['per_class'], candidate['per_class'])}
    promoted = gain >= policy['minimum_overall_ap50_95_gain'] and min(deltas.values()) >= -policy['maximum_per_class_ap50_95_drop']
    selection = {'selection_split': 'val', 'promoted': promoted, 'overall_ap_gain': gain,
                 'per_class_ap_gains': deltas, 'policy': policy,
                 'selected_checkpoint': str(best if promoted else BASE),
                 'selected_checkpoint_sha256': sha256(best if promoted else BASE),
                 'candidate_checkpoint_sha256': sha256(best), 'baseline_checkpoint_sha256': sha256(BASE)}
    write(TRIAL / 'selection.json', selection)
    assert sha256(BASE) == plan['initial_checkpoint_sha256']
    assert verified['dataset_fingerprint'] == dataset_fingerprint(DATASETS / 'boards_v1')
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
    print('FINETUNE_COMPLETE ' + json.dumps(selection), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if TRIAL.exists():
            write(TRIAL / 'failure.json', {'error': repr(exc), 'failed_at': utc(), 'traceback': traceback.format_exc()})
        raise
