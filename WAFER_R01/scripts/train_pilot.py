"""Local ten-epoch microscopy wafer bbox pilot; requires a separately audited split."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import hashlib
import json
import math
import os

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'runtime_config'))
os.environ.setdefault('TQDM_DISABLE', '1')
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model


def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


def parameter_hash(model):
    h = hashlib.sha256()
    for name, p in unwrap_model(model).named_parameters():
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class EvidenceTrainer(DetectionTrainer):
    evidence = ROOT / 'reports'

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        if mode=='train' and batch_size!=4:
            raise RuntimeError('Unexpected batch-size change; preserve this run for review')
        return super().get_dataloader(dataset_path,batch_size,rank,mode)

    def preprocess_batch(self, batch):
        if hasattr(self, 'actual_batches'):
            self.actual_batches += 1
            self.actual_images.update(str(Path(p).resolve()) for p in batch['im_file'])
            self.transformed_class_exposures.update(str(int(c)) for c in batch['cls'].flatten().tolist())
        return super().preprocess_batch(batch)

    def optimizer_step(self):
        if any(not torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None):
            raise RuntimeError('Nonfinite gradient: training stopped without silent recovery')
        super().optimizer_step()

    def _handle_nan_recovery(self, epoch):
        if not math.isfinite(float(self.fitness)):
            raise RuntimeError('Nonfinite validation fitness')
        return False

    def final_eval(self):
        self.in_final_eval = True
        try: super().final_eval()
        finally: self.in_final_eval = False


def on_start(t):
    t.direct_calls = 0; t.actual_batches = 0; t.epoch_records = []
    t.actual_images = Counter(); t.transformed_class_exposures = Counter(); t.in_final_eval = False
    def step_hook(optimizer, args, kwargs): t.direct_calls += 1
    t.count_handle = t.optimizer.register_step_post_hook(step_hook)
    t.initial_hash = parameter_hash(t.model)
    torch.cuda.reset_peak_memory_stats()
    save(t.evidence / 'training_start.json', {
        'started_utc': now(), 'initial_parameter_sha256': t.initial_hash,
        'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'parameters': sum(p.numel() for p in t.model.parameters()),
        'actual_batch': t.batch_size, 'train_batches_per_epoch': len(t.train_loader),
        'names': t.data['names'], 'script_sha256': sha(__file__),
        'class_weight_tensor': getattr(unwrap_model(t.model), 'class_weights', None).tolist()
            if torch.is_tensor(getattr(unwrap_model(t.model), 'class_weights', None)) else None,
        'class_loss_policy': 'No added per-class multiplier; stock classification loss',
    })
    print('WAFER_TRAIN_STARTED 10 epochs FP32 batch4', flush=True)


def on_batch(t):
    losses = t.tloss.values() if isinstance(t.tloss, dict) else [t.tloss]
    if any(not torch.isfinite(v).all() for v in losses):
        raise RuntimeError('Nonfinite training loss')


def on_epoch(t):
    if t.in_final_eval: return
    row = {'epoch': t.epoch + 1, 'utc': now(), 'optimizer_calls': t.direct_calls,
           'batches_cumulative': t.actual_batches, 'ema_updates': t.ema.updates,
           'validation': {k: float(v) for k, v in t.metrics.items()}}
    t.epoch_records.append(row)
    save(t.evidence / 'epochs.json', t.epoch_records)
    print(f"WAFER_EPOCH {row['epoch']}/10 calls={t.direct_calls} map={row['validation'].get('metrics/mAP50-95(B)')}", flush=True)


def on_end(t):
    steps = [int(v['step'].item()) for v in t.optimizer.state.values() if 'step' in v]
    final_hash = parameter_hash(t.model)
    if t.direct_calls <= 0 or final_hash == t.initial_hash:
        raise RuntimeError('No verified parameter updates')
    save(t.evidence / 'training_summary.json', {
        'status': 'TRAINING_COMPLETE', 'finished_utc': now(), 'epochs_completed': len(t.epoch_records),
        'direct_optimizer_calls': t.direct_calls, 'batches': t.actual_batches,
        'image_exposures': sum(t.actual_images.values()), 'unique_training_images_seen': len(t.actual_images),
        'transformed_class_exposures': dict(t.transformed_class_exposures),
        'initial_parameter_sha256': t.initial_hash, 'final_parameter_sha256': final_hash,
        'parameter_step_min': min(steps), 'parameter_step_max': max(steps), 'ema_updates': t.ema.updates,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'best_checkpoint': str(t.best), 'best_checkpoint_sha256': sha(t.best),
        'class_weight_note': 'No extra class weighting; source label counts and actual transformed exposures are distinct',
        'd455_verified': False, 'instance_masks_trained': False,
    })
    save(t.evidence / 'image_exposures.json', dict(t.actual_images))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--audit', required=True)
    args = p.parse_args()
    data = Path(args.data).resolve(); audit_path = Path(args.audit).resolve()
    audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
    if audit.get('status') != 'PASS_DATA_AUDIT': raise RuntimeError('Audited split required')
    if audit.get('data_yaml_sha256') != sha(data): raise RuntimeError('Data YAML differs from audit')
    for row in audit['files']:
        if sha(row['path']) != row['sha256']: raise RuntimeError(f"Dataset file changed: {row['path']}")
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required for this budget')
    if (ROOT / 'runs/yolo11n_10ep').exists(): raise RuntimeError('Preserve existing run; do not retrain or overwrite')
    initial = BASE / 'work/r03/models/yolo11n.pt'
    config = dict(data=str(data), epochs=10, batch=4, imgsz=1024, nbs=8, device=0, workers=0,
                  optimizer='AdamW', lr0=.001, lrf=.01, warmup_epochs=1, warmup_bias_lr=0.,
                  amp=False, seed=42, deterministic=True, patience=20, cache=False,
                  mosaic=0., mixup=0., copy_paste=0., degrees=0., translate=0., scale=0., shear=0., perspective=0.,
                  fliplr=.5, flipud=.5, hsv_h=0., hsv_s=0., hsv_v=.15,
                  box=7.5, cls=.5, dfl=1.5, cls_pw=0., conf=.001, iou=.7, max_det=300,
                  val=True, plots=True, save=True, save_period=5,
                  project=str(ROOT / 'runs'), name='yolo11n_10ep', exist_ok=False)
    save(ROOT / 'protocol.json', {
        'created_utc': now(), 'task': 'microscopy wafer surface defect bounding box detection',
        'initial_weights': str(initial), 'initial_weights_sha256': sha(initial), 'configuration': config,
        'dataset_audit_sha256': sha(audit_path), 'selection': 'Highest validation mAP50-95 (stock YOLO detection fitness)',
        'test': 'Selected best checkpoint evaluated once after SHA freeze; test AP used only for reporting',
        'evaluation': {'confidence_floor': .001, 'nms_iou': .7, 'max_detections_per_image': 300},
        'scope': 'Public microscopy wafer image pilot; paper describes electron microscopy with attached camera; physical wafer/lot independence unknown; not D455 RGB training or field validation',
        'license': 'Author-published research data; repository has no explicit dataset license. No source-data redistribution.',
        'new_human_labels': 0, 'instance_masks_trained': False,
    })
    model = YOLO(str(initial))
    model.add_callback('on_pretrain_routine_end', on_start)
    model.add_callback('on_train_batch_end', on_batch)
    model.add_callback('on_fit_epoch_end', on_epoch)
    model.add_callback('on_train_end', on_end)
    model.train(trainer=EvidenceTrainer, **config)
    summary = json.loads((ROOT / 'reports/training_summary.json').read_text(encoding='utf-8'))
    best = Path(summary['best_checkpoint'])
    save(ROOT / 'reports/selection_frozen.json', {
        'frozen_utc': now(), 'checkpoint': str(best), 'sha256': sha(best),
        'selected_on': 'validation only', 'test_results_seen': False,
        'protocol_sha256': sha(ROOT / 'protocol.json'), 'data_audit_sha256': sha(audit_path),
    })
    test_dir = ROOT / 'evaluation/test'
    if test_dir.exists(): raise RuntimeError('Existing test output: preserve it and inspect before retry')
    save(ROOT / 'reports/test_started.json', {'utc': now(), 'selection_sha256': sha(ROOT / 'reports/selection_frozen.json')})
    evaluator = YOLO(str(best))
    metrics = evaluator.val(data=str(data), split='test', imgsz=1024, batch=4, device=0,
                            workers=0, half=False, conf=.001, iou=.7, max_det=300,
                            save_json=True, plots=True, project=str(ROOT / 'evaluation'), name='test', exist_ok=False)
    classes = [{'class_id': int(cid), 'name': evaluator.names[int(cid)],
                'ap50': float(metrics.box.ap50[i]), 'ap50_95': float(metrics.box.ap[i])}
               for i, cid in enumerate(metrics.box.ap_class_index)]
    save(ROOT / 'reports/test_metrics.json', {
        'status': 'PUBLIC_IMAGE_PILOT_EVALUATED_NOT_D455_VALIDATED', 'finished_utc': now(),
        'checkpoint_sha256': sha(best), 'selection_sha256': sha(ROOT / 'reports/selection_frozen.json'),
        'bbox_map50': float(metrics.box.map50), 'bbox_map50_95': float(metrics.box.map),
        'per_class': classes, 'speed': metrics.speed,
        'metric_definition': 'Ultralytics 8.4.120 bbox AP, conf floor0.001, NMS IoU0.7, global max_det300; no test threshold tuning',
        'operating_precision_recall': 'Not reported: default library P/R optimize the split F1 and are not a frozen operating point',
        'd455_verified': False, 'instance_masks_trained': False,
    })
    print('WAFER_PILOT_COMPLETE', flush=True)


if __name__ == '__main__': main()
