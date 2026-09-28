"""Matched R04 training with directly counted optimizer calls and sampled exposure logs."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'runtime_config'))
os.environ.setdefault('TQDM_DISABLE', '1')
import torch
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO, __version__
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.utils.torch_utils import unwrap_model
from ignore_adapter import IgnoreDetectionTrainer


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def progress(message):
    line = f'[{utc()}] {message}'
    with (ROOT.parent / 'r04_review' / 'progress.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)


def parameter_steps(optimizer):
    values = [int(state['step'].item() if torch.is_tensor(state['step']) else state['step'])
              for state in optimizer.state.values() if 'step' in state]
    return {'min': min(values) if values else 0, 'max': max(values) if values else 0,
            'state_tensors': len(values), 'meaning': 'Per-parameter AdamW counters, separate from directly observed optimizer calls'}


def parameter_sha(model):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        digest.update(name.encode())
        digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class MatchedTrainer(IgnoreDetectionTrainer):
    protocol = None
    arm = None
    evidence_dir = None

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        if mode != 'train':
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if batch_size != self.protocol['batch']:
            raise RuntimeError('R04 forbids silent OOM batch-size changes; re-plan both arms explicitly')
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        views = json.loads(Path(self.data['r04_views_manifest']).read_text(encoding='utf-8'))['records']
        self.views_by_path = {str(Path(v['image']).resolve()): v for v in views}
        if set(map(str, map(Path, dataset.im_files))) != set(self.views_by_path):
            raise RuntimeError('Sampler view inventory differs from training dataset')
        weights = [self.views_by_path[str(Path(p).resolve())]['sampling_weight'] for p in dataset.im_files]
        generator = torch.Generator().manual_seed(42)
        sampler = WeightedRandomSampler(weights, self.protocol['epoch_replacement_draws'], replacement=True, generator=generator)
        self.sampled_groups, self.sampled_views, self.sampled_classes = Counter(), Counter(), Counter()
        self.sampled_ignore, self.sampled_kinds = Counter(), Counter()
        self.sampled_source_ids, self.sampled_complete_ids = set(), set()
        self.actual_batches = 0
        self.epoch_batches = Counter()
        return InfiniteDataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                                  num_workers=0, pin_memory=True, collate_fn=dataset.collate_fn,
                                  generator=torch.Generator().manual_seed(12345), drop_last=False)

    def preprocess_batch(self, batch):
        for path in batch['im_file']:
            view = self.views_by_path[str(Path(path).resolve())]
            self.sampled_groups[view['group_id']] += 1
            self.sampled_views[view['id']] += 1
            self.sampled_kinds[view['kind']] += 1
            for target in view['targets']:
                self.sampled_classes[str(target['class_id'])] += 1
                self.sampled_source_ids.add(target['instance_id'])
                if target['visible_fraction'] >= 1-1e-8:
                    self.sampled_complete_ids.add(target['instance_id'])
            for region in view['ignore_regions']:
                self.sampled_ignore[region['reason']] += 1
        self.actual_batches += 1
        self.epoch_batches[self.epoch+1] += 1
        return super().preprocess_batch(batch)

    def optimizer_step(self):
        if self.optimizer is not self.observed_optimizer:
            raise RuntimeError('Optimizer replaced after counting hook was installed')
        if self.direct_calls >= self.protocol['max_direct_optimizer_calls_per_arm']:
            raise RuntimeError('Optimizer budget would be exceeded')
        if any(not torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None):
            raise RuntimeError('Nonfinite gradients; no silent recovery is allowed in this matched experiment')
        super().optimizer_step()

    def _handle_nan_recovery(self, epoch):
        if not math.isfinite(float(self.fitness)):
            raise RuntimeError('Nonfinite validation fitness')
        return False

    def final_eval(self):
        # Upstream best is full-image loader best, not deployment-selected best.
        # Keep its bookkeeping but never count its extra callback as an epoch.
        self.in_final_evaluation = True
        try:
            super().final_eval()
        finally:
            self.in_final_evaluation = False


def on_start(t):
    t.direct_calls = 0
    t.candidates, t.epoch_evidence = [], []
    t.in_final_evaluation = False
    t.observed_optimizer = t.optimizer
    def observed_step(optimizer, args, kwargs):
        t.direct_calls += 1
    t.counter_hook = t.optimizer.register_step_post_hook(observed_step)
    t.initial_parameter_sha = parameter_sha(unwrap_model(t.model))
    torch.cuda.reset_peak_memory_stats()
    progress(f'TRAIN {t.arm} started: 20 epochs maximum; 1165 optimizer calls maximum; batch=4 FP32; 448 weighted draws/epoch')
    save(t.evidence_dir / 'start.json', {
        'utc': utc(), 'arm': t.arm, 'protocol_sha256': sha(ROOT/'protocol.json'),
        'training_script_sha256': sha(__file__), 'ignore_adapter_sha256': sha(Path(__file__).with_name('ignore_adapter.py')),
        'initial_model_parameter_sha256': t.initial_parameter_sha,
        'initial_checkpoint_sha256': sha(t.protocol['initial_weights']),
        'torch': torch.__version__, 'ultralytics': __version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'actual_batch': t.batch_size,
        'loader_batches_per_epoch': len(t.train_loader), 'parameters': sum(p.numel() for p in t.model.parameters()),
        'class_weights': getattr(unwrap_model(t.model), 'class_weights', None).tolist() if torch.is_tensor(getattr(unwrap_model(t.model), 'class_weights', None)) else None,
        'names': t.data['names'], 'data_yaml_sha256': sha(t.args.data),
        'sidecar_sha256': t.train_loader.dataset.ignore_sidecar_sha256,
    })


def on_batch(t):
    if any(not torch.isfinite(v).all() for v in t.tloss.values()):
        raise RuntimeError('Nonfinite training loss')
    if t.direct_calls >= t.protocol['max_direct_optimizer_calls_per_arm']:
        t.stop = True


def on_epoch(t):
    if t.in_final_evaluation:
        return
    epoch = t.epoch + 1
    if any(x['epoch'] == epoch for x in t.epoch_evidence):
        raise RuntimeError('Duplicate actual epoch callback')
    row = {
        'utc': utc(), 'epoch': epoch, 'optimizer_calls': t.direct_calls,
        'parameter_steps': parameter_steps(t.optimizer), 'ema_updates': t.ema.updates,
        'processed_batches': t.epoch_batches[epoch], 'planned_batches': len(t.train_loader),
        'partial_epoch': t.epoch_batches[epoch] != len(t.train_loader),
        'train_loss': {k: float(v) for k, v in t.tloss.items()},
        'full_image_loader_validation': {k: float(v) for k, v in t.metrics.items()},
        'validation_scope': 'Diagnostic only; native/dual validation selects deployment checkpoint after training',
        'lr': t.lr, 'elapsed_seconds': time.time()-t.train_time_start,
    }
    t.epoch_evidence.append(row)
    save(t.evidence_dir/'epochs.json', t.epoch_evidence)
    if epoch in t.protocol['save_epoch_candidates']:
        model = deepcopy(unwrap_model(t.ema.ema)).half()
        if hasattr(model, 'criterion'):
            model.criterion = None
        path = t.evidence_dir/'candidates'/f'epoch_{epoch:02d}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model': model, 'ema': None, 'optimizer': None, 'epoch': epoch-1,
                    'updates': t.ema.updates, 'train_args': vars(t.args), 'date': utc(), 'version': __version__,
                    'r04': {'arm': t.arm, 'actual_epoch': epoch, 'direct_optimizer_calls': t.direct_calls,
                            'partial_epoch': row['partial_epoch'], 'protocol_sha256': sha(ROOT/'protocol.json')}}, path)
        t.candidates.append({'epoch': epoch, 'optimizer_calls': t.direct_calls, 'path': str(path.resolve()),
                             'sha256': sha(path), 'partial_epoch': row['partial_epoch']})
        save(t.evidence_dir/'candidates.json', t.candidates)
        del model
    save(t.evidence_dir/'sampled_exposures.json', {
        'draws': sum(t.sampled_views.values()), 'groups': dict(t.sampled_groups), 'view_types': dict(t.sampled_kinds),
        'class_exposures': dict(t.sampled_classes), 'ignore_exposures': dict(t.sampled_ignore),
        'unique_positive_source_instances_seen': len(t.sampled_source_ids),
        'unique_positive_source_instances_seen_complete': len(t.sampled_complete_ids),
        'unique_views_drawn': len(t.sampled_views), 'view_draw_counts': dict(t.sampled_views),
        'meaning': 'Actual drawn labels/regions including repeats; not new annotations and not gradient-share measurements',
    })
    progress(f'TRAIN {t.arm} epoch {epoch}/20: optimizer={t.direct_calls}/1165, batches={row["processed_batches"]}/{row["planned_batches"]}, loss={sum(row["train_loss"].values()):.4f}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['baseline', 'improved'], required=True)
    args = p.parse_args()
    protocol = json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    for path, expected in [(protocol['initial_weights'], protocol['initial_weights_sha256']),
                           (ROOT/'data/summary.json', protocol['data_summary_sha256']),
                           (ROOT/'data/native_manifest.json', protocol['native_manifest_sha256'])]:
        if sha(path) != expected:
            raise RuntimeError(f'Frozen input hash mismatch: {path}')
    audit = json.loads((ROOT/'reports/data_independent_audit.json').read_text(encoding='utf-8'))
    if audit.get('status') != 'PASS_DATA_AUDIT' or audit.get('failed_checks'):
        raise RuntimeError('Independent data audit must pass before training')
    if audit['native_manifest_sha256'] != sha(ROOT/'data/native_manifest.json') or audit['protocol_sha256'] != sha(ROOT/'protocol.json'):
        raise RuntimeError('Frozen data changed since the independent audit')
    tests = json.loads((ROOT/'reports/ignore_adapter_tests.json').read_text(encoding='utf-8'))
    if tests['status'] != 'PASS' or tests['cuda_smoke']['status'] != 'PASS':
        raise RuntimeError('Loss adapter and real CUDA batch smoke must pass')
    if tests['source_sha256']['ignore_adapter.py'] != sha(Path(__file__).with_name('ignore_adapter.py')):
        raise RuntimeError('Ignore adapter changed since tests')
    output = ROOT/'runs'/args.arm
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Training output is not empty; preserve it and investigate, do not silently rerun')
    output.mkdir(parents=True, exist_ok=True)
    MatchedTrainer.protocol, MatchedTrainer.arm, MatchedTrainer.evidence_dir = protocol, args.arm, output
    model = YOLO(protocol['initial_weights'])
    for event, callback in [('on_train_start', on_start), ('on_train_batch_end', on_batch), ('on_fit_epoch_end', on_epoch)]:
        model.add_callback(event, callback)
    started = utc()
    try:
        model.train(trainer=MatchedTrainer, data=str(ROOT/'data'/args.arm/'data.yaml'),
                    project=str(ROOT/'training_logs'), name=args.arm, exist_ok=False,
                    epochs=20, imgsz=1024, batch=4, nbs=8, amp=False, device=0, workers=0,
                    optimizer='AdamW', lr0=.001, lrf=.01, warmup_epochs=2, warmup_bias_lr=0.,
                    weight_decay=.0005, seed=42, deterministic=True, cache=False, patience=100,
                    mosaic=0., mixup=0., cutmix=0., copy_paste=0., degrees=0., translate=0.,
                    scale=0., shear=0., perspective=0., multi_scale=0., close_mosaic=0,
                    hsv_h=.005, hsv_s=.15, hsv_v=.2, fliplr=.5, flipud=.5, bgr=0.,
                    box=7.5, cls=.5, dfl=1.5, cls_pw=0., plots=False, verbose=False,
                    save=True, save_period=-1, val=True, max_det=1000, rect=False)
        t = model.trainer
        if t.direct_calls > 1165 or len(t.epoch_evidence) > 20:
            raise RuntimeError('Training budget exceeded')
        final_sha = parameter_sha(unwrap_model(t.model))
        if final_sha == t.initial_parameter_sha:
            raise RuntimeError('No measured parameter changes')
        summary = {'status': 'COMPLETE', 'started_utc': started, 'finished_utc': utc(), 'arm': args.arm,
                   'actual_epochs_reached': t.epoch_evidence[-1]['epoch'],
                   'last_epoch_partial': t.epoch_evidence[-1]['partial_epoch'],
                   'actual_direct_optimizer_calls': t.direct_calls, 'actual_batches': t.actual_batches,
                   'parameter_steps': parameter_steps(t.optimizer), 'ema_updates': t.ema.updates,
                   'initial_parameter_sha256': t.initial_parameter_sha, 'final_parameter_sha256': final_sha,
                   'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                   'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                   'selection_status': 'PENDING_NATIVE_VALIDATION', 'candidate_count': len(t.candidates),
                   'protocol_sha256': sha(ROOT/'protocol.json'), 'class_weights': [1., 1., 1., 1.],
                   'stop_reason': 'direct_optimizer_call_cap' if t.direct_calls == 1165 else 'max_epochs',
                   'candidates_manifest_sha256': sha(output/'candidates.json')}
        save(output/'training_summary.json', summary)
        progress(f'TRAIN {args.arm} COMPLETE: {t.direct_calls} directly observed optimizer calls; {len(t.candidates)} candidates ready for native validation')
    except BaseException as exc:
        save(output/'failure.json', {'utc': utc(), 'error': repr(exc), 'started_utc': started})
        progress(f'TRAIN {args.arm} FAILED: {exc}')
        raise


if __name__ == '__main__':
    main()
