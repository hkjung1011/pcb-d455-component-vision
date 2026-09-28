"""Bounded model training with dataset binding, loss weights and real update accounting."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'runtime_config'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    import yaml
    import torch
    import ultralytics
    from ultralytics import YOLO
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8-sig'))
    out = (ROOT / cfg['run_dir']).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit('New output directory required')
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = (ROOT / cfg['manifest']).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if not manifest.get('training_eligible'):
        raise SystemExit('Dataset has not passed the preparation audit')
    data_path = (ROOT / cfg['data']).resolve()
    data = yaml.safe_load(data_path.read_text(encoding='utf-8-sig'))
    data_root = Path(data['path'])
    if not data_root.is_absolute():
        data_root = (data_path.parent / data_root).resolve()
    model_names = manifest['names']
    yaml_names = data['names']
    ordered = [yaml_names[i] for i in range(len(yaml_names))] if isinstance(yaml_names, dict) else yaml_names
    if ordered != model_names:
        raise SystemExit('Dataset class order mismatch')
    for split in ['train', 'val']:
        spec = data_root / data[split]
        actual = {Path(line).resolve() for line in spec.read_text(encoding='utf-8').splitlines() if line.strip()} if spec.is_file() else {v.resolve() for v in spec.rglob('*') if v.suffix.lower() in ['.jpg','.jpeg','.png']}
        expected = set()
        for row in manifest['records']:
            if row['split'] == split:
                path = Path(row['image'])
                expected.add((path if path.is_absolute() else manifest_path.parent / path).resolve())
        if actual != expected:
            raise SystemExit(f'Actual loader {split} differs from manifest')
    # Preparation records every image SHA and shared origin group. Recheck hash leakage.
    seen, groups = {}, {}
    for row in manifest['records']:
        split = row['split']
        for value, mapping, kind in [(row['sha256'], seen, 'image hash'), (row['group_id'], groups, 'origin group')]:
            if value in mapping and mapping[value] != split:
                raise SystemExit(f'{kind} leakage')
            mapping[value] = split
    weights = (ROOT / cfg['weights']).resolve()
    model = YOLO(str(weights))
    if model.task != cfg['task']:
        raise SystemExit('Wrong model task')
    overrides = cfg['train']
    if not 1 <= int(overrides['epochs']) <= 20:
        raise SystemExit('Per-model epoch budget is 1..20')
    if overrides.get('amp', True):
        raise SystemExit('This protocol uses FP32 after previous skipped-step observations')
    overrides.update(data=str(data_path), project=str(out), name='fit', exist_ok=False, device=0,
                     workers=0, cache=False, plots=True, save=True, save_period=-1)
    events = []
    initial_fingerprint = None

    def started(trainer):
        nonlocal initial_fingerprint
        initial_fingerprint = hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in trainer.model.parameters())).hexdigest()
        layers = []
        for index, module in enumerate(trainer.model.model):
            layers.append({'index':index,'module':str(getattr(module,'type',type(module).__name__)),
                           'from':getattr(module,'f',None),'parameters':sum(v.numel() for v in module.parameters())})
        save(out / 'architecture.json', {'model_id':cfg['model_id'],'task':cfg['task'],'classes':model_names,
             'parameters':sum(v.numel() for v in trainer.model.parameters()),
             'trainable_parameters':sum(v.numel() for v in trainer.model.parameters() if v.requires_grad),
             'layers':layers, 'yaml':trainer.model.yaml})

    def epoch_done(trainer):
        steps = [float(v['step']) for v in trainer.optimizer.state.values() if 'step' in v]
        row = {'epoch':int(trainer.epoch)+1,'optimizer_steps_min':min(steps) if steps else 0,
               'optimizer_steps_max':max(steps) if steps else 0,'learning_rates':[v['lr'] for v in trainer.optimizer.param_groups],
               'metrics':{key:float(value) for key,value in trainer.metrics.items()}}
        events.append(row)
        save(out / 'epoch_updates.json', events)
        print('ACTUAL_UPDATE ' + json.dumps(row), flush=True)

    model.add_callback('on_pretrain_routine_end', started)
    model.add_callback('on_fit_epoch_end', epoch_done)
    environment = {'model_id':cfg['model_id'],'created_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'ultralytics':ultralytics.__version__,'gpu':torch.cuda.get_device_name(0),'configuration':cfg,
        'manifest_sha256':sha(manifest_path),'initial_weights_sha256':sha(weights),
        'scope':cfg['scope'],'class_loss_weights':'Uniform 1.0 for each class; no custom per-class multiplier',
        'source_weighting':manifest['source_weighting'],'label_counts':manifest['counts'],
        'test_used_for_training_or_selection':False}
    save(out / 'environment.json', environment)
    model.train(**overrides)
    actual = max((row['optimizer_steps_min'] for row in events), default=0)
    best = out / 'fit' / 'weights' / 'best.pt'
    final_fingerprint = hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in model.trainer.model.parameters())).hexdigest()
    summary = {'model_id':cfg['model_id'],'status':'TRAINED_OFFLINE_NOT_D455_VALIDATED',
        'requested_epochs':overrides['epochs'],'last_epoch':int(model.trainer.epoch)+1,
        'actual_optimizer_steps':actual,'weights_changed':initial_fingerprint != final_fingerprint,
        'best_path':str(best),'best_sha256':sha(best),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'test_evaluated':False,'scope':cfg['scope'],'camera_verified':False}
    save(out / 'run_summary.json', summary)
    if actual < 1 or not summary['weights_changed']:
        raise SystemExit('No real parameter updates')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
