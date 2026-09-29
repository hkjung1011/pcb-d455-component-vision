#!/usr/bin/env python3
"""Train one model (boards or ports). Not run during setup — see NEXT_STEPS_PROMPT.md.

Selection uses val only (Ultralytics best.pt = best val fitness). test is evaluated once,
after the checkpoint is frozen, with evaluate_test.py.
"""
import argparse
import json

import torch
from ultralytics import YOLO

from common import DATASETS, ROOT, sha256, write_json
from verify_datasets import dataset_fingerprint, holdout_inventory

CONFIGS = {
    # IoTKITs images are 640x640; boards are large objects
    'boards': dict(imgsz=640, batch=8, epochs=80, patience=20, degrees=10, fliplr=0.5,
                   mosaic=1.0, close_mosaic=10, scale=0.5, translate=0.1),
    # ports are small; keep the D455 prototype resolution, no flips (left/right layout matters)
    'ports': dict(imgsz=960, batch=4, epochs=100, patience=25, degrees=10, fliplr=0.0,
                  mosaic=0.5, close_mosaic=10, scale=0.3, translate=0.1),
}
COMMON = dict(optimizer='AdamW', lr0=0.001, lrf=0.05, cos_lr=True, weight_decay=0.0005,
              warmup_epochs=3, amp=False, seed=20260929, deterministic=True, workers=2,
              hsv_h=0.015, hsv_s=0.4, hsv_v=0.3, mixup=0.0, plots=True, save=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=sorted(CONFIGS))
    parser.add_argument('--weights', default=str(ROOT.parent / 'r03' / 'models' / 'yolo11s.pt'),
                        help='official COCO checkpoint to start from')
    parser.add_argument('--name', default=None)
    args = parser.parse_args()
    data = DATASETS / f'{args.model}_v1' / 'data.yaml'
    if not data.exists():
        raise SystemExit(f'{data} missing; run assemble.py and verify_datasets.py first')
    verification = json.loads((DATASETS / 'verification.json').read_text(encoding='utf-8'))
    verified = verification.get(f'{args.model}_v1', {})
    if verified.get('schema') != 'boards_ports_verification_v2' or not verified.get('training_ready'):
        raise SystemExit('dataset is not training_ready; resolve missing classes/annotations and verify first')
    if (verified.get('dataset_fingerprint') != dataset_fingerprint(data.parent)
            or verified.get('class_map_sha256') != sha256(ROOT / 'class_map.json')
            or verified.get('holdout_inventory') != holdout_inventory()):
        raise SystemExit('dataset/mapping/holdout changed since verification; rebuild or verify again')
    torch.set_num_threads(4)
    name = args.name or f'{args.model}_v1_yolo11s'
    model = YOLO(args.weights)
    result = model.train(data=str(data), project=str(ROOT / 'runs'), name=name, device=0,
                         **COMMON, **CONFIGS[args.model])
    best = model.trainer.best
    write_json(ROOT / 'runs' / name / 'training-summary.json', {
        'model': args.model, 'weights_init': args.weights, 'best': str(best), 'best_sha256': sha256(best),
        'val_metrics': result.results_dict, 'data_yaml_sha256': sha256(data),
        'config': {**COMMON, **CONFIGS[args.model]},
        'scope': 'val metrics only; test not yet evaluated',
    })


if __name__ == '__main__':
    main()
