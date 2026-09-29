#!/usr/bin/env python3
"""One-time test evaluation of a frozen checkpoint. Refuses to run twice for the same run."""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from common import DATASETS, ROOT, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['boards', 'ports'])
    parser.add_argument('run', help='run folder name under runs/')
    args = parser.parse_args()
    run = ROOT / 'runs' / args.run
    out = run / 'test-evaluation.json'
    if out.exists():
        raise SystemExit(f'{out} exists: test was already evaluated once for this run')
    summary = json.loads((run / 'training-summary.json').read_text(encoding='utf-8'))
    best = Path(summary['best'])
    if sha256(best) != summary['best_sha256']:
        raise SystemExit('best.pt changed after training summary was written')
    data = DATASETS / f'{args.model}_v1' / 'data.yaml'
    metrics = YOLO(str(best)).val(data=str(data), split='test', imgsz=summary['config']['imgsz'],
                                  batch=1, conf=0.001, iou=0.7, plots=True,
                                  project=str(run), name='test', exist_ok=False)
    names = metrics.names
    write_json(out, {
        'checkpoint_sha256': summary['best_sha256'], 'split': 'test',
        'overall': metrics.results_dict,
        'per_class_map50_95': {names[int(c)]: float(m) for c, m in zip(metrics.box.ap_class_index, metrics.box.maps[metrics.box.ap_class_index])},
        'scope': 'held-out groups of the public sources; not D455 real-camera accuracy',
    })
    print(out.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
