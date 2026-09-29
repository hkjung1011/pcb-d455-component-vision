#!/usr/bin/env python3
"""Fine-tune the existing YOLO11s checkpoint for the inspected board parts."""
from pathlib import Path
import json
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    model = YOLO(str('yolo11s.pt'))
    result = model.train(
        data=str(ROOT / 'dataset-prototype/data.yaml'),
        project=str(ROOT / 'runs'), name='board-parts-prototype',
        epochs=50, imgsz=960, batch=4, device=0, workers=2,
        optimizer='AdamW', lr0=.001, lrf=.05, weight_decay=.0005,
        freeze=10, patience=15, amp=False, seed=20260929,
        deterministic=True, mosaic=.25, close_mosaic=8,
        degrees=0, translate=.05, scale=.15, fliplr=0, flipud=0,
        hsv_h=.01, hsv_s=.2, hsv_v=.2, mixup=0,
        plots=True, save=True, verbose=False,
    )
    summary = {
        'weights': str(model.trainer.best),
        'metrics': result.results_dict,
        'evaluation_scope': 'Synthetic development set derived from one source scene. Not independent real-world accuracy.',
    }
    (ROOT / 'training-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
