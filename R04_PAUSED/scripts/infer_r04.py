"""Offline R04 target4 inference with the validation-frozen crop/NMS pipeline."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection', type=Path, default=Path(__file__).resolve().parents[1]/'selected_models.json')
    parser.add_argument('--arm', choices=['recommended', 'baseline', 'improved'], default='recommended')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be a new or empty directory')
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from evaluate_r04 import NAMES, sha, inference_plan, restore_prediction, deduplicate, save, utc
    selected = json.loads(args.selection.read_text(encoding='utf-8-sig'))
    if selected.get('schema') != 'r04-inference-selection-v1':
        raise ValueError('Expected packaged R04 selection registry')
    arm = selected['recommended_arm'] if args.arm == 'recommended' else args.arm
    config = selected['arms'][arm]
    frozen_path = args.selection.resolve().parent/'reports/evaluation_suite/selections_frozen.json'
    if sha(frozen_path) != selected['selection_sha256']:
        raise ValueError('Frozen validation selection SHA mismatch')
    frozen = json.loads(frozen_path.read_text(encoding='utf-8-sig'))
    reference = frozen['selections'][arm]
    if (config['sha256'] != reference['checkpoint_sha256'] or
        config['pipeline'] != reference['candidate']['pipeline'] or
        config['confidence'] != reference['operating_confidence'] or
        config['epoch'] != reference['candidate']['epoch'] or
        selected['recommended_arm'] != frozen['recommended_arm']):
        raise ValueError('Registry settings differ from frozen validation choices')
    weights = Path(config['checkpoint'])
    if not weights.is_absolute():
        weights = args.selection.resolve().parent/weights
    if sha(weights) != config['sha256']:
        raise ValueError('Checkpoint SHA differs from validation-frozen registry')
    paths = [args.input] if args.input.is_file() else sorted(p for p in args.input.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'})
    if not paths:
        parser.error('No input images')
    model = YOLO(str(weights))
    if model.task != 'detect' or list(model.names.values()) != NAMES:
        raise ValueError('Model must be a target4 bbox detector')
    args.output.mkdir(parents=True, exist_ok=True)
    kwargs = dict(imgsz=1024, conf=.001, iou=.6, max_det=1000, half=False, device=args.device, verbose=False)
    colors = [(50, 180, 245), (100, 200, 70), (240, 140, 70), (200, 80, 220)]
    inventory = []
    for index, path in enumerate(paths):
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot decode {path}')
        h, w = image.shape[:2]
        tiles = inference_plan(h, w, config['pipeline'])
        predictions = []
        began = time.perf_counter()
        if index == 0:
            ox, oy, ex, ey = tiles[0]['bounds']
            model.predict(image[oy:ey, ox:ex], **kwargs)
        for tile in tiles:
            ox, oy, ex, ey = tile['bounds']
            result = model.predict(image[oy:ey, ox:ex], **kwargs)[0]
            if tuple(result.orig_shape) != (ey-oy, ex-ox):
                raise RuntimeError('Unexpected original crop coordinates')
            if result.boxes is None:
                continue
            for box, cls, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                item = restore_prediction(box, tile['bounds'], cls, score)
                if item is not None:
                    predictions.append(item)
        raw_merged = deduplicate(predictions)
        visible = [p for p in raw_merged if p['score'] >= config['confidence']]
        elapsed = (time.perf_counter()-began)*1000
        folder = args.output/f'{index:04d}_{path.stem}'
        folder.mkdir()
        canvas = image.copy()
        for prediction in visible:
            cid = prediction['class_id']
            x1, y1, x2, y2 = map(round, prediction['bbox_xyxy'])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colors[cid], 2)
            label = f'{NAMES[cid]} {prediction["score"]:.2f}'
            cv2.putText(canvas, label, (max(0,x1), max(13,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .45, colors[cid], 1, cv2.LINE_AA)
        ok, encoded = cv2.imencode('.png', canvas)
        if not ok:
            raise RuntimeError('Preview encoding failed')
        encoded.tofile(folder/'annotated.png')
        report = {'utc': utc(), 'input': str(path.resolve()), 'input_sha256': sha(path),
                  'width': w, 'height': h, 'arm': arm, 'checkpoint_sha256': sha(weights),
                  'registry_sha256': sha(args.selection), 'frozen_selection_sha256': selected['selection_sha256'],
                  'pipeline': config['pipeline'],
                  'confidence': config['confidence'], 'tiles': tiles,
                  'raw_merged_predictions': raw_merged, 'predictions': visible,
                  'counts': dict(Counter(NAMES[p['class_id']] for p in visible)),
                  'task': 'target4_bbox_detection', 'component_masks': None, 'd455_verified': False,
                  'metrics': None, 'status': 'PREDICTIONS' if visible else 'NO_DETECTIONS_AT_SELECTED_CONFIDENCE',
                  'offline_elapsed_ms_including_first_call_setup': elapsed,
                  'notice': 'No ground truth supplied: counts/scores are predictions, not accuracy or calibrated probabilities. No camera was opened.'}
        save(folder/'predictions.json', report)
        inventory.append({'source': str(path.resolve()), 'result': str(folder.resolve()), 'predictions': len(visible)})
        print(f'{index+1}/{len(paths)} {path.name}: {len(visible)} detections', flush=True)
    save(args.output/'inference_manifest.json', {'arm': arm, 'inputs': inventory, 'registry_sha256': sha(args.selection),
                                                'frozen_selection_sha256': selected['selection_sha256']})


if __name__ == '__main__':
    main()
