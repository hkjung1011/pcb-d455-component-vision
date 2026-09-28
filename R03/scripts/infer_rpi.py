"""Offline RPi board/board-mask/component-box inference in original image coordinates.

No camera access. Checkpoints and confidence come from explicit flags or a
validation-selected registry. Native tiles are cropped BEFORE model letterboxing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'runtime_config'))
# Release bundles carry ROOT/src; the development workspace has a sibling package.
for source_root in (ROOT / 'src', ROOT.parent / 'pcb_components' / 'src'):
    if (source_root / 'pcb_components' / 'geometry.py').is_file():
        sys.path.insert(0, str(source_root))
        break

BOARD_NAMES = ['raspberry_pi_sbc']
PART_NAMES = ['resistor', 'capacitor', 'ic', 'connector']
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
EVALUATION_CONFIDENCE_FLOOR = .001
QUARANTINED_CHECKPOINT_SHA256 = {
    '54cafd9348bde613945218fb0696e703de76ddef6d09210ab30d092bd1e3f2d4':
        'Historical A/H/I board-class mapping contradicted by image review',
}


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input', required=True, type=Path, help='One image or image directory (nonrecursive)')
    p.add_argument('--output', required=True, type=Path, help='New, empty output directory')
    p.add_argument('--registry', type=Path, help='Selection JSON; auto-uses ROOT/selected_models.json if present')
    p.add_argument('--board-weights', type=Path, help='R03 single-class board DETECTION checkpoint')
    p.add_argument('--board-conf', type=float, help='Explicit operating confidence when no registry supplies it')
    p.add_argument('--board-imgsz', type=int, help='Board model letterbox size; explicit fallback 1024')
    p.add_argument('--board-seg-weights', type=Path, help='Optional R03 native board SEGMENTATION checkpoint')
    p.add_argument('--board-seg-conf', type=float, help='Board mask operating confidence')
    p.add_argument('--board-seg-imgsz', type=int, help='Board mask model letterbox size; explicit fallback 1024')
    p.add_argument('--no-board-mask', action='store_true', help='Disable optional registry board-mask role')
    p.add_argument('--parts-weights', type=Path, help='R03 four-class component DETECTION n/s checkpoint')
    p.add_argument('--parts-conf', type=float, help='Component operating confidence')
    p.add_argument('--parts-imgsz', type=int, help='Per-crop model letterbox size; explicit fallback 1024')
    p.add_argument('--parts-pipeline', choices=['native-tiles', 'whole'],
                   help='Registry choice; otherwise native-tiles. Whole allows model full-image resizing.')
    p.add_argument('--parts-tile-size', type=int, help='Original-pixel crop side; fallback 1024')
    p.add_argument('--overlap', type=float, help='Tile overlap fraction; fallback 0.2')
    p.add_argument('--board-roi', action='store_true', help='Restrict parts to detected board boxes; unbenchmarked cascade')
    p.add_argument('--roi-margin', type=float, default=0.10, help='Each-side margin as fraction of board box width/height')
    p.add_argument('--device', default='cpu', help='cpu or CUDA device, e.g. 0; CPU default does not require GPU')
    p.add_argument('--max-det', type=int, default=1000, help='Per model/crop cap and final tiled component cap')
    return p


def resolve_settings(args):
    registry_path = args.registry
    if registry_path is None and (ROOT / 'selected_models.json').is_file():
        registry_path = ROOT / 'selected_models.json'
    registry = {}
    if registry_path is not None:
        registry_path = registry_path.resolve()
        registry = json.loads(registry_path.read_text(encoding='utf-8-sig'))
        if registry.get('schema') != 'r03-inference-selection-v1':
            raise ValueError('Unsupported inference registry schema')
    roles = {}
    flags = [('board_detector', 'board'), ('board_segmenter', 'board_seg'), ('parts', 'parts')]
    overrides = []
    for role, prefix in flags:
        original = registry.get(role) or {}
        if original.get('training_eligible') is False or 'QUARANTIN' in str(original.get('status', '')).upper():
            raise ValueError(f'{role} registry entry is quarantined or ineligible')
        explicit = getattr(args, prefix + '_weights')
        if role == 'board_segmenter' and args.no_board_mask:
            if explicit is not None:
                raise ValueError('--no-board-mask conflicts with --board-seg-weights')
            overrides.append('board_segmenter disabled')
            continue
        checkpoint = explicit if explicit is not None else original.get('checkpoint')
        if checkpoint is None:
            if role == 'board_segmenter':
                if args.board_seg_conf is not None:
                    raise ValueError('--board-seg-conf requires a board segmentation checkpoint')
                continue
            raise ValueError(f'{role}: provide a checkpoint path explicitly or in the registry')
        checkpoint = Path(checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = ((registry_path.parent if explicit is None else Path.cwd()) / checkpoint).resolve()
        if not checkpoint.is_file():
            raise ValueError(f'{role}: checkpoint does not exist: {checkpoint}')
        confidence = getattr(args, prefix + '_conf')
        if confidence is None:
            confidence = original.get('confidence')
        if confidence is None or not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
            raise ValueError(f'{role}: an explicit confidence in [0,1] is required')
        imgsz = getattr(args, prefix + '_imgsz')
        imgsz = int(imgsz if imgsz is not None else original.get('imgsz', 1024))
        if imgsz <= 0 or imgsz % 32:
            raise ValueError(f'{role}: imgsz must be a positive multiple of 32')
        actual_hash = sha256(checkpoint)
        if actual_hash in QUARANTINED_CHECKPOINT_SHA256:
            raise ValueError(f'{role}: QUARANTINED_CLASS_MAPPING_CONFLICT: {QUARANTINED_CHECKPOINT_SHA256[actual_hash]}')
        if explicit is None and original.get('sha256') and actual_hash != original['sha256']:
            raise ValueError(f'{role}: registry checkpoint hash mismatch')
        role_settings = dict(original, checkpoint=str(checkpoint.resolve()), sha256=actual_hash,
                             confidence=float(confidence), imgsz=imgsz)
        for field, value in [('checkpoint', explicit), ('confidence', getattr(args, prefix + '_conf')),
                             ('imgsz', getattr(args, prefix + '_imgsz'))]:
            if value is not None:
                overrides.append(f'{role}.{field}')
        if role == 'parts':
            role_settings['pipeline'] = args.parts_pipeline or original.get('pipeline', 'native-tiles')
            role_settings['tile_size'] = int(args.parts_tile_size if args.parts_tile_size is not None else original.get('tile_size', 1024))
            role_settings['overlap'] = float(args.overlap if args.overlap is not None else original.get('overlap', 0.2))
            if role_settings['pipeline'] not in ['native-tiles', 'whole']:
                raise ValueError('Parts pipeline must be native-tiles or whole')
            if (role_settings['tile_size'] <= 0 and role_settings['pipeline']=='native-tiles') or not 0 <= role_settings['overlap'] < 1:
                raise ValueError('Tile size must be positive and overlap must be in [0,1)')
            for field in ['parts_pipeline', 'parts_tile_size', 'overlap']:
                if getattr(args, field) is not None:
                    overrides.append(field)
        roles[role] = role_settings
    if not math.isfinite(args.roi_margin) or not 0 <= args.roi_margin <= 1:
        raise ValueError('ROI margin must be finite and in [0,1]')
    if args.max_det < 1:
        raise ValueError('max-det must be positive')
    if args.max_det != 1000:
        overrides.append('max_det')
    return roles, {
        'registry': str(registry_path) if registry_path else None,
        'registry_sha256': sha256(registry_path) if registry_path else None,
        'explicit_overrides': overrides,
        'selection_status': 'USER_SUPPLIED_CONFIGURATION' if not registry else
                            ('REGISTRY_WITH_EXPLICIT_OVERRIDES' if overrides else 'REGISTRY_CONFIGURATION'),
        'cascade_status': 'UNBENCHMARKED_BOARD_TO_PARTS_ROI_CASCADE' if args.board_roi else 'FULL_FRAME_PARTS',
        'd455_verified': False,
    }


def restore_box_predictions(boxes, classes, scores, tile, names):
    """Translate clipped crop-local boxes into original image pixels, preserving index."""
    import numpy as np
    from pcb_components.geometry import restore_bbox
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    classes = np.asarray(classes, dtype=float).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(boxes) != len(classes) or len(boxes) != len(scores):
        raise ValueError('Prediction arrays have different lengths')
    if not all(np.isfinite(v).all() for v in (boxes, classes, scores)):
        raise ValueError('Nonfinite model prediction')
    output = []
    for index, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        class_id = int(cls)
        if class_id != cls or not 0 <= class_id < len(names) or not 0 <= score <= 1:
            raise ValueError('Invalid prediction class or score')
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        restored = restore_bbox(box, tile)
        if restored is None:
            continue
        output.append({'class_id': class_id, 'class_name': names[class_id], 'confidence': float(score),
                       'bbox_xyxy': list(restored), 'source_tile': tile.index, 'source_prediction_index': index})
    return output


def merge_boxes(predictions, iou=0.5, max_det=1000):
    """Same class-aware torchvision box NMS as native tile evaluation."""
    if not predictions:
        return []
    import torch
    from torchvision.ops import batched_nms
    keep = batched_nms(torch.tensor([r['bbox_xyxy'] for r in predictions], dtype=torch.float32),
                       torch.tensor([r['confidence'] for r in predictions], dtype=torch.float32),
                       torch.tensor([r['class_id'] for r in predictions], dtype=torch.int64), iou).tolist()
    return [predictions[index] for index in keep[:max_det]]


def make_regions(image_shape, board_predictions, use_roi=False, margin=0.10):
    from pcb_components.geometry import Rect
    h, w = image_shape[:2]
    if not use_roi:
        return [Rect(0, 0, w, h)]
    regions = []
    seen = set()
    for item in board_predictions:
        x1, y1, x2, y2 = item['bbox_xyxy']
        dx, dy = (x2-x1)*margin, (y2-y1)*margin
        bounds = (max(0, math.floor(x1-dx)), max(0, math.floor(y1-dy)),
                  min(w, math.ceil(x2+dx)), min(h, math.ceil(y2+dy)))
        if bounds[2] > bounds[0] and bounds[3] > bounds[1] and bounds not in seen:
            regions.append(Rect(*bounds))
            seen.add(bounds)
    return regions


def predict_boxes(model, image, config, args, regions, tiled=False):
    from pcb_components.geometry import Tile, generate_tiles
    all_predictions = []
    tile_metadata = []
    for region_index, region in enumerate(regions):
        tiles = generate_tiles(image.shape, config['tile_size'], config['overlap'], region) if tiled else (Tile(0, region, image.shape[:2]),)
        for tile in tiles:
            crop = tile.extract(image)
            # Match native evaluation: retain the low-score candidate pool through
            # model/cross-crop NMS, then apply the frozen operating threshold.
            result = model.predict(crop, imgsz=config['imgsz'], conf=EVALUATION_CONFIDENCE_FLOOR, iou=.6,
                                   max_det=args.max_det, device=args.device, verbose=False)[0]
            tile_id = len(tile_metadata)
            tile_metadata.append({'index': tile_id, 'region_index': region_index,
                                  'bounds_xyxy': list(tile.bounds.as_xyxy()), 'source_width': tile.width,
                                  'source_height': tile.height, 'model_imgsz': config['imgsz']})
            if result.boxes is None:
                continue
            restored = restore_box_predictions(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(),
                                                result.boxes.conf.cpu().numpy(), tile,
                                                [model.names[i] for i in range(len(model.names))])
            for item in restored:
                item['source_tile'] = tile_id
                item['source_region'] = region_index
            all_predictions.extend(restored)
    merged = merge_boxes(all_predictions, .5, args.max_det) if tiled or len(regions) > 1 else all_predictions
    retained = [item for item in merged if item['confidence'] >= config['confidence']]
    return retained, {'crops': tile_metadata, 'predictions_before_cross_crop_nms': len(all_predictions),
                    'predictions_after_cross_crop_nms': len(merged), 'predictions_after_operating_threshold': len(retained),
                    'raw_prediction_confidence_floor': EVALUATION_CONFIDENCE_FLOOR,
                    'operating_threshold': config.get('confidence'), 'operating_comparison': '>= after NMS',
                    'cross_crop_class_aware_box_nms_iou': .5 if tiled or len(regions) > 1 else None,
                    'pre_tiling_full_image_resized': False,
                    'model_preprocessing': 'Each native crop is independently letterboxed/resized by Ultralytics to imgsz; coordinates restored to crop pixels before translation.'}


def predict_board_masks(model, image, config, args, outdir):
    """Store authoritative native binary board masks as cropped PNGs with origins."""
    import cv2
    import numpy as np
    from pcb_components.geometry import Rect, Tile
    h, w = image.shape[:2]
    result = model.predict(image, imgsz=config['imgsz'], conf=EVALUATION_CONFIDENCE_FLOOR, iou=.6,
                           max_det=args.max_det, retina_masks=True, device=args.device, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return [], {'status': 'NO_DETECTIONS_AT_SELECTED_CONFIDENCE', 'empty_masks_skipped': 0, 'retina_masks': True}
    if result.masks is None or len(result.masks.data) != len(result.boxes):
        raise ValueError('Board segmentation boxes/masks are misaligned')
    records = restore_box_predictions(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(),
                                      result.boxes.conf.cpu().numpy(), Tile(0, Rect(0, 0, w, h), (h, w)), BOARD_NAMES)
    output = []
    skipped = 0
    for record in records:
        if record['confidence'] < config['confidence']:
            continue
        binary = result.masks.data[record['source_prediction_index']].cpu().numpy()
        if binary.shape != (h, w) or not np.isfinite(binary).all():
            raise ValueError('Expected finite native-resolution board mask; resizing is forbidden here')
        if not np.all((binary == 0) | (binary == 1)):
            raise ValueError('Expected binary native board mask values 0/1, as in native evaluation')
        binary = binary > .5
        yy, xx = np.nonzero(binary)
        if not len(xx):
            skipped += 1
            continue
        x1, x2, y1, y2 = int(xx.min()), int(xx.max())+1, int(yy.min()), int(yy.max())+1
        crop = binary[y1:y2, x1:x2].astype(np.uint8)*255
        filename = f'board_mask_{len(output):03d}.png'
        ok, encoded = cv2.imencode('.png', crop)
        if not ok:
            raise RuntimeError('Board mask PNG encoding failed')
        encoded.tofile(outdir/filename)
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outlines = [(c.reshape(-1, 2)+np.array([x1, y1])).tolist() for c in contours if len(c) >= 3]
        record.update(mask={'encoding': 'cropped_binary_png_0_255', 'file': filename,
                            'bounds_xyxy': [x1, y1, x2, y2], 'origin_xy': [x1, y1],
                            'original_image_size_wh': [w, h], 'foreground_pixels': int(binary.sum()),
                            'binary_png_is_authoritative': True},
                      exterior_contours_xy=outlines,
                      polygon_note='Exterior contours are for display only; PNG retains holes and all disconnected regions.',
                      mask_scope='WHOLE_RASPBERRY_PI_BOARD_NOT_COMPONENT_MASK')
        output.append(record)
    return output, {'status': 'DETECTIONS_PRESENT' if output else 'NO_NONEMPTY_MASK_AT_SELECTED_CONFIDENCE',
                    'empty_masks_skipped': skipped, 'retina_masks': True,
                    'raw_prediction_confidence_floor': EVALUATION_CONFIDENCE_FLOOR,
                    'operating_threshold': config['confidence'], 'operating_comparison': '>= after model NMS',
                    'empty_mask_note': 'Empty bitmaps are omitted from display/export and counted here; native mask evaluation still scores empty mask predictions.'}


def annotate(image, boards, board_masks, parts, output_folder, status_text):
    import cv2
    import numpy as np
    canvas = image.copy()
    h, w = canvas.shape[:2]
    for item in board_masks:
        mask = item['mask']
        x1, y1, x2, y2 = mask['bounds_xyxy']
        crop = cv2.imdecode(np.fromfile(output_folder/mask['file'], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        patch = canvas[y1:y2, x1:x2]
        foreground = crop > 0
        patch[foreground] = (.78*patch[foreground]+.22*np.array([255, 100, 30])).astype(np.uint8)
    text_scale = max(.4, min(1.0, max(h, w)/2100))
    for collection, role in [(boards, 'board'), (parts, 'part')]:
        for item in collection:
            x1, y1, x2, y2 = item['bbox_xyxy']
            color = (255, 130, 25) if role == 'board' else [(40, 210, 40), (20, 210, 240), (220, 80, 180), (240, 190, 50)][item['class_id']]
            a = (max(0, int(math.floor(x1))), max(0, int(math.floor(y1))))
            b = (min(w-1, int(math.ceil(x2))-1), min(h-1, int(math.ceil(y2))-1))
            cv2.rectangle(canvas, a, b, color, 2 if role == 'board' else 1)
            name = 'RPi board' if role == 'board' else item['class_name']
            cv2.putText(canvas, f'{name} {item["confidence"]:.2f}', (a[0], max(14, a[1]-3)),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, 1, cv2.LINE_AA)
    # A visible prediction summary is not a statement that low-confidence objects are absent.
    cv2.rectangle(canvas, (0, 0), (w, min(h, 31)), (20, 20, 20), -1)
    cv2.putText(canvas, status_text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode('.png', canvas)
    if not ok:
        raise RuntimeError('Overlay PNG encoding failed')
    encoded.tofile(output_folder/'annotated.png')


def sync_if_cuda(device):
    import torch
    if str(device).lower() != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        roles, selection = resolve_settings(args)
        if args.input.is_file():
            files = [args.input.resolve()]
        elif args.input.is_dir():
            files = sorted(v.resolve() for v in args.input.iterdir() if v.is_file() and v.suffix.lower() in EXTENSIONS)
        else:
            raise ValueError('Input does not exist')
        if not files:
            raise ValueError('Input directory has no supported image files')
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError('Output directory must be new or empty')
    except (ValueError, OSError) as error:
        p.error(str(error))
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from pcb_components.geometry import Rect
    models = {}
    for role, config in roles.items():
        model = YOLO(config['checkpoint'])
        names = [str(model.names[i]).lower() for i in range(len(model.names))]
        expected = PART_NAMES if role == 'parts' else BOARD_NAMES
        expected_task = 'segment' if role == 'board_segmenter' else 'detect'
        if names != expected or model.task != expected_task:
            p.error(f'{role}: requires {expected_task} with exact classes {expected}; got {model.task} {names}')
        models[role] = model
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output/'effective_configuration.json', dict(selection=selection, roles=roles,
        input=str(args.input.resolve()), device=args.device, max_det=args.max_det,
        board_roi=args.board_roi, roi_margin=args.roi_margin,
        component_mask_model_available=False, model_scores_are_calibrated_probabilities=False))
    summaries = []
    for image_path in files:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Image cannot be decoded: {image_path}')
        h, w = image.shape[:2]
        output_folder = args.output/(image_path.stem+'_'+hashlib.sha256(str(image_path).encode()).hexdigest()[:10])
        output_folder.mkdir()
        sync_if_cuda(args.device)
        start = time.perf_counter()
        boards, board_process = predict_boxes(models['board_detector'], image, roles['board_detector'], args,
                                               [Rect(0, 0, w, h)])
        board_masks, mask_process = ([], {'status': 'NOT_REQUESTED'})
        if 'board_segmenter' in models:
            board_masks, mask_process = predict_board_masks(models['board_segmenter'], image, roles['board_segmenter'], args, output_folder)
        regions = make_regions(image.shape, boards, args.board_roi, args.roi_margin)
        parts, part_process = predict_boxes(models['parts'], image, roles['parts'], args, regions,
                                            tiled=roles['parts']['pipeline']=='native-tiles')
        sync_if_cuda(args.device)
        elapsed = time.perf_counter()-start
        part_status = 'DETECTIONS_PRESENT' if parts else ('SKIPPED_NO_BOARD_ROI' if args.board_roi and not regions else 'NO_DETECTIONS_AT_SELECTED_CONFIDENCE')
        board_status = 'DETECTIONS_PRESENT' if boards else 'NO_DETECTIONS_AT_SELECTED_CONFIDENCE'
        status_text = f'Board {len(boards)} | board mask {len(board_masks)} | parts {len(parts)} | offline / D455 unverified'
        annotate(image, boards, board_masks, parts, output_folder, status_text)
        report = {'schema': 'r03-offline-inference-v1', 'input': str(image_path), 'input_sha256': sha256(image_path),
            'width': w, 'height': h, 'coordinate_space': 'Original decoded image pixel edges; xyxy right/bottom exclusive',
            'selection': selection, 'effective_models': roles,
            'scope': 'RPi whole-board detection and optional whole-board masks; four-component BOX detection only',
            'component_masks': {'status': 'NOT_AVAILABLE_IN_R03_COMPONENT_DETECTORS'},
            'no_detection_interpretation': 'No retained prediction at configured threshold; this does not establish absence of the object.',
            'board_status': board_status, 'component_status': part_status,
            'board_detections': boards, 'board_masks': board_masks, 'component_detections': parts,
            'board_processing': board_process, 'mask_processing': mask_process, 'parts_processing': part_process,
            'parts_regions_xyxy': [list(r.as_xyxy()) for r in regions],
            'timing': {'seconds': elapsed, 'includes_first_predict_warmup': True,
                       'scope': 'Board + optional mask + component inference/postprocessing; includes mask PNG writes; excludes image decode/overlay/report writes. Not benchmark latency.'},
            'd455_verified': False}
        save(output_folder/'prediction.json', report)
        summary = {'image': str(image_path), 'output': str(output_folder.resolve()), 'boards': len(boards),
                   'board_masks': len(board_masks), 'components': len(parts), 'component_status': part_status}
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    save(args.output/'index.json', {'images': summaries, 'selection': selection})


if __name__ == '__main__':
    main()
