#!/usr/bin/env python3
"""Create a small, explicitly synthetic prototype dataset from inspected boxes.

Train and development images use different transforms of the SAME source scene.
The development scores are not evidence of accuracy on unseen physical boards.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
SEED = json.loads((ROOT / 'seed-labels.json').read_text())
NAMES = SEED['classes']
WIDTH, HEIGHT = 1280, 800


def draw_boxes(image, boxes, destination):
    image = image.copy()
    for cls, x1, y1, x2, y2 in boxes:
        color = ((37 * int(cls) + 90) % 240 + 15, (71 * int(cls) + 40) % 240 + 15,
                 (53 * int(cls) + 100) % 240 + 15)
        p1, p2 = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(image, p1, p2, color, 2)
        cv2.putText(image, NAMES[int(cls)], (p1[0], max(15, p1[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, .40, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(destination), image)


def overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy > 0


def create_image(source, backgrounds, rng):
    image = backgrounds[int(rng.integers(len(backgrounds)))].copy()
    used = []
    boxes = []
    wanted = [] if rng.random() < .12 else list(rng.permutation(2)[:int(rng.integers(1, 3))])
    for index in wanted:
        board = SEED['boards'][index]
        x1, y1, x2, y2 = board['crop']
        crop = source[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        angle = float(rng.uniform(0, 360))
        scale = float(rng.uniform(.70, 1.25))
        transform = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        rotated = cv2.transform(corners[None], transform)[0]
        low, high = rotated.min(axis=0), rotated.max(axis=0)
        bw, bh = np.ceil(high - low).astype(int)
        location = None
        for _ in range(60):
            left = int(rng.integers(8, max(9, WIDTH - bw - 8)))
            top = int(rng.integers(8, max(9, HEIGHT - bh - 8)))
            candidate = [left, top, left + bw, top + bh]
            if candidate[2] < WIDTH and candidate[3] < HEIGHT and not any(overlap(candidate, other) for other in used):
                location = candidate
                break
        if location is None:
            continue
        used.append(location)
        transform[:, 2] += np.array(location[:2]) - low
        warped = cv2.warpAffine(crop, transform, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), transform, (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST)
        image[mask > 0] = warped[mask > 0]
        for cls, bx1, by1, bx2, by2 in board['boxes']:
            points = np.float32([[bx1-x1, by1-y1], [bx2-x1, by1-y1], [bx2-x1, by2-y1], [bx1-x1, by2-y1]])
            points = cv2.transform(points[None], transform)[0]
            lo, hi = points.min(axis=0), points.max(axis=0)
            boxes.append([cls, *lo.tolist(), *hi.tolist()])
    image = np.clip(image.astype(np.float32) * rng.uniform(.78, 1.18) + rng.uniform(-12, 12), 0, 255).astype(np.uint8)
    if rng.random() < .25:
        image = cv2.GaussianBlur(image, (3, 3), .5)
    return image, boxes


def main():
    source = cv2.imread(str(ROOT / 'source/boards.png'))
    assert source is not None and source.shape[:2] == (HEIGHT, WIDTH)
    all_boxes = [b for board in SEED['boards'] for b in board['boxes']]
    draw_boxes(source, all_boxes, ROOT / 'seed-annotations.jpg')
    empty = cv2.imread(str(ROOT / 'source/background.png'))
    assert empty is not None
    backgrounds = [cv2.resize(empty, (WIDTH, HEIGHT)),
                   cv2.resize(source[:, :400], (WIDTH, HEIGHT)),
                   cv2.resize(source[:, 955:], (WIDTH, HEIGHT)),
                   np.full((HEIGHT, WIDTH, 3), 45, np.uint8)]
    destination = ROOT / 'dataset-prototype'
    manifest = {'source_scene_count': 1, 'physical_boards': 2,
                'limitations': 'Synthetic transformations of one camera scene. Development split is not an independent real-world test.',
                'source': SEED['source'], 'classes': NAMES, 'splits': {}}
    for split, count, seed in [('train', 160, 20260929), ('val', 32, 20260930)]:
        image_dir, label_dir = destination / 'images' / split, destination / 'labels' / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        distribution = dict.fromkeys(NAMES, 0)
        negatives = 0
        for index in range(count):
            image, boxes = create_image(source, backgrounds, rng)
            name = f'{split}-{index:04d}'
            cv2.imwrite(str(image_dir / f'{name}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lines = []
            for cls, x1, y1, x2, y2 in boxes:
                assert 0 <= x1 < x2 <= WIDTH and 0 <= y1 < y2 <= HEIGHT
                lines.append(f'{cls} {(x1+x2)/2/WIDTH:.7f} {(y1+y2)/2/HEIGHT:.7f} {(x2-x1)/WIDTH:.7f} {(y2-y1)/HEIGHT:.7f}')
                distribution[NAMES[cls]] += 1
            negatives += not bool(boxes)
            (label_dir / f'{name}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
            if index < 4:
                draw_boxes(image, boxes, destination / f'preview-{split}-{index}.jpg')
        manifest['splits'][split] = {'images': count, 'negative_images': negatives, 'instances': distribution, 'random_seed': seed}
    (destination / 'data.yaml').write_text(yaml.safe_dump({'path': str(destination), 'train': 'images/train', 'val': 'images/val', 'names': dict(enumerate(NAMES))},sort_keys=False))
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
