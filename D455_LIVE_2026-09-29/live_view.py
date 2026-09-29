#!/usr/bin/env python3
"""D455 preview combining R04 with the existing local board/port prototype."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
PACKAGE = Path(os.environ.get('PCB_MODEL_ROOT', str(ROOT.parent)))
sys.path.insert(0, str(PACKAGE / 'R04/scripts'))
from evaluate_r04 import NAMES, sha, inference_plan, restore_prediction, deduplicate

WINDOW = 'D455 - R04 PCB Parts - Live'
KO = ['저항', '커패시터', 'IC', '커넥터']
COLORS = [(246, 192, 87), (75, 211, 157), (113, 174, 255), (231, 138, 210)]
DETAIL_NAMES = ['usb_stack', 'ethernet_port', 'gpio_header', 'micro_hdmi', 'usb_c_power', 'heatsink']
CLASS_NAMES = NAMES + DETAIL_NAMES
KO += ['USB 묶음', '랜 포트', 'GPIO', 'micro HDMI', 'USB-C', '방열판']
SHORT_KO = ['저항', '커패시터', 'IC', '커넥터', 'USB', '랜', 'GPIO', 'HDMI', 'USB-C', '방열판']
COLORS += [(255, 195, 104), (92, 223, 217), (186, 231, 91), (242, 163, 244), (153, 202, 255), (211, 197, 255)]
PROTOTYPE = ROOT / 'models/board-parts-prototype.pt'
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONTS = {size: ImageFont.truetype(FONT_PATH, size) for size in (15, 17, 19, 23, 28)}
HEADER = 68
SIDE = 380


def write_json(path, data):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)


def write_image(path, data):
    tmp = path.with_name(path.stem + '.tmp' + path.suffix)
    if not cv2.imwrite(str(tmp), data):
        raise RuntimeError(f'Cannot write image: {path}')
    os.replace(tmp, path)


class Detector:
    def __init__(self, device='0'):
        self.device = device
        self.selection = json.loads((PACKAGE / 'R04/selected_models.json').read_text())
        self.config = self.selection['arms']['improved']
        frozen_file = PACKAGE / 'R04/reports/evaluation_suite/selections_frozen.json'
        if sha(frozen_file) != self.selection['selection_sha256']:
            raise ValueError('Frozen model selection hash mismatch')
        reference = json.loads(frozen_file.read_text())['selections']['improved']
        if (self.config['sha256'] != reference['checkpoint_sha256'] or
                self.config['confidence'] != reference['operating_confidence'] or
                self.config['pipeline'] != reference['candidate']['pipeline']):
            raise ValueError('R04 configuration differs from the frozen selection')
        weights = PACKAGE / 'R04' / self.config['checkpoint']
        if sha(weights) != self.config['sha256']:
            raise ValueError('R04 checkpoint hash mismatch')
        self.parts = YOLO(str(weights))
        if self.parts.task != 'detect' or list(self.parts.names.values()) != NAMES:
            raise ValueError('Expected the four-class R04 detector')
        self.board_config = json.loads((PACKAGE / 'R03/selected_models.json').read_text())['board_detector']
        board_weights = PACKAGE / 'R03' / self.board_config['checkpoint']
        if sha(board_weights) != self.board_config['sha256']:
            raise ValueError('Board checkpoint hash mismatch')
        self.board = YOLO(str(board_weights))
        self.detail = YOLO(str(PROTOTYPE))
        if self.detail.task != 'detect' or list(self.detail.names.values()) != ['raspberry_pi_5', 'raspberry_pi_4'] + DETAIL_NAMES:
            raise ValueError('Unexpected classes in the local board/port prototype')
        self.metadata = {
            'parts_model': 'R04 improved epoch16', 'parts_sha256': self.config['sha256'],
            'board_model': 'R03 board_yolo11n', 'board_sha256': self.board_config['sha256'],
            'pipeline': self.config['pipeline'], 'parts_threshold': self.config['confidence'],
            'board_threshold': self.board_config['confidence'], 'imgsz': 1024,
            'raw_confidence': 0.001, 'tile_nms': 0.6, 'merge_nms': 0.5,
            'precision': 'FP32', 'device': device,
            'runtime': {p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'ultralytics', 'pyrealsense2']},
            'scope': 'Live predictions; no ground-truth accuracy measurement',
            'additional_model': {
                'name': 'local board-parts-prototype', 'path': str(PROTOTYPE),
                'sha256': sha(PROTOTYPE), 'classes': self.detail.names,
                'imgsz': 1024, 'confidence': .6, 'iou': .5,
                'training_scope': 'Synthetic variations of one source scene containing these two boards; not an independent real-world test.',
            },
            'combined_display': 'R04 four-class detections plus local prototype port/heatsink detections; R toggles the original R04/R03 display.',
        }

    def predict(self, image):
        began = time.perf_counter()
        tiles = inference_plan(*image.shape[:2], self.config['pipeline'])
        predictions = []
        for tile in tiles:
            ox, oy, ex, ey = tile['bounds']
            result = self.parts.predict(
                image[oy:ey, ox:ex], imgsz=1024, conf=.001, iou=.6,
                max_det=1000, quantize=None, device=self.device, verbose=False,
            )[0]
            if tuple(result.orig_shape) != (ey - oy, ex - ox):
                raise RuntimeError('Unexpected crop coordinate system')
            for box, cls, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                item = restore_prediction(box, tile['bounds'], cls, score)
                if item is not None:
                    predictions.append(item)
        merged = deduplicate(predictions)
        board_result = self.board.predict(
            image, imgsz=self.board_config['imgsz'], conf=self.board_config['confidence'],
            iou=.7, max_det=300, quantize=None, device=self.device, verbose=False,
        )[0]
        boards = [{'bbox_xyxy': box.tolist(), 'score': float(score), 'class_name': board_result.names[int(cls)]}
                  for box, score, cls in zip(board_result.boxes.xyxy.cpu(), board_result.boxes.conf.cpu(), board_result.boxes.cls.cpu())]
        detail_result = self.detail.predict(
            image, imgsz=1024, conf=.25, iou=.5, max_det=300,
            quantize=None, device=self.device, verbose=False,
        )[0]
        detail_boards, additional_parts = [], []
        for box, cls, score in zip(detail_result.boxes.xyxy.cpu(), detail_result.boxes.cls.cpu(), detail_result.boxes.conf.cpu()):
            cls = int(cls)
            item = {'bbox_xyxy': box.tolist(), 'score': float(score), 'model_class_id': cls,
                    'class_name': detail_result.names[cls], 'source': 'local_board_parts_prototype'}
            if cls < 2:
                detail_boards.append(item)
            else:
                item['class_id'] = cls + 2
                additional_parts.append(item)
        return {'parts': merged, 'boards': boards, 'tiles': len(tiles),
                'additional_parts': additional_parts, 'detail_boards': detail_boards,
                'inference_ms': (time.perf_counter() - began) * 1000}


class Camera:
    def __init__(self, config):
        self.pipeline = rs.pipeline()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.times = deque(maxlen=120)
        self.error = None
        cfg = rs.config()
        cfg.enable_device(config['serial'])
        cfg.disable_all_streams()
        c = config['color']
        cfg.enable_stream(rs.stream.color, c['width'], c['height'], rs.format.bgr8, c['fps'])
        self.profile = self.pipeline.start(cfg)
        try:
            for sensor in self.profile.get_device().query_sensors():
                if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
                    for key, value in config.get('options', {}).get('RGB Camera', {}).items():
                        sensor.set_option(getattr(rs.option, key), value)
            self.thread = threading.Thread(target=self.read, daemon=True)
            self.thread.start()
        except Exception:
            self.pipeline.stop()
            raise

    def read(self):
        try:
            while not self.stop.is_set():
                frames = self.pipeline.wait_for_frames(2000)
                frame = frames.get_color_frame()
                if not frame:
                    continue
                now = time.monotonic()
                sample = (frame.get_frame_number(), now, np.asanyarray(frame.get_data()).copy())
                with self.lock:
                    self.latest = sample
                    self.times.append(now)
        except Exception as exc:
            if not self.stop.is_set():
                self.error = str(exc)

    def sample(self):
        with self.lock:
            fps = (len(self.times) - 1) / (self.times[-1] - self.times[0]) if len(self.times) > 1 else 0
            return self.latest, fps

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
        self.pipeline.stop()


class View:
    def __init__(self):
        self.selection = None
        self.candidates = []
        self.diagnostic = False
        self.expanded = True
        self.show_boards = True
        self.paused = False

    def describe(self, result):
        r04_threshold = .25 if self.diagnostic else .65
        detail_threshold = .25 if self.diagnostic else .6
        parts = [dict(p, source='R04', class_name=NAMES[p['class_id']])
                 for p in result['parts'] if p['score'] >= r04_threshold]
        if self.expanded:
            parts += [p for p in result['additional_parts'] if p['score'] >= detail_threshold]
            boards = [p for p in result['detail_boards'] if p['score'] >= detail_threshold]
        else:
            boards = result['boards']
        # Spatial ordering keeps drawing and click selection consistent between frames.
        parts.sort(key=lambda p: (int(p['bbox_xyxy'][1] // 40), p['bbox_xyxy'][0], p['class_id']))
        return {'display_mode': 'combined_prototype' if self.expanded else 'R04_R03_reference',
                'display_thresholds': {'R04': r04_threshold, 'local_board_parts_prototype': detail_threshold if self.expanded else None},
                'counts': dict(Counter(CLASS_NAMES[p['class_id']] for p in parts)),
                'parts': parts, 'boards': boards,
                'board_counts': dict(Counter(p['class_name'] for p in boards)),
                'paused': self.paused, 'low_score_candidates_enabled': self.diagnostic}

    def click(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        y -= HEADER
        containing = [p for p in self.candidates if p['bbox_xyxy'][0] <= x <= p['bbox_xyxy'][2]
                      and p['bbox_xyxy'][1] <= y <= p['bbox_xyxy'][3]]
        if containing:
            p = min(containing, key=lambda p: (p['bbox_xyxy'][2]-p['bbox_xyxy'][0])*(p['bbox_xyxy'][3]-p['bbox_xyxy'][1]))
            b = p['bbox_xyxy']
            self.selection = (p['class_id'], (b[0]+b[2])/2, (b[1]+b[3])/2)

    def render(self, image, result, stats):
        h, w = image.shape[:2]
        canvas = Image.new('RGB', (w + SIDE, max(h + HEADER, 840)), (17, 22, 30))
        canvas.paste(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), (0, HEADER))
        draw = ImageDraw.Draw(canvas)
        def text(x, y, value, size=17, color=(210, 218, 230)):
            draw.text((x, y), value, font=FONTS[size], fill=color)
        text(20, 10, 'PCB 객체 · 세부 부품 인식', 28, (245, 247, 250))
        text(420, 22, '일시 정지' if self.paused else 'D455 실시간 RGB', 19, (85, 216, 171))
        text(w + 22, 20, 'R04 + 보드·포트 시험 모델' if self.expanded else 'R04 / R03 원본 비교', 19)
        display = self.describe(result)
        if self.show_boards:
            for i, board in enumerate(display['boards'], 1):
                x1, y1, x2, y2 = board['bbox_xyxy']
                draw.rectangle((x1, y1+HEADER, x2, y2+HEADER), outline=(245, 245, 245), width=2)
                name = {'raspberry_pi_5': 'Pi 5', 'raspberry_pi_4': 'Pi 4'}.get(board['class_name'], 'RPi board')
                label = f'{name}  {board["score"]:.2f}'
                bx = max(0, x1-145)
                draw.rectangle((bx, y1+HEADER, min(w, bx+142), y1+HEADER+26), fill=(25, 33, 43))
                text(bx+5, y1+HEADER+2, label, 17)
        visible = display['parts']
        self.candidates = visible
        labels = []
        for part in visible:
            cls, score = part['class_id'], part['score']
            x1, y1, x2, y2 = part['bbox_xyxy']
            color = COLORS[cls]
            low = score < (.65 if cls < 4 else .6)
            draw.rectangle((x1, y1+HEADER, x2, y2+HEADER), outline=color, width=1 if low else 3)
            label = f'{SHORT_KO[cls]} {score:.2f}' + (' ?' if low else '')
            ty = max(HEADER, y1+HEADER-25)
            label_w = draw.textlength(label, font=FONTS[15]) + 8
            tx = max(0, min(x1, w-label_w))
            # Separate labels of adjacent HDMI/USB connectors while preserving their boxes.
            for _ in range(8):
                if not any(tx < r and tx+label_w > l and ty < b and ty+25 > t for l,t,r,b in labels):
                    break
                ty = max(HEADER, ty-26)
            labels.append((tx,ty,tx+label_w,ty+25))
            if abs(ty-(y1+HEADER-25)) > 2:
                draw.line((x1,y1+HEADER,tx+label_w/2,ty+25),fill=color,width=1)
            draw.rectangle((tx, ty, tx+label_w, ty+25), fill=(20, 25, 33))
            text(tx+4, ty, label, 15, color)
        x = w + 22
        text(x, 80, f'부품 후보 {len(visible)}개 · 보드 {len(display["boards"])}개', 23, (245, 247, 250))
        text(x, 116, '낮은 점수 후보 포함 · 0.25' if self.diagnostic else 'R04 0.65 / 보드·포트 0.60', 15)
        counts = Counter(p['class_id'] for p in visible)
        text(x, 154, 'R04 · IC / 소형 부품', 17)
        for i in range(4):
            text(x+(i%2)*172, 183+(i//2)*31, f'{KO[i]}  {counts[i]}', 19, COLORS[i])
        text(x, 254, '보드·포트 시험 모델' + ('' if self.expanded else ' · 표시 꺼짐'), 17)
        for i in range(4,10):
            text(x+((i-4)%2)*172, 284+((i-4)//2)*31, f'{KO[i]}  {counts[i]}', 17, COLORS[i])
        text(x, 382, '시험 모델: 단일 장면 학습, 일반화 미검증', 15, (238, 193, 123))
        text(x, 414, f'카메라 {stats["camera_fps"]:.1f} / 추론 {stats["inference_fps"]:.1f} fps', 17)
        text(x, 446, '부품을 클릭하면 확대됩니다', 19, (245, 247, 250))
        selected = visible[0] if visible else None
        if visible and self.selection:
            cls, cx, cy = self.selection
            nearby = [p for p in visible if p['class_id'] == cls and
                      abs((p['bbox_xyxy'][0]+p['bbox_xyxy'][2])/2-cx) < 80 and
                      abs((p['bbox_xyxy'][1]+p['bbox_xyxy'][3])/2-cy) < 80]
            if nearby:
                selected = min(nearby, key=lambda p: ((p['bbox_xyxy'][0]+p['bbox_xyxy'][2])/2-cx)**2 + ((p['bbox_xyxy'][1]+p['bbox_xyxy'][3])/2-cy)**2)
        draw.rounded_rectangle((x, 480, x+336, 662), radius=8, fill=(29, 36, 47))
        if selected:
            b = selected['bbox_xyxy']
            x1, y1, x2, y2 = map(lambda v: int(round(v)), b)
            crop = image[max(0,y1-12):min(h,y2+12), max(0,x1-12):min(w,x2+12)]
            zoom = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            scale = min(324/zoom.width, 170/zoom.height)
            zoom = zoom.resize((max(1,round(zoom.width*scale)), max(1,round(zoom.height*scale))), Image.Resampling.NEAREST)
            canvas.paste(zoom, (x+(336-zoom.width)//2, 480+(182-zoom.height)//2))
            text(x, 672, f'{KO[selected["class_id"]]}  {selected["score"]:.2f} · {x2-x1}×{y2-y1}px', 17, COLORS[selected['class_id']])
            text(x, 700, '검출 출처: ' + ('R04' if selected['source']=='R04' else '보드·포트 시험 모델'), 15)
        else:
            text(x+22, 548, '현재 표시 기준에서 검출 없음', 17)
        text(x, 737, 'R  전체 부품 / R04 원본    D  낮은 점수', 15)
        text(x, 764, 'B  보드 표시      Space  일시 정지', 15)
        text(x, 791, 'S  사진·검출 저장      Q / Esc  종료', 15)
        text(x, 827, '확대 영상은 원본 픽셀입니다.', 15, (145, 159, 180))
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'camera.example.json')
    parser.add_argument('--device', default='0')
    parser.add_argument('--max-fps', type=float, default=5)
    parser.add_argument('--seconds', type=float, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-image', type=Path)
    parser.add_argument('--reference', action='store_true', help='Start with the original R04/R03 display')
    args = parser.parse_args()
    if args.max_fps <= 0:
        parser.error('--max-fps must be positive')
    output = args.output or ROOT/'sessions'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    detector = Detector(args.device)
    view = View()
    view.expanded = not args.reference
    if args.check_image:
        image = cv2.imread(str(args.check_image))
        if image is None:
            raise ValueError('Cannot read preflight image')
        result = detector.predict(image)
        write_json(output/'predictions.json', {**result, 'display':view.describe(result), 'models':detector.metadata})
        write_image(output/'preview.png', view.render(image, result, {'camera_fps':0, 'inference_fps':0}))
        print(json.dumps({'parts':view.describe(result)['counts'], 'boards':view.describe(result)['board_counts']}, ensure_ascii=False))
        return
    lock = (ROOT/'camera.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = threading.Event()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda _sig, _frame: stop.set())
    config = json.loads(args.config.read_text())
    camera = None
    state = {}
    try:
        detector.predict(np.zeros((config['color']['height'], config['color']['width'], 3), np.uint8))
        camera = Camera(config)
        write_json(output/'startup.json', {**detector.metadata, 'camera_config':config,
                   'active_profiles':[str(p) for p in camera.profile.get_streams()]})
        write_json(ROOT/'current-session.json', {'pid':os.getpid(), 'output':str(output), 'window':WINDOW})
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW, 1560, 830)
        cv2.moveWindow(WINDOW, 35, 35)
        cv2.setMouseCallback(WINDOW, view.click)
        inference_times = deque(maxlen=120)
        durations = deque(maxlen=300)
        start = time.monotonic()
        last_sequence = -1
        next_inference = next_save = start
        image = result = canvas = None
        while not stop.is_set() and (args.seconds <= 0 or time.monotonic()-start < args.seconds):
            now = time.monotonic()
            sample, camera_fps = camera.sample()
            if camera.error:
                raise RuntimeError(camera.error)
            if sample is None and now-start > 6:
                raise RuntimeError('No camera frames received')
            if sample is not None and now-sample[1] > 3:
                raise RuntimeError('Camera frames are stale')
            if not view.paused and sample is not None and sample[0] != last_sequence and now >= next_inference:
                sequence, received, image = sample
                result = detector.predict(image)
                finished = time.monotonic()
                inference_times.append(finished)
                durations.append(result['inference_ms'])
                fps = (len(inference_times)-1)/(inference_times[-1]-inference_times[0]) if len(inference_times)>1 else 0
                primary = [p for p in result['parts'] if p['score'] >= .65]
                state = {'updated_at':datetime.now().astimezone().isoformat(), 'elapsed_seconds':finished-start,
                         'frame_number':sequence, 'image_shape':list(image.shape), 'camera_fps':camera_fps,
                         'inference_fps':fps, 'inference_ms':result['inference_ms'],
                         'frame_to_result_ms':(finished-received)*1000,
                         'inference_ms_p50':float(np.median(durations)), 'inference_ms_p95':float(np.percentile(durations,95)),
                         'r04_threshold':.65, 'r04_counts':dict(Counter(NAMES[p['class_id']] for p in primary)),
                         'r04_parts':primary, 'r04_diagnostic_candidates':[p for p in result['parts'] if .25 <= p['score'] < .65],
                         'r03_boards':result['boards'], 'additional_parts':result['additional_parts'],
                         'detail_boards':result['detail_boards'], 'tile_count':result['tiles'],
                         'models':detector.metadata, 'accuracy_evaluated':False}
                last_sequence = sequence
                next_inference = now + 1/args.max_fps
            if result is not None:
                state.update(view.describe(result))
                canvas = view.render(image, result, state)
                cv2.imshow(WINDOW, canvas)
                if now >= next_save:
                    write_json(output/'live.json', state)
                    write_image(output/'latest-input.png', image)
                    write_image(output/'latest.png', canvas)
                    print(json.dumps({k:state[k] for k in ['frame_number','counts','camera_fps','inference_fps','inference_ms']},ensure_ascii=False),flush=True)
                    next_save = now + 3
            key = cv2.waitKey(15) & 0xff
            if key in [27, ord('q')]:
                break
            if key == ord('d'):
                view.diagnostic = not view.diagnostic
            if key == ord('r'):
                view.expanded = not view.expanded
            if key == ord('b'):
                view.show_boards = not view.show_boards
            if key == 32:
                view.paused = not view.paused
            if key == ord('s') and canvas is not None:
                folder = output/'snapshots'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
                folder.mkdir(parents=True)
                write_image(folder/'input.png', image)
                write_image(folder/'annotated.png', canvas)
                write_json(folder/'predictions.json', state)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except Exception as exc:
        write_json(output/'error.json', {'error':str(exc)})
        raise
    finally:
        if camera:
            camera.close()
        cv2.destroyAllWindows()
        write_json(output/'final.json', state)
        lock.close()


if __name__ == '__main__':
    main()
