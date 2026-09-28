"""Local D455 inventory or user-triggered bounded raw capture; no reset/firmware changes."""
import argparse
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--capture', action='store_true', help='Only when the target PCB is in view')
    p.add_argument('--count', type=int, default=10)
    p.add_argument('--interval', type=float, default=1.0)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=800)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--board-id', default='unassigned')
    p.add_argument('--session-id', default='unassigned')
    p.add_argument('--design-id', default='unknown')
    p.add_argument('--side', choices=['top','bottom','other','unknown'], default='unknown')
    p.add_argument('--distance-mm', type=float, help='Manually measured lens-to-board distance; not estimated depth')
    p.add_argument('--lighting-id', default='unknown')
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Choose a new output directory to preserve earlier camera/frame records')
    if not 1 <= args.count <= 200 or not 0.2 <= args.interval <= 60:
        p.error('count must be 1..200 and interval 0.2..60 seconds')
    if args.distance_mm is not None and (not math.isfinite(args.distance_mm) or args.distance_mm <= 0):
        p.error('distance-mm must be positive when measured')
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+', value) for value in (args.board_id, args.session_id)):
        p.error('board/session IDs must use ASCII letters, digits, hyphens or underscores')
    import pyrealsense2 as rs
    context = rs.context()
    devices = list(context.query_devices())
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'device_count': len(devices),
              'capture_attempted': False, 'frames_saved': 0, 'devices': []}
    targets = []
    for device in devices:
        info = {}
        for field in ['name', 'serial_number', 'firmware_version', 'usb_type_descriptor']:
            key = getattr(rs.camera_info, field)
            if device.supports(key):
                info[field] = device.get_info(key)
        profiles = []
        sensor_options = []
        for sensor in device.query_sensors():
            options = {}
            for option_name in ['exposure', 'gain', 'white_balance', 'enable_auto_exposure', 'enable_auto_white_balance']:
                option = getattr(rs.option, option_name)
                if sensor.supports(option):
                    options[option_name] = sensor.get_option(option)
            sensor_options.append(options)
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() == rs.stream.color:
                    vp = profile.as_video_stream_profile()
                    profiles.append({'width': vp.width(), 'height': vp.height(), 'fps': vp.fps(), 'format': str(vp.format())})
        info['color_profiles'] = profiles
        info['observed_sensor_options'] = sensor_options
        report['devices'].append(info)
        if 'D455' in info.get('name', ''):
            targets.append((device, info))
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / 'camera_status.json'
    status_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not args.capture:
        print(json.dumps(report, indent=2))
        return
    if len(targets) != 1:
        raise SystemExit('Exactly one D455 is required; no camera stream was started')
    if args.board_id == 'unassigned' or args.session_id == 'unassigned':
        raise SystemExit('Set --board-id and --session-id before capture to preserve split groups')
    import cv2
    import numpy as np
    from pcb_components.quality import analyze_image_quality
    device, info = targets[0]
    wanted = {'width': args.width, 'height': args.height, 'fps': args.fps, 'format': str(rs.format.bgr8)}
    if wanted not in info['color_profiles']:
        raise SystemExit('Requested color profile unsupported; choose one listed by inventory')
    pipe = rs.pipeline(context)
    cfg = rs.config()
    cfg.enable_device(info['serial_number'])
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    report['capture_attempted'] = True
    records = []
    try:
        active = pipe.start(cfg)
        intr = active.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        report['intrinsics'] = {'width': intr.width, 'height': intr.height, 'fx': intr.fx, 'fy': intr.fy,
                                'ppx': intr.ppx, 'ppy': intr.ppy, 'model': str(intr.model), 'coeffs': intr.coeffs}
        previous = -1
        # Initial frames are acquisition warmup, not training images.
        for _ in range(15):
            pipe.wait_for_frames(3000)
        for index in range(args.count):
            frame = pipe.wait_for_frames(3000).get_color_frame()
            number = frame.get_frame_number()
            raw = np.asanyarray(frame.get_data())
            if number <= previous or raw.size == 0 or float(raw.std()) < 1.0:
                raise RuntimeError('Non-increasing frame number or near-uniform pixels; acquisition not verified')
            previous = number
            name = f'{args.board_id}_{args.session_id}_{index:04d}.png'
            dest = args.output / name
            if dest.exists():
                raise FileExistsError(dest)
            ok, encoded = cv2.imencode('.png', raw)
            if not ok:
                raise RuntimeError(f'Could not save {dest}')
            encoded.tofile(dest)
            metadata = {}
            for key in ['actual_exposure', 'gain_level', 'white_balance', 'actual_fps']:
                enum = getattr(rs.frame_metadata_value, key, None)
                metadata[key] = frame.get_frame_metadata(enum) if enum is not None and frame.supports_frame_metadata(enum) else None
            records.append({'file': name, 'frame_number': number, 'timestamp_ms': frame.get_timestamp(),
                            'board_id': args.board_id, 'session_id': args.session_id,
                            'physical_board_id': args.board_id, 'board_design_id': args.design_id,
                            'board_side': args.side, 'capture_session_id': args.session_id,
                            'distance_mm_manually_measured': args.distance_mm, 'lighting_id': args.lighting_id,
                            'native_width': int(raw.shape[1]), 'native_height': int(raw.shape[0]),
                            'image_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
                            'frame_metadata_raw_sdk_units': metadata,
                            'quality': analyze_image_quality(raw).to_dict(),
                            'quality_is_not_an_acceptance_gate': True,
                            'pixel_std': float(raw.std()), 'utc': datetime.now(timezone.utc).isoformat()})
            time.sleep(args.interval)
        report['status'] = 'RAW_CAPTURE_SAVED_NOT_ANNOTATED'
    except Exception as error:
        report['status'] = 'CAPTURE_FAILED'
        report['error'] = str(error)
        raise
    finally:
        try:
            pipe.stop()
        except RuntimeError:
            pass
        report['frames_saved'] = len(records)
        report['frames'] = records
        status_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
