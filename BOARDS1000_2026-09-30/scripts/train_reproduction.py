"""Portable launcher for a NEW validation-only run; defaults to checking the bundle."""
from pathlib import Path
import argparse
import json
import os
import re
import tempfile
from verify_bundle import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--action', choices=['check', 'train'], default='check')
    parser.add_argument('--name', default='boards1000_reproduction')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.name), 'Use a simple run name'
    print(json.dumps(verify(root), indent=2))
    if args.action == 'train' and args.output is None:
        parser.error('--output is required for a new training run')
    output = args.output.resolve() if args.output else None
    if output:
        assert not output.is_relative_to(root), 'Keep new runs outside the frozen snapshot'
        assert not (output / args.name).exists(), 'Existing run is preserved; choose a new name'
    config = json.loads((root / 'configs/train.json').read_text(encoding='utf-8'))
    # Isolate library settings; no account settings, integrations, or credentials are loaded.
    with tempfile.TemporaryDirectory(prefix='boards1000_config_') as config_dir:
        os.environ['YOLO_CONFIG_DIR'] = config_dir
        os.environ['YOLO_OFFLINE'] = 'true'
        os.environ['WANDB_MODE'] = 'disabled'
        os.environ['COMET_MODE'] = 'DISABLED'
        import torch
        import yaml
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
        SETTINGS.update({k: False for k in ['sync', 'clearml', 'comet', 'dvc', 'mlflow', 'neptune', 'raytune', 'tensorboard', 'wandb'] if k in SETTINGS})
        assert torch.cuda.is_available(), 'The archived configuration requires a CUDA GPU'
        torch.set_num_threads(4)
        model = YOLO(str(root / 'weights/initial_yolo11s.pt'))
        print(json.dumps({'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'action': args.action}))
        if args.action == 'check':
            return
        names = json.loads((root / 'configs/classes.json').read_text(encoding='utf-8'))
        data = {'path': (root / 'datasets/boards_v1').as_posix(), 'train': 'images/train', 'val': 'images/val', 'test': 'images/test', 'names': dict(enumerate(names))}
        output.mkdir(parents=True, exist_ok=True)
        data_path = output / (args.name + '_data.yaml')
        with data_path.open('x', encoding='utf-8') as f:
            yaml.safe_dump(data, f, sort_keys=False)
        model.train(data=str(data_path), project=str(output), name=args.name, exist_ok=False, **config)
        print('New training and validation finished. No test inference was invoked.')


if __name__ == '__main__':
    main()
