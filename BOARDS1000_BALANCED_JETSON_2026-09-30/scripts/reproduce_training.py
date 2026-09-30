"""Portable NEW training launcher; default action only checks snapshot bytes."""
from pathlib import Path
import argparse, json, os, re, tempfile
from verify_snapshot import verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--action',choices=['check','train'],default='check');p.add_argument('--output',type=Path);p.add_argument('--name',default='balanced_reproduction_01');a=p.parse_args()
    root=a.root.resolve();print(json.dumps(verify(root),indent=2))
    if a.action=='check':return
    assert a.output is not None,'Specify an external --output folder for a NEW training run'
    out=a.output.resolve();assert not out.is_relative_to(root),'Preserve the frozen snapshot'
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',a.name),'Use a simple run name'
    assert not (out/a.name).exists(),'Existing run preserved; choose a new name'
    cfg=json.loads((root/'configs/train.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='balanced_yolo_settings_') as settings:
        os.environ['YOLO_CONFIG_DIR']=settings;os.environ['YOLO_OFFLINE']='true';os.environ['WANDB_MODE']='disabled';os.environ['COMET_MODE']='DISABLED'
        import torch,yaml
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
        SETTINGS.update({k:False for k in ['sync','clearml','comet','dvc','mlflow','neptune','raytune','tensorboard','wandb'] if k in SETTINGS})
        assert torch.cuda.is_available(),'Archived training configuration requires CUDA';torch.set_num_threads(4)
        classes=json.loads((root/'configs/classes.json').read_text(encoding='utf-8'))
        data={'path':(root/'datasets/boards_v1').as_posix(),'train':'images/train','val':'images/val','test':'images/test','names':dict(enumerate(classes))}
        out.mkdir(parents=True,exist_ok=True);yaml_path=out/(a.name+'_data.yaml')
        with yaml_path.open('x',encoding='utf-8') as f:yaml.safe_dump(data,f,sort_keys=False)
        model=YOLO(str(root/'weights/original-baseline-best.pt'))
        model.train(data=str(yaml_path),project=str(out),name=a.name,exist_ok=False,**cfg)
        print('NEW training/validation finished. Test inference was not invoked. Archived selection/test evidence was not overwritten.')

if __name__=='__main__':main()
