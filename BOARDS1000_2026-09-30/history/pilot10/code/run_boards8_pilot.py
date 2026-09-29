from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
import sys
import time
import traceback

TRIAL=Path(__file__).resolve().parent/'boards8_pilot_20260929'
os.environ['YOLO_CONFIG_DIR']=str(TRIAL/'ultralytics_config')
os.environ['YOLO_OFFLINE']='true'
os.environ['WANDB_MODE']='disabled'
os.environ['COMET_MODE']='DISABLED'
sys.path.insert(0,str(TRIAL))
from common import ROOT,DATASETS,MODEL_CLASSES,sha256,write_json
from verify_datasets import dataset_fingerprint,holdout_inventory
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

SETTINGS.update({k:False for k in ['sync','clearml','comet','dvc','mlflow','neptune','raytune','tensorboard','wandb'] if k in SETTINGS})
WEIGHTS=Path(r'C:\Users\hkjun\Documents\Codex\2026-09-27\new-chat\work\r03\models\yolo11s.pt')
ATTEMPT=int(os.environ.get('BOARDS_PILOT_ATTEMPT','1'))
SUFFIX='' if ATTEMPT==1 else f'_attempt{ATTEMPT}'
RUN=TRIAL/f'runs/boards8_yolo11s_10ep{SUFFIX}'
STATE=TRIAL/f'execution_state{SUFFIX}.json'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
utc=lambda:datetime.now(timezone.utc).isoformat()
config=dict(epochs=10,imgsz=640,batch=8,device=0,optimizer='AdamW',lr0=0.001,lrf=0.05,cos_lr=True,weight_decay=0.0005,warmup_epochs=1.0,amp=False,seed=20260929,deterministic=True,workers=2,nbs=64,patience=10,degrees=10,fliplr=0.5,mosaic=1.0,close_mosaic=3,scale=0.5,translate=0.1,hsv_h=0.015,hsv_s=0.4,hsv_v=0.3,mixup=0.0,plots=True,save=True,save_period=-1,cache=False,verbose=False)

def per_class(metrics,split,verified):
    by_index={int(cid):i for i,cid in enumerate(metrics.box.ap_class_index)}
    result=[]
    for cid,name in enumerate(MODEL_CLASSES['boards']):
        i=by_index.get(cid)
        result.append({'class_id':cid,'class_name':name,'ground_truth_boxes':verified['boxes'][split][name],
            'map50':float(metrics.box.ap50[i]) if i is not None else None,
            'map50_95':float(metrics.box.ap[i]) if i is not None else None})
    return result

def main():
    if STATE.exists() or RUN.exists():
        raise SystemExit('Existing run/state preserved. Do not silently retrain or repeat test.')
    verified=read(DATASETS/'verification.json')['boards_v1']
    assert verified['training_ready'] and verified['structure_pass']
    assert verified['dataset_fingerprint']==dataset_fingerprint(DATASETS/'boards_v1')
    assert verified['class_map_sha256']==sha256(ROOT/'class_map.json')
    assert verified['holdout_inventory']==holdout_inventory()
    assert 'stm32_nucleo' not in MODEL_CLASSES['boards'] and len(MODEL_CLASSES['boards'])==8
    assert WEIGHTS.is_file() and torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    state={'status':'training','started_at':utc(),'classes':MODEL_CLASSES['boards'],'scope':'8-class board pilot; Nucleo and all port training excluded','config':config,'initial_weights':str(WEIGHTS),'initial_weights_sha256':sha256(WEIGHTS),'dataset_fingerprint':verified['dataset_fingerprint'],'torch':torch.__version__,'ultralytics':ultralytics.__version__,'gpu':torch.cuda.get_device_name(0),'optimizer_steps':0,'epochs':[],'test_evaluation_executed':False}
    write_json(STATE,state)
    started=time.monotonic()
    model=YOLO(str(WEIGHTS))
    hook=None

    def count_step(optimizer,args,kwargs):
        state['optimizer_steps']+=1

    def train_start(trainer):
        nonlocal hook
        hook=trainer.optimizer.register_step_post_hook(count_step)

    def loss_values(trainer):
        values=trainer.tloss.values() if isinstance(trainer.tloss,dict) else trainer.tloss
        return [float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for v in values]

    def check_batch(trainer):
        if trainer.tloss is not None and not all(math.isfinite(v) for v in loss_values(trainer)):
            raise RuntimeError('Nonfinite training loss; stopped without promoting checkpoint')

    def check_weights(trainer):
        if not all(bool(torch.isfinite(t).all()) for t in trainer.model.state_dict().values() if t.is_floating_point()):
            raise RuntimeError('Nonfinite model state; stopped before checkpoint selection')

    def epoch_end(trainer):
        epoch=trainer.epoch+1
        # final_eval can call this callback with epoch already advanced beyond the loop.
        if epoch>config['epochs'] or any(e['epoch']==epoch for e in state['epochs']):
            return
        row={'epoch':epoch,'elapsed_seconds':time.monotonic()-started,'optimizer_steps':state['optimizer_steps'],'losses':loss_values(trainer),'val_metrics':{k:float(v) for k,v in trainer.metrics.items()},'peak_cuda_memory_gib':torch.cuda.max_memory_allocated()/2**30,'best_fitness':float(trainer.best_fitness),'fitness':float(trainer.fitness)}
        assert all(math.isfinite(x) for x in row['losses'])
        state['epochs'].append(row)
        if epoch==3:
            state['three_epoch_checkpoint']={'finite_losses':True,'finite_parameters':True,'optimizer_steps':state['optimizer_steps'],'continue_to_total_epochs':10}
        write_json(STATE,state)
        print('PILOT_EPOCH '+json.dumps(row),flush=True)

    model.add_callback('on_train_start',train_start)
    model.add_callback('on_train_batch_end',check_batch)
    model.add_callback('on_train_epoch_end',check_weights)
    model.add_callback('on_fit_epoch_end',epoch_end)
    result=model.train(data=str(DATASETS/'boards_v1/data.yaml'),project=str(TRIAL/'runs'),name=RUN.name,exist_ok=False,**config)
    if hook is not None:
        hook.remove()
    best=Path(model.trainer.best)
    assert best.is_file()
    assert model.names==dict(enumerate(MODEL_CLASSES['boards']))
    actual_epochs=list(csv.DictReader((RUN/'results.csv').open(encoding='utf-8-sig')))
    summary={'scope':state['scope'],'training_started_at':state['started_at'],'training_finished_at':utc(),'training_seconds':time.monotonic()-started,'epochs_completed':len(actual_epochs),'actual_optimizer_steps':state['optimizer_steps'],'best':str(best),'best_sha256':sha256(best),'initial_weights_sha256':state['initial_weights_sha256'],'classes':MODEL_CLASSES['boards'],'config':config,'dataset_fingerprint':verified['dataset_fingerprint'],'val_metrics':{k:float(v) for k,v in result.results_dict.items()},'val_per_class':per_class(result,'val',verified),'checkpoint_selection':'Ultralytics best validation fitness only; test untouched until checkpoint hash frozen','training_curve':str(RUN/'results.csv'),'peak_cuda_memory_gib':torch.cuda.max_memory_allocated()/2**30}
    write_json(RUN/'training-summary.json',summary)
    state.update({'status':'checkpoint_frozen','best_sha256':summary['best_sha256'],'training_finished_at':summary['training_finished_at']})
    write_json(STATE,state)
    # Exactly one held-out test invocation after checkpoint selection and integrity checks.
    assert verified['dataset_fingerprint']==dataset_fingerprint(DATASETS/'boards_v1')
    assert summary['best_sha256']==sha256(best)
    marker=RUN/'test-evaluation-attempt.json'
    with marker.open('x',encoding='utf-8') as f:
        json.dump({'started_at':utc(),'checkpoint_sha256':summary['best_sha256'],'split':'test'},f)
    state['status']='test_evaluation'
    write_json(STATE,state)
    metrics=YOLO(str(best)).val(data=str(DATASETS/'boards_v1/data.yaml'),split='test',imgsz=640,batch=1,device=0,conf=0.001,iou=0.7,max_det=300,half=False,plots=True,project=str(RUN),name='test',exist_ok=False,verbose=False)
    test={'checkpoint_sha256':summary['best_sha256'],'completed_at':utc(),'images':verified['images']['test'],'split':'test','invocations':1,'overall':{k:float(v) for k,v in metrics.results_dict.items()},'per_class':per_class(metrics,'test',verified),'scope':'Held-out groups of public datasets; statistical device/session independence and D455 real-camera accuracy not established. Nucleo absent from this model.'}
    write_json(RUN/'test-evaluation.json',test)
    state.update({'status':'complete','finished_at':utc(),'test_evaluation_executed':True,'test_invocations':1})
    write_json(STATE,state)
    print('PILOT_COMPLETE '+json.dumps({'run':str(RUN),'best_sha256':summary['best_sha256'],'optimizer_steps':state['optimizer_steps'],'test':test['overall']}),flush=True)

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        failure=read(STATE) if STATE.exists() else {}
        failure.update({'status':'failed','failed_at':utc(),'error':repr(exc)})
        write_json(TRIAL/f'failure{SUFFIX}.json',failure)
        traceback.print_exc()
        raise
