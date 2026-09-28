# Post-test audit adapter v1.1: original analysis plan/source remain unchanged.
# Only prediction geometry admissibility changes: clipped zero-area detections
# are allowed and retained. Thresholds, matching and AP implementation unchanged.
"""CPU-only final evidence audit and predeclared fixed-confidence diagnostics.

Reads existing local checkpoints/predictions; never predicts, trains or changes
training, splits, primary evaluation metrics, or selected checkpoint.
"""
from pathlib import Path
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime,timezone
import argparse,copy,hashlib,io,json,math,os
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def stamp(s):
    d=datetime.fromisoformat(s)
    if d.utcoffset() is None:raise ValueError('Timezone missing')
    return d
def near(a,b,tolerance=1e-10):return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tolerance

def iou(box,boxes):
    b=np.array(box,dtype=float);g=np.array(boxes,dtype=float).reshape(-1,4)
    inter=np.maximum(0,np.minimum(b[2:],g[:,2:])-np.maximum(b[:2],g[:,:2])).prod(axis=1)
    union=np.maximum(0,b[2:]-b[:2]).prod()+np.maximum(0,g[:,2:]-g[:,:2]).prod(axis=1)-inter
    return np.divide(inter,union,out=np.zeros(len(g)),where=union>0)
def operating(samples,names,confidence=.25,iou_threshold=.5):
    counts=np.zeros((len(names),3),dtype=np.int64);per_image=[]
    for s in samples:
        truth=s['truth'];found=np.zeros(len(truth),dtype=bool);local=np.zeros_like(counts)
        for p in sorted(s['predictions'],key=lambda p:(-p['score'],p['json_index'])):
            if p['score']<confidence:break
            overlaps=iou(p['bbox_xyxy'],[t['bbox_xyxy'] for t in truth]);cid=p['class_id']
            choices=[i for i,t in enumerate(truth) if t['class_id']==cid and not found[i] and overlaps[i]>=iou_threshold]
            if choices:
                gt=min(choices,key=lambda i:(-float(overlaps[i]),i));found[gt]=True;local[cid,0]+=1
            else:local[cid,1]+=1
        for t,ok in zip(truth,found):
            if not ok:local[t['class_id'],2]+=1
        counts+=local;per_image.append({'id':s['id'],'counts_tp_fp_fn':local.tolist()})
    def fields(c):
        tp,fp,fn=map(int,c)
        return {'tp':tp,'fp':fp,'fn':fn,'gt':tp+fn,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
    return {'confidence':confidence,'iou_threshold':iou_threshold,'definition':'Fixed descriptive point, not selected on validation or test; rounded saved predictions','micro':fields(counts.sum(axis=0)),'per_class':{n:fields(counts[i]) for i,n in enumerate(names)},'per_image':per_image}

def coco_reference(samples,names):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    dataset={'images':[],'categories':[{'id':i+1,'name':n} for i,n in enumerate(names)],'annotations':[]};preds=[]
    for index,s in enumerate(samples,1):
        dataset['images'].append({'id':index,'width':s['width'],'height':s['height']})
        for t in s['truth']:
            x1,y1,x2,y2=t['bbox_xyxy'];w=x2-x1;h=y2-y1
            dataset['annotations'].append({'id':len(dataset['annotations'])+1,'image_id':index,'category_id':t['class_id']+1,'bbox':[x1,y1,w,h],'area':w*h,'iscrowd':0})
        for p in s['predictions']:
            x1,y1,x2,y2=p['bbox_xyxy'];preds.append({'image_id':index,'category_id':p['class_id']+1,'bbox':[x1,y1,x2-x1,y2-y1],'score':p['score']})
    with redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=dataset;gt.createIndex()
        if preds:dt=gt.loadRes(preds)
        else:dt=COCO();dt.dataset={'images':dataset['images'],'categories':dataset['categories'],'annotations':[]};dt.createIndex()
        calc=COCOeval(gt,dt,'bbox');calc.params.maxDets=[1,100,300];calc.evaluate();calc.accumulate()
    precision=calc.eval['precision']
    def avg(a):
        vals=a[a>=0];return float(vals.mean()) if vals.size else None
    return {'definition':'Independent pycocotools COCO AP on rounded saved predictions; global300 cap precedes per-class100/300; not the primary Ultralytics implementation','ap50_95_max100':avg(precision[:,:,:,0,1]),'ap50_95_max300':avg(precision[:,:,:,0,2]),'ap50_max300':avg(precision[0,:,:,0,2]),'per_class':{n:{'ap50_95_max300':avg(precision[:,:,i,0,2]),'ap50_max300':avg(precision[0,:,i,0,2])} for i,n in enumerate(names)}}

def expected_calls(config,nb):
    nw=round(min(config['warmup_epochs'],max(config['epochs']-1,0))*nb);last=-1;calls=0;accumulate=max(round(config['nbs']/config['batch']),1);rows=[]
    for epoch in range(config['epochs']):
        for i in range(nb):
            ni=epoch*nb+i
            if ni<nw:accumulate=max(1,int(np.interp(ni,[0,nw],[1,config['nbs']/config['batch']]).round()))
            if ni-last>=accumulate:calls+=1;last=ni
        rows.append(calls)
    return rows

def run():
    report={'schema':'wafer-r01-independent-final-audit-v1','status':'RUNNING','created_utc':now(),'checks':[],'failures':[],'input_sha256':{}}
    def check(c,text):
        if not c:raise ValueError(text)
        report['checks'].append(text)
    def load(p):
        report['input_sha256'][str(p)]=sha(p);return read(p)
    try:
        amendment=load(ROOT/'reports/post_test_audit_amendment_v1_1.json')
        check(sha(__file__)==amendment['adapter_sha256'],'Executed adapter SHA matches disclosed post-test amendment')
        report['post_test_amendment']=amendment
        report['pretest_plan_unchanged']=True
        report['executed_code_was_amended_after_test']=True
        plan=load(ROOT/'reports/fixed_conf_analysis_plan.json');binding=load(ROOT/'reports/fixed_conf_analysis_code_binding.json')
        check(binding['script_sha256']==sha(ROOT/'scripts/audit_final_results.py') and binding['plan_sha256']==sha(ROOT/'reports/fixed_conf_analysis_plan.json'),'Original pre-test source and plan preserved; executed post-test geometry adapter disclosed')
        check((ROOT/'reports/fixed_conf_analysis_plan.sha256').read_text().split()[0]==binding['plan_sha256'],'Analysis plan sidecar hash matches')
        protocol=load(ROOT/'protocol.json');manifest=load(ROOT/'data/manifest.json');data_audit=load(ROOT/'reports/data_audit.json');independent=load(ROOT/'reports/independent_data_audit.json')
        for key,path in [('protocol_sha256',ROOT/'protocol.json'),('manifest_sha256',ROOT/'data/manifest.json'),('data_audit_sha256',ROOT/'reports/data_audit.json'),('train_pilot_sha256',ROOT/'scripts/train_pilot.py')]:check(sha(path)==plan['bindings'][key],f'Frozen {key} preserved')
        check(independent['status']=='PASS_INDEPENDENT_DATA_AUDIT','Independent data audit passed')
        for row in data_audit['files']:
            if sha(row['path'])!=row['sha256']:raise ValueError('Frozen data changed: '+row['path'])
        check(True,f'All{len(data_audit["files"])} frozen data file hashes preserved')
        check(protocol['dataset_audit_sha256']==sha(ROOT/'reports/data_audit.json'),'Training bound to frozen dataset audit')
        config=protocol['configuration'];names=manifest['names']
        check(names==plan['names'] and len(names)==6,'Six class IDs match analysis plan')
        check(config['epochs']==10 and config['batch']==4 and config['amp'] is False and config['cls_pw']==0,'Authorized ten-epoch FP32 batch4 policy')
        start=load(ROOT/'reports/training_start.json');summary=load(ROOT/'reports/training_summary.json');epochs=load(ROOT/'reports/epochs.json');exposures=load(ROOT/'reports/image_exposures.json')
        check(start['script_sha256']==sha(ROOT/'scripts/train_pilot.py'),'Executed training source equals frozen source')
        check(summary['status']=='TRAINING_COMPLETE' and summary['epochs_completed']==10 and [r['epoch'] for r in epochs]==list(range(1,11)),'Exactly ten real epoch callbacks; no final-validation pseudo epoch')
        nb=start['train_batches_per_epoch'];check(nb==368 and start['actual_batch']==4,'Actual train batches368 and batch4')
        schedule=expected_calls(config,nb)
        for row,calls in zip(epochs,schedule):
            check(row['batches_cumulative']==row['epoch']*nb and row['optimizer_calls']==row['ema_updates']==calls,f'Epoch{row["epoch"]}: observed batch/optimizer/EMA counts match schedule')
            check(all(math.isfinite(v) for v in row['validation'].values()),f'Epoch{row["epoch"]}: finite validation metrics')
        check(summary['direct_optimizer_calls']==summary['ema_updates']==schedule[-1]>0 and summary['batches']==10*nb,'Final observed direct calls/EMA/batch budget match')
        check(0<summary['parameter_step_min']<=summary['parameter_step_max']<=summary['direct_optimizer_calls'],'Parameter optimizer state counters independently bounded')
        check(start['initial_parameter_sha256']==summary['initial_parameter_sha256']!=summary['final_parameter_sha256'],'Measured live model parameters changed')
        train=[r for r in manifest['records'] if r['split']=='train'];test=[r for r in manifest['records'] if r['split']=='test']
        check(set(exposures)=={str(Path(r['prepared_image']).resolve()) for r in train} and set(exposures.values())=={10},'Each of1472 training images seen exactly10 times; no val/test image seen')
        check(summary['image_exposures']==sum(exposures.values())==14720 and summary['unique_training_images_seen']==1472,'Repeated image exposures and unique count separated')
        transformed=summary['transformed_class_exposures'];check(set(transformed)=={str(i) for i in range(6)} and all(v>0 for v in transformed.values()),'Actual transformed label exposures recorded for all6 classes')
        check(len(test)==287 and sum(len(r['objects']) for r in test)==496,'Held-out cohort exactly287 images/496 GT boxes')
        selection=load(ROOT/'reports/selection_frozen.json');test_start=load(ROOT/'reports/test_started.json');metrics=load(ROOT/'reports/test_metrics.json')
        check(selection['test_results_seen'] is False and selection['selected_on']=='validation only','Selection declared validation only')
        check(selection['protocol_sha256']==sha(ROOT/'protocol.json') and selection['data_audit_sha256']==sha(ROOT/'reports/data_audit.json'),'Selection binds training and data protocol')
        check(stamp(summary['finished_utc'])<=stamp(selection['frozen_utc'])<stamp(test_start['utc'])<stamp(metrics['finished_utc']),'Training finish and checkpoint freeze precede test start and completion')
        check(stamp(plan['frozen_utc'])<stamp(test_start['utc']) and stamp(binding['frozen_utc'])<stamp(test_start['utc']),'Supplemental plan/code frozen before any test results')
        check(test_start['selection_sha256']==metrics['selection_sha256']==sha(ROOT/'reports/selection_frozen.json'),'Test start and results use frozen selection SHA')
        best=Path(selection['checkpoint']);check(best==Path(summary['best_checkpoint']) and sha(best)==selection['sha256']==summary['best_checkpoint_sha256']==metrics['checkpoint_sha256'],'Checkpoint SHA unchanged across training summary/selection/test')
        import torch
        torch.set_num_threads(2)
        checkpoint=torch.load(best,map_location='cpu',weights_only=False);model=checkpoint['model']
        check(type(model).__name__=='DetectionModel' and [model.names[i] for i in range(6)]==names,'Portable six-class DetectionModel checkpoint')
        check(all(torch.isfinite(t).all() for t in model.state_dict().values() if torch.is_tensor(t)),'All saved model weights/buffers finite')
        best_map=max(r['validation']['metrics/mAP50-95(B)'] for r in epochs)
        check(near(checkpoint['train_metrics']['metrics/mAP50-95(B)'],best_map),'Best checkpoint stored validation AP equals max recorded epoch AP')
        check(metrics['d455_verified'] is False and metrics['instance_masks_trained'] is False,'Primary result scope is bbox-only/public-image/not-D455')
        predictions=load(ROOT/'evaluation/test/predictions.json');byid={r['id']:r for r in test};grouped={sid:[] for sid in byid}
        check(all(p['image_id'] in byid for p in predictions),'Saved predictions contain test images only')
        for index,p in enumerate(predictions):
            sid=p['image_id'];r=byid[sid];cid=p['category_id']-1;x,y,w,h=p['bbox'];score=p['score']
            check(0<=cid<6 and isinstance(p['category_id'],int),f'Prediction{index}: canonical JSON class index')
            if not all(math.isfinite(float(v)) for v in [x,y,w,h,score]) or w<0 or h<0 or min(x,y)<-.002 or x+w>r['width']+.002 or y+h>r['height']+.002 or not .00099<=score<=1:raise ValueError('Prediction coordinates/score invalid')
            if p.get('file_name')!=Path(r['prepared_image']).name:raise ValueError('Prediction image/file binding mismatch')
            grouped[sid].append({'class_id':cid,'bbox_xyxy':[x,y,x+w,y+h],'score':score,'json_index':index})
        check(all(len(p)<=300 for p in grouped.values()),'Saved predictions respect global300 detections/image cap')
        report['zero_area_predictions_retained']={'all_saved':sum(p['bbox'][2]==0 or p['bbox'][3]==0 for p in predictions),'at_fixed_confidence':sum((p['bbox'][2]==0 or p['bbox'][3]==0) and p['score']>=plan['confidence'] for p in predictions),'policy':'All remain in saved AP inputs; IoU is zero; any at fixed confidence count as FP'}
        samples=[{'id':r['id'],'width':r['width'],'height':r['height'],'truth':[{'class_id':names.index(o['class']),'bbox_xyxy':o['bbox_xyxy']} for o in r['objects']],'predictions':grouped[r['id']]} for r in test]
        op=operating(samples,names,plan['confidence'],plan['iou']);reference=coco_reference(samples,names)
        check(op['micro']['gt']==496 and sum(v['gt'] for v in op['per_class'].values())==496,'Fixed point class GT denominators sum to496')
        report.update(status='PASS_INDEPENDENT_FINAL_AUDIT_NOT_D455',training={'epochs':10,'direct_optimizer_calls':summary['direct_optimizer_calls'],'ema_updates':summary['ema_updates'],'image_exposures':14720,'unique_images':1472,'transformed_class_exposures':transformed,'best_validation_map50_95':best_map,'epochs_attaining_best':[r['epoch'] for r in epochs if r['validation']['metrics/mAP50-95(B)']==best_map]},holdout={'images':287,'gt':496,'predictions':len(predictions),'source_groups':len({r['group_id'] for r in test})},fixed_confidence=op,primary_ultralytics={'bbox_map50':metrics['bbox_map50'],'bbox_map50_95':metrics['bbox_map50_95'],'per_class':metrics['per_class']},independent_coco_reference=reference,reference_minus_primary_map50_95=reference['ap50_95_max300']-metrics['bbox_map50_95'],metric_comparison_note='COCO reference uses different matching/interpolation and rounded JSON. Differences are reported, not treated as a required equality or replacement primary metric.',selection_sha256=sha(ROOT/'reports/selection_frozen.json'),prediction_sha256=sha(ROOT/'evaluation/test/predictions.json'),script_sha256=sha(__file__),limitations=plan['limitations'])
        for p,expected in report['input_sha256'].items():check(sha(p)==expected,'Read-only input preserved: '+str(p))
    except Exception as exc:
        import traceback
        report['status']='FAIL_INDEPENDENT_FINAL_AUDIT';report['failures'].append(f'{type(exc).__name__}: {exc}');traceback.print_exc()
    report['finished_utc']=now()
    out=ROOT/'reports/independent_final_results_audit.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# 웨이퍼 R01 저장 결과 독립 감사','',report['status'],'',f"검사 {len(report['checks'])}개 통과, 실패 {len(report['failures'])}개. CPU로 기록과 저장 예측만 읽었다.",'']
    if 'fixed_confidence' in report:
        lines+=['confidence0.25와 IoU0.5는 test 전에 정한 설명용 운용점이다. val 최적화 기준이나 실사용 합격선이 아니다.','', '| 클래스 | TP | FP | FN | GT | Precision | Recall |','|---|---:|---:|---:|---:|---:|---:|']
        for n,r in {**report['fixed_confidence']['per_class'],'micro':report['fixed_confidence']['micro']}.items():lines.append(f"| {n} | {r['tp']} | {r['fp']} | {r['fn']} | {r['gt']} | {r['precision']*100:.2f}% | {r['recall']*100:.2f}% |")
        lines+=['',f"원래 Ultralytics mAP50–95: {report['primary_ultralytics']['bbox_map50_95']*100:.3f}%. 별도 COCO 구현 참고 mAP50–95: {report['independent_coco_reference']['ap50_95_max300']*100:.3f}%.",'두 구현은 matching·보간·저장 JSON의 좌표/점수 반올림이 달라 정확한 일치를 요구하지 않는다. 원래 결과나 선택을 수정하지 않았다.']
    lines+=['','사후 감사 코드 정정 v1.1: 첫 감사는 native 좌표가 0면적인 검출을 거부했다. 라이브러리의 경계 clipping/JSON 저장에서 생길 수 있어 크기0을 허용하도록 입력 검사를 수정했다. 원본 계획·코드·binding·실패 보고서는 보존했다. 임계값·매칭·AP 계산은 그대로이며 예측을 삭제하지 않았다. 이 어댑터를 test 전 동결 코드라고 표현하지 않는다.']
    lines+=['']+report['failures']+['','D455 실사용·물리적 웨이퍼/lot 독립성·mask 학습은 검증하지 않았다.']
    (ROOT/'reports/independent_final_results_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'checks':len(report['checks']),'failures':report['failures']}))
    if report['failures']:raise SystemExit(1)

def self_test():
    import unittest
    class Tests(unittest.TestCase):
        def pred(self,box=[0,0,10,10],score=.9,cid=0,index=0):return {'bbox_xyxy':box,'score':score,'class_id':cid,'json_index':index}
        def score(self,truth,preds):return operating([{'id':'x','truth':truth,'predictions':preds}],['a','b'])
        def test_duplicates_and_class_confusion(self):
            truth=[{'class_id':0,'bbox_xyxy':[0,0,10,10]}];r=self.score(truth,[self.pred(index=0),self.pred(score=.8,index=1),self.pred(score=.7,cid=1,index=2)])
            self.assertEqual([r['micro'][k] for k in ['tp','fp','fn']],[1,2,0])
        def test_exact_threshold_and_low_score(self):
            t=[{'class_id':0,'bbox_xyxy':[0,0,10,10]}]
            self.assertEqual(self.score(t,[self.pred(score=.25)])['micro']['tp'],1)
            self.assertEqual(self.score(t,[self.pred(score=.24999)])['micro']['tp'],0)
        def test_exact_iou_boundary(self):
            t=[{'class_id':0,'bbox_xyxy':[0,0,10,10]}]
            self.assertEqual(self.score(t,[self.pred(box=[0,0,20,10])])['micro']['tp'],1)
        def test_empty_class_and_image(self):
            r=self.score([],[]);self.assertEqual(r['micro']['tp'],0);self.assertIsNone(r['per_class']['b']['recall'])
        def test_wrong_class_false_negative(self):
            t=[{'class_id':0,'bbox_xyxy':[0,0,10,10]}];r=self.score(t,[self.pred(cid=1)])
            self.assertEqual([r['micro'][k] for k in ['tp','fp','fn']],[0,1,1])
        def test_exact_iou_tie_uses_lower_source_index(self):
            t=[{'class_id':0,'bbox_xyxy':[0,0,10,10]},{'class_id':0,'bbox_xyxy':[10,0,20,10]}]
            r=self.score(t,[self.pred(box=[0,0,10,10],score=.8,index=1),self.pred(box=[0,0,20,10],score=.9,index=0)])
            self.assertEqual([r['micro'][k] for k in ['tp','fp','fn']],[1,1,1])
        def test_training_step_schedule(self):
            s=expected_calls({'warmup_epochs':1,'epochs':10,'nbs':8,'batch':4},368)
            self.assertEqual((s[0],s[-1]),(276,1932))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');args=p.parse_args()
    self_test() if args.self_test else run()
