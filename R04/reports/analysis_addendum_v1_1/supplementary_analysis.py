"""Frozen-definition CPU diagnostics; saved predictions only, never reselection."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from contextlib import redirect_stdout
import argparse, copy, hashlib, importlib.util, io, json, os
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent/'r04'; SUITE=ROOT/'reports/evaluation_suite'
OUT=HERE/'reports'; NAMES=['resistor','capacitor','ic','connector']
os.environ['CUDA_VISIBLE_DEVICES']=''
spec=importlib.util.spec_from_file_location('frozen_audit',ROOT/'scripts/audit_r04_results.py')
audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
read=audit.read; sha=audit.sha

def now(): return datetime.now(timezone.utc).isoformat()
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def average(values):
    values=[v for v in values if v is not None]
    return sum(values)/len(values) if values else None

class CocoSubsets:
    """Compute per-image matching once, then official COCO accumulation by group."""
    def __init__(self,gt_data,pred_data):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        with redirect_stdout(io.StringIO()):
            gt=COCO();gt.dataset=copy.deepcopy(gt_data);gt.createIndex()
            if pred_data: pred=gt.loadRes(copy.deepcopy(pred_data))
            else:
                pred=COCO();pred.dataset={'images':gt_data['images'],'categories':gt_data['categories'],'annotations':[]};pred.createIndex()
            self.calc=COCOeval(gt,pred,'bbox');self.calc.params.maxDets=[1,100,300];self.calc.evaluate()
        self.gt_data=gt_data; self.pred_data=pred_data
        self.original_eval_imgs=list(self.calc.evalImgs)
        self.original_params=copy.deepcopy(self.calc._paramsEval)
    def subset(self,ids):
        params=copy.deepcopy(self.original_params);params.imgIds=sorted(ids)
        # COCO accumulate indexes supplied image positions, not original IDs.
        # Rebuild the flat category/area/image layout for the exact subset.
        original_ids=self.original_params.imgIds
        indices=[original_ids.index(i) for i in params.imgIds]
        ni=len(original_ids);na=len(self.original_params.areaRng)
        self.calc.evalImgs=[self.original_eval_imgs[k*na*ni+a*ni+i] for k in range(len(params.catIds)) for a in range(na) for i in indices]
        self.calc._paramsEval=copy.deepcopy(params)
        with redirect_stdout(io.StringIO()): self.calc.accumulate(params)
        precision=self.calc.eval['precision']
        def avg(arr):
            values=arr[arr>=0];return float(values.mean()) if len(values) else None
        result={}
        for maximum,slot in [(100,1),(300,2)]:
            result[f'ap50_95_max{maximum}']=avg(precision[:,:,:,0,slot])
            result[f'ap50_max{maximum}']=avg(precision[0,:,:,0,slot])
            result[f'ap75_max{maximum}']=avg(precision[5,:,:,0,slot])
        result['per_class']={name:{f'ap50_95_max{maximum}':avg(precision[:,:,cls,0,slot]) for maximum,slot in [(100,1),(300,2)]} for cls,name in enumerate(NAMES)}
        return result
    def verify_subset(self,ids):
        gt=copy.deepcopy(self.gt_data); gt['images']=[r for r in gt['images'] if r['id'] in ids];gt['annotations']=[r for r in gt['annotations'] if r['image_id'] in ids]
        pred=[p for p in self.pred_data if p['image_id'] in ids]
        expected=audit.independent_coco(gt,pred);actual=self.subset(ids)
        compare_bbox(actual,expected)

def compare_bbox(actual,expected):
    for key,value in actual.items():
        if key=='per_class':
            for cls in NAMES:
                for name,v in value[cls].items():
                    if not audit.near(v,expected[key][cls][name]): raise ValueError(f'Class AP parity failed {cls}/{name}')
        elif not audit.near(value,expected[key]): raise ValueError(f'Pooled AP parity failed {key}: {value} vs {expected[key]}')

def geometry(box,boxes):
    b=np.array(box,dtype=float); a=np.array(boxes,dtype=float).reshape(-1,4)
    inter=np.maximum(0,np.minimum(b[2:],a[:,2:])-np.maximum(b[:2],a[:,:2])).prod(axis=1)
    pa=float(np.maximum(0,b[2:]-b[:2]).prod());ea=np.maximum(0,a[:,2:]-a[:,:2]).prod(axis=1)
    union=pa+ea-inter
    return np.divide(inter,union,out=np.zeros(len(a)),where=union>0),np.divide(inter,pa,out=np.zeros(len(a)),where=pa>0),np.divide(inter,ea,out=np.zeros(len(a)),where=ea>0)

PASSIVE={'resistor network','resistor jumper','capacitor jumper','ferrite bead','inductor','fuse','emi filter','potentiometer'}
def subtotal(typ):
    if typ in PASSIVE:return 'passive_lookalike'
    if typ in {'pins','pads'}:return 'pins_pads'
    if typ in {'text','component text'}:return 'text'
    return 'rest'

def classify(pred,truth,found,excluded):
    ious,_,_=geometry(pred['bbox_xyxy'],[t['bbox_xyxy'] for t in truth])
    same=[i for i,t in enumerate(truth) if t['class_id']==pred['class_id']]
    diff=[i for i,t in enumerate(truth) if t['class_id']!=pred['class_id']]
    si=min(same,key=lambda i:(-ious[i],i)) if same else None
    di=min(diff,key=lambda i:(-ious[i],i)) if diff else None
    ei,ep,ee=geometry(pred['bbox_xyxy'],[e['bbox_xyxy'] for e in excluded])
    records={}
    for variant,overlap in [('strict_iou',ei),('sensitivity_ioa_pred',ep)]:
        if excluded:
            order=lambda i:(-float(overlap[i]),-float(ei[i]) if variant=='sensitivity_ioa_pred' else 0,excluded[i]['source_type']!='unknown',excluded[i]['source_object_index'])
            ex=min(range(len(excluded)),key=order)
        else:ex=None
        same_rel=si is not None and ious[si]>=.1; confusion=di is not None and ious[di]>=.5
        unknown=any(float(overlap[i])>=.5 and e['source_type']=='unknown' for i,e in enumerate(excluded))
        known=any(float(overlap[i])>=.5 and e['source_type']!='unknown' for i,e in enumerate(excluded))
        detail={};category='other_background'
        if same_rel:
            category='other_same_class';detail={'kind':'duplicate' if ious[si]>=.5 and found[si] else 'localization','gt_index':si,'iou':float(ious[si])}
        elif confusion:
            category='other_class_confusion';detail={'gt_index':di,'gt_class':truth[di]['class_id'],'iou':float(ious[di])}
        elif ex is not None and overlap[ex]>=.5:
            e=excluded[ex];category='source_unknown' if e['source_type']=='unknown' else 'known_non_target'
            detail={'source_type':e['source_type'],'source_object_index':e['source_object_index'],'iou':float(ei[ex]),'ioa_pred':float(ep[ex]),'ioa_excl':float(ee[ex]),'subtotal':subtotal(e['source_type'])}
        records[variant]={'category':category,'detail':detail,'conditions':{'same_class_iou_ge_010':bool(same_rel),'other_class_iou_ge_050':bool(confusion),'unknown_overlap_ge_050':unknown,'non_target_overlap_ge_050':known}}
    return records

def fp_diagnostics(samples,manifest_by_id,metric):
    rows=[];tp=0;fn=0;threshold=metric['operating']['confidence']
    for sample in samples:
        truth=sample['truth'];found=np.zeros(len(truth),dtype=bool)
        for index,pred in sorted(enumerate(sample['predictions']),key=lambda v:-v[1]['score']):
            if pred['score']<threshold:break
            ious,_,_=geometry(pred['bbox_xyxy'],[t['bbox_xyxy'] for t in truth])
            eligible=[i for i,t in enumerate(truth) if t['class_id']==pred['class_id'] and not found[i] and ious[i]>=.5]
            if eligible:
                best=max(eligible,key=lambda i:(float(ious[i]),i));found[best]=True;tp+=1
            else:rows.append({'image':sample['id'],'prediction_index':index,'class_id':pred['class_id'],'score':pred['score'],'bbox_xyxy':pred['bbox_xyxy'],**classify(pred,truth,found,manifest_by_id[sample['id']].get('excluded_source_objects',[]))})
        fn+=int((~found).sum())
    expected=metric['operating']
    if (tp,len(rows),fn)!=(expected['tp'],expected['fp'],expected['fn']):raise ValueError('FP taxonomy changed operating TP/FP/FN')
    summaries={}
    for variant in ['strict_iou','sensitivity_ioa_pred']:
        categories=Counter();sources=Counter();subtotals=Counter();cross=Counter();suppressed=Counter()
        for row in rows:
            detail=row[variant];category=detail['category'];categories[(category,NAMES[row['class_id']])]+=1
            if category in {'source_unknown','known_non_target'}:sources[detail['detail']['source_type']]+=1
            if category=='known_non_target':subtotals[detail['detail']['subtotal']]+=1
            flags=detail['conditions'];cross['|'.join(f'{k}={int(v)}' for k,v in sorted(flags.items()))]+=1
            if category.startswith('other_'):
                if flags['unknown_overlap_ge_050']:suppressed['target_priority_over_unknown']+=1
                if flags['non_target_overlap_ge_050']:suppressed['target_priority_over_non_target']+=1
        summaries[variant]={'category_class_counts':[{'category':cat,'class':cls,'count':n,'fraction_all_fp':n/len(rows) if rows else 0} for (cat,cls),n in sorted(categories.items())],'source_type_counts':dict(sources),'reference_subtotals':dict(subtotals),'all_condition_cross_table':dict(cross),'overlap_reassigned_by_priority':dict(suppressed)}
    excluded=Counter(e['source_type'] for s in samples for e in manifest_by_id[s['id']].get('excluded_source_objects',[]))
    return {'confidence':threshold,'tp_unchanged':tp,'fp':len(rows),'fn_unchanged':fn,'excluded_source_type_denominators':dict(excluded),'summaries':summaries,'fp_records':rows,'adjusted_precision_or_ap_computed':False}

def candidate_analysis(item,manifest_by_id,with_logo):
    folder=Path(item['metrics_path']).parent; metric=read(folder/'metrics.json'); samples=read(folder/'samples.json');gt=read(folder/'native_ground_truth.json');pred=read(folder/'bbox_predictions.json')
    for name,expected in read(item['job_path'])['output_sha256'].items():
        if sha(folder/name)!=expected:raise ValueError(f'Changed input {folder/name}')
    engine=CocoSubsets(gt,pred); pooled=engine.subset([x['id'] for x in gt['images']]);compare_bbox(pooled,metric['bbox'])
    groups=sorted({s['group_id'] for s in samples}); group_metrics={};logo={}
    for group in groups:
        subset=[s for s in samples if s['group_id']==group];ids=[s['image_id'] for s in subset]
        ap=engine.subset(ids);match=audit.independent_match(subset,metric['operating']['confidence'])
        group_metrics[group]={'images':len(subset),'gt_instances':sum(len(s['truth']) for s in subset),'gt_by_class':dict(Counter(NAMES[t['class_id']] for s in subset for t in s['truth'])),'bbox':ap,'operating':{k:v for k,v in match.items() if k!='per_image'}}
        if with_logo:logo[group]=engine.subset([s['image_id'] for s in samples if s['group_id']!=group])
    # Independent subset rebuild guards accumulation indexing; first group suffices for each candidate.
    engine.verify_subset([s['image_id'] for s in samples if s['group_id']==groups[0]])
    return {'candidate_id':item['candidate_id'],'candidate':item['candidate'],'metric_sha256':sha(folder/'metrics.json'),'sample_sha256':sha(folder/'samples.json'),'p50_ms':metric['latency']['offline_predict_and_postprocess_ms_p50'],'pooled_bbox':pooled,'operating_confidence':metric['operating']['confidence'],'groups':group_metrics,'macro_ap300':average([v['bbox']['ap50_95_max300'] for v in group_metrics.values()]),'macro_ap100':average([v['bbox']['ap50_95_max100'] for v in group_metrics.values()]),'macro_recall':average([v['operating']['recall'] for v in group_metrics.values()]),'leave_one_group_out_bbox':logo,'class_by_native_shortside_recall':metric['operating']['class_by_native_shortside_recall'],'group_bootstrap_recall95':metric['group_bootstrap_recall95']},samples,metric

def ranking(rows,group=None):
    def key(row):
        ap=row['pooled_bbox'] if group is None else row['leave_one_group_out_bbox'][group]
        return (-ap['ap50_95_max300'],-ap['ap50_95_max100'],row['p50_ms'],row['candidate_id'])
    return sorted(rows,key=key)
def kendall(a,b):
    position={r['candidate_id']:i for i,r in enumerate(b)};ids=[r['candidate_id'] for r in a]
    discordant=sum(position[ids[i]]>position[ids[j]] for i in range(len(ids)) for j in range(i+1,len(ids)))
    pairs=len(ids)*(len(ids)-1)//2
    return 1-2*discordant/pairs if pairs else None

def validation(aggregate,frozen,manifest_by_id):
    rows=[];fp={};cache=OUT/'validation_cache'
    wanted={s['candidate']['candidate_id'] for s in frozen['selections'].values()}|{f'{arm}__epoch_20__{p}' for arm in ['baseline','improved'] for p in ['native','dual']}
    for arm in ['baseline','improved']:
        for item in aggregate['arms'][arm]['validation_candidates']:
            cp=cache/(item['candidate_id']+'.json')
            if cp.exists():
                rec=read(cp)
                if rec['script_sha256']!=sha(__file__) or rec['metric_sha256']!=item['metrics_sha256']:raise ValueError('Cache belongs to changed code/input')
                row=rec['result']
                if item['candidate_id'] in wanted:fp[item['candidate_id']]=rec['fp_diagnostics']
            else:
                row,samples,metric=candidate_analysis(item,manifest_by_id,True)
                tax=fp_diagnostics(samples,manifest_by_id,metric) if item['candidate_id'] in wanted else None
                if tax is not None:fp[item['candidate_id']]=tax
                write(cp,{'script_sha256':sha(__file__),'metric_sha256':item['metrics_sha256'],'result':row,'fp_diagnostics':tax})
            rows.append(row);print('SUPPLEMENTARY '+item['candidate_id'],flush=True)
    groups=sorted(rows[0]['groups']);stability={};poolwinners=[];excluded_winners={g:[] for g in groups}
    for arm in ['baseline','improved']:
        ar=[r for r in rows if r['candidate']['arm']==arm];full=ranking(ar);selected=frozen['selections'][arm]['candidate']['candidate_id']
        if full[0]['candidate_id']!=selected:raise ValueError('Supplementary pooled rank differs from frozen')
        entries=[];poolwinners.append(full[0])
        for group in groups:
            omitted=ranking(ar,group);excluded_winners[group].append(omitted[0])
            entries.append({'omitted_group':group,'winner':omitted[0]['candidate_id'],'same_as_frozen':omitted[0]['candidate_id']==selected,'frozen_winner_rank':next(i+1 for i,r in enumerate(omitted) if r['candidate_id']==selected),'kendall_tau':kendall(full,omitted)})
        stability[arm]={'frozen_candidate':selected,'pooled_top_two_ap300_gap':full[0]['pooled_bbox']['ap50_95_max300']-full[1]['pooled_bbox']['ap50_95_max300'],'leave_one_group_out':entries}
    global_changes=[{'omitted_group':g,'diagnostic_recommended_arm':ranking(excluded_winners[g],g)[0]['candidate']['arm'],'same_as_frozen':ranking(excluded_winners[g],g)[0]['candidate']['arm']==frozen['recommended_arm']} for g in groups]
    byid={r['candidate_id']:r for r in rows};paired=[]
    for epoch in range(2,21,2):
        for pipeline in ['native','dual']:
            b=byid[f'baseline__epoch_{epoch:02d}__{pipeline}'];i=byid[f'improved__epoch_{epoch:02d}__{pipeline}']
            gd={g:i['groups'][g]['bbox']['ap50_95_max300']-b['groups'][g]['bbox']['ap50_95_max300'] for g in groups}
            if b['candidate']['optimizer_calls']!=i['candidate']['optimizer_calls']:raise ValueError('Paired budgets differ')
            paired.append({'epoch':epoch,'epoch_label':'e20(partial90/112)' if epoch==20 else f'e{epoch}','pipeline':pipeline,'optimizer_calls':b['candidate']['optimizer_calls'],'pooled_ap300_delta':i['pooled_bbox']['ap50_95_max300']-b['pooled_bbox']['ap50_95_max300'],'pooled_ap100_delta':i['pooled_bbox']['ap50_95_max100']-b['pooled_bbox']['ap50_95_max100'],'macro_ap300_delta':i['macro_ap300']-b['macro_ap300'],'group_ap300_deltas':gd,'groups_improved_higher':sum(v>0 for v in gd.values()),'groups_equal':sum(v==0 for v in gd.values()),'class_ap300_deltas':{n:i['pooled_bbox']['per_class'][n]['ap50_95_max300']-b['pooled_bbox']['per_class'][n]['ap50_95_max300'] for n in NAMES}})
    return {'candidate_group_metrics':rows,'selection_stability':stability,'overall_arm_leave_one_group_out':global_changes,'paired_same_epoch_pipeline':paired,'fp_taxonomy':fp}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['val','holdout'],required=True);args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    aggregate=read(SUITE/'aggregate.json');frozen=read(SUITE/'selections_frozen.json');freeze=read(HERE/'freeze_record.json')
    if sha(HERE/'ANALYSIS_ADDENDUM_v1.1.md')!=freeze['document_sha256']:raise ValueError('Addendum changed')
    selection_sha=sha(SUITE/'selections_frozen.json');primary_before={p:sha(p) for p in [SUITE/'aggregate.json',SUITE/'selections_frozen.json']}
    source={r['id']:r for r in read(ROOT/'data/native_manifest.json')['records']}
    if args.phase=='val':
        if aggregate['status']!='SELECTIONS_FROZEN_DEVELOPMENT_HOLDOUT_PENDING':raise ValueError('Val analysis must finish before holdout')
        if any(read(p)['binding']['split']=='test' for p in (SUITE/'jobs').glob('*.json')):raise ValueError('Holdout already exists')
        result=validation(aggregate,frozen,source)
    else:
        if aggregate['status']!='COMPLETED_DEVELOPMENT_HOLDOUT_NOT_D455_VALIDATED':raise ValueError('Main holdout must be complete')
        prior=read(OUT/'supplementary_val.json')
        if prior['script_sha256']!=sha(__file__) or prior['selection_sha256']!=selection_sha:raise ValueError('Supplementary code/selection changed since val freeze')
        groups={};fp={}
        for arm in ['baseline','improved']:
            for cohort,item in aggregate['arms'][arm]['tests'].items():
                key=f'{arm}__{cohort}';row,samples,metric=candidate_analysis(item,source,False);groups[key]=row;fp[key]=fp_diagnostics(samples,source,metric)
                job=read(item['job_path'])
                if audit.stamp(job['started_utc'])<=audit.stamp(prior['completed_utc']):raise ValueError('Holdout predates supplementary val freeze')
        result={'holdout_group_metrics':groups,'fp_taxonomy':fp}
    for p,expected in primary_before.items():
        if sha(p)!=expected:raise ValueError(f'Primary changed during supplementary computation: {p}')
    result.update(schema='r04-supplementary-v1.1',phase=args.phase,status='PASS_SUPPLEMENTARY_SAVED_PREDICTION_DIAGNOSTICS',completed_utc=now(),script_sha256=sha(__file__),selection_sha256=selection_sha,analysis_addendum_sha256=freeze['document_sha256'],primary_selection_unchanged=True,primary_metrics_unchanged=True,ap_definition='COCO bbox AP50-95 classwise max300 after whole-image global1000 prediction cap; max100 compatibility',limitations=['single_seed','development_holdout_seen_in_R03','not_D455','no_target4_instance_masks','FP-overlap does not establish correct detection or annotation error','no adjusted AP/precision computed'])
    write(OUT/f'supplementary_{args.phase}.json',result)
    lines=['# R04 보조 분석 — '+args.phase,'','저장 예측 CPU 분석. 모델·confidence·GT·1차 결과는 변경하지 않았다.','']
    if args.phase=='val':
        lines+=['| 실험 | val 동결 후보 | 한 그룹 제외 후 같은 후보 |','|---|---|---:|']
        for arm,r in result['selection_stability'].items():lines.append(f"| {arm} | {r['frozen_candidate']} | {sum(v['same_as_frozen'] for v in r['leave_one_group_out'])}/6 |")
        lines+=['','| epoch | pipeline | AP300 차이(I−B, %p) | 그룹 평균 차이(%p) | 높은 그룹 수 |','|---:|---|---:|---:|---:|']
        for r in result['paired_same_epoch_pipeline']:lines.append(f"| {r['epoch_label']} | {r['pipeline']} | {r['pooled_ap300_delta']*100:.3f} | {r['macro_ap300_delta']*100:.3f} | {r['groups_improved_higher']}/6 |")
    else:
        lines+=['| 실험/집합 | pooled AP300 | 그룹 평균 AP300 |','|---|---:|---:|']
        for key,r in result['holdout_group_metrics'].items():lines.append(f"| {key} | {r['pooled_bbox']['ap50_95_max300']*100:.3f}% | {r['macro_ap300']*100:.3f}% |")
    lines+=['','AP는 이미지 전체 최대1000개 제한 후 COCO 클래스별 최대300개다. 단일 seed와 작은 그룹 수에서 통계적 우월성은 주장하지 않는다.','FP 분류·source_type·교차표·클래스/크기 분모는 대응 JSON에 있다. unknown 위 FP를 정답이나 라벨 오류로 단정하지 않으며 보정 AP는 계산하지 않았다.']
    (OUT/f'supplementary_{args.phase}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'phase':args.phase,'status':result['status'],'output':str(OUT/f'supplementary_{args.phase}.json')}))
if __name__=='__main__': main()
