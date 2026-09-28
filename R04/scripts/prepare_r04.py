"""Freeze a train/val-only grouped split and paired PCB training views.

No test labels or predictions enter split optimization, tiling or sampling.
Known non-target objects remain background; unknown regions are not guessed.
"""
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime, timezone
import copy
import hashlib
import json
import math
import numpy as np
from PIL import Image
import yaml

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'r03'
NAMES=['resistor','capacitor','ic','connector']
PROGRESS=ROOT.parent/'r04_review/progress.log'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def log(s):
    print(s,flush=True)
    with PROGRESS.open('a',encoding='utf-8') as f:f.write('[R04 DATA] '+s+'\n')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sizebin(o):
    a,b,c,d=o['bbox_xyxy'];s=min(c-a,d-b)
    return 0 if s<8 else 1 if s<16 else 2 if s<32 else 3
def group_features(rows):
    v=np.zeros(25)
    for r in rows:
        v[-1]+=1
        for o in r['objects']:
            c=o['class_id'];b=sizebin(o)
            v[c]+=1;v[4+b]+=1;v[8+c*4+b]+=1
    return v

def freeze_split(rows):
    pool=[r for r in rows if r['split'] in ['train','val']]
    groups=sorted({r['board_group'] for r in pool})
    features=np.array([group_features([r for r in pool if r['board_group']==g]) for g in groups])
    total=features.sum(0);active=total>0
    weights=np.array([2]*4+[2]*4+[1]*16+[2],dtype=float)
    best=None;eligible=0
    # Target the TRAIN+VAL pool distribution, not the observed test distribution.
    for combo in combinations(range(len(groups)),6):
        counts=features[list(combo)].sum(0)
        if (counts[:4]<25).any():continue
        if not .18<=counts[:4].sum()/total[:4].sum()<=.34:continue
        if ((counts[:4]/total[:4]<.12)|(counts[:4]/total[:4]>.45)).any():continue
        if counts[5]<50:continue
        eligible+=1
        deviation=np.abs(counts[active]/total[active]-.25)
        score=float(np.average(deviation,weights=weights[active]))
        item=(score,tuple(groups[i] for i in combo))
        if best is None or item<best:best=item
    if best is None:raise ValueError('No feasible train/val-only stratified split')
    val_groups=set(best[1]);result=[]
    for original in rows:
        r=copy.deepcopy(original);r['r03_split']=r['split']
        if r['split'] in ['train','val']:r['split']='val' if r['board_group'] in val_groups else 'train'
        result.append(r)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'policy':'Six whole groups chosen from R03 train+val ONLY, target 25% of pool class/size cross-counts; lexicographic deterministic tie break',
        'candidate_groups':groups,'eligible_combinations':eligible,'objective':best[0],
        'validation_groups':sorted(val_groups),'training_groups':sorted(set(groups)-val_groups),
        'test_policy':'R03 general and Pi groups remain unchanged; previously inspected development holdouts, not pristine final validation',
        'initialization':'Both arms start from the same official yolo11s.pt; R03 weights prohibited because new val can contain former train groups',
        'source_sha256':sha(OLD/'component_assets/component_records.json'),'names':NAMES}
    for split in ['train','val','test','pi_test']:
        subset=[r for r in result if r['split']==split]
        manifest[split]={'images':len(subset),'groups':len({r['board_group'] for r in subset}),
            'class_counts':dict(Counter(o['class_name'] for r in subset for o in r['objects'])),
            'short_side_counts':dict(Counter(['lt8','8to16','16to32','ge32'][sizebin(o)] for r in subset for o in r['objects']))}
    save(ROOT/'data/split.json',manifest)
    return result

def positions(n,size,overlap=.2):
    if n<=size:return [0]
    result=list(range(0,n-size+1,max(1,int(size*(1-overlap)))))
    if result[-1]!=n-size:result.append(n-size)
    return result
def native_crops(row,size):
    return [(x,y,min(row['width'],x+size),min(row['height'],y+size))
            for y in positions(row['height'],size) for x in positions(row['width'],size)]
def intersection(box,crop):
    a,b,c,d=box;x,y,r,t=crop
    result=[max(a,x),max(b,y),min(c,r),min(d,t)]
    area=max(0,result[2]-result[0])*max(0,result[3]-result[1])
    return result,area/((c-a)*(d-b))
def centered_crop(row,o):
    a,b,c,d=o['bbox_xyxy'];side=max(1024,int(math.ceil(max(c-a,d-b)*1.1/32)*32))
    width=min(side,row['width']);height=min(side,row['height'])
    x=min(max(int((a+c-width)/2),0),row['width']-width)
    y=min(max(int((b+d-height)/2),0),row['height']-height)
    return (x,y,x+width,y+height)
def normalize(box,crop):
    a,b,c,d=box;x,y,r,t=crop;w=r-x;h=t-y
    return [(a+c-2*x)/(2*w),(b+d-2*y)/(2*h),(c-a)/w,(d-b)/h]

def make_arm(rows,arm):
    improved=arm=='improved';base=ROOT/'data'/arm
    view_records=[];sidecar={};eligible_original=set();fully_seen=set()
    for row in [r for r in rows if r['split']=='train']:
        crops={crop:'native1024' for crop in native_crops(row,1024)}
        if improved:
            for crop in native_crops(row,2048):crops.setdefault(crop,'context2048')
            for obj in row['objects']:
                if obj['class_id'] in [2,3] or not any(intersection(obj['bbox_xyxy'],c)[1]>=1-1e-12 for c in crops):
                    crops.setdefault(centered_crop(row,obj),'object_centered')
        with Image.open(row['image']) as original:
            original=original.convert('RGB')
            for index,(crop,kind) in enumerate(sorted(crops.items())):
                targets=[];ignores=[]
                for obj in row['objects']:
                    clipped,fraction=intersection(obj['bbox_xyxy'],crop)
                    if fraction<=0:continue
                    nbox=normalize(clipped,crop)
                    item={'instance_id':obj['instance_id'],'class_id':obj['class_id'],'bbox_xywhn':nbox,
                          'visible_fraction':fraction,'source_bbox_xyxy':obj['bbox_xyxy']}
                    if improved and fraction<.5:
                        ignores.append({'bbox_xywhn':nbox,'reason':'target_fragment_lt50','source_instance_id':obj['instance_id']})
                    else:
                        targets.append(item);eligible_original.add(obj['instance_id'])
                        if fraction>=1-1e-12:fully_seen.add(obj['instance_id'])
                if improved:
                    for obj in row['excluded_source_objects']:
                        if obj['source_type']!='unknown':continue
                        clipped,fraction=intersection(obj['bbox_xyxy'],crop)
                        if fraction>0:ignores.append({'bbox_xywhn':normalize(clipped,crop),'reason':'source_unknown','source_instance_id':obj['instance_id']})
                name=f"{row['image_id']}__{index:04d}_{kind}"
                dest=base/'images/train'/(name+'.jpg');label=base/'labels/train'/(name+'.txt')
                dest.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
                image=original.crop(crop);factor=min(1,1024/max(image.size))
                if factor<1:image=image.resize((max(1,round(image.width*factor)),max(1,round(image.height*factor))),Image.Resampling.LANCZOS)
                image.save(dest,quality=92,subsampling=0)
                label.write_text('\n'.join(str(o['class_id'])+' '+' '.join(f'{x:.10f}' for x in o['bbox_xywhn']) for o in targets)+ ('\n' if targets else ''),encoding='utf-8')
                path=str(dest.resolve());sidecar[path]=ignores
                view_records.append({'id':name,'image':path,'label':str(label.resolve()),'sha256':sha(dest),'label_sha256':sha(label),
                    'source_image_id':row['image_id'],'group_id':row['board_group'],'crop_xyxy':crop,'kind':kind,
                    'width':image.width,'height':image.height,'source_to_saved_scale':factor,'targets':targets,'ignore_regions':ignores})
    originals={o['instance_id'] for r in rows if r['split']=='train' for o in r['objects']}
    assert eligible_original==originals, 'A source target is missing from supervised views'
    if improved:assert fully_seen==originals,'Every source target needs at least one complete view'
    for r in [r for r in rows if r['split']=='val']:
        # Copies are avoided with local hardlinks; source bytes are immutable.
        dest=base/'images/val'/(r['image_id']+'.jpg');dest.parent.mkdir(parents=True,exist_ok=True)
        dest.hardlink_to(Path(r['image']))
        label=base/'labels/val'/(r['image_id']+'.txt');label.parent.mkdir(parents=True,exist_ok=True)
        crop=(0,0,r['width'],r['height'])
        label.write_text('\n'.join(str(o['class_id'])+' '+' '.join(f'{x:.10f}' for x in normalize(o['bbox_xyxy'],crop)) for o in r['objects'])+'\n',encoding='utf-8')
    # Equal expected image draws per board group, then 1.5x preference within a
    # group for views containing source IC/connector. Same rule for both arms.
    group_totals=defaultdict(float)
    for r in view_records:
        r['within_group_priority']=1.5 if any(o['class_id'] in [2,3] for o in r['targets']) else 1.0
        group_totals[r['group_id']]+=r['within_group_priority']
    for r in view_records:r['sampling_weight']=r['within_group_priority']/group_totals[r['group_id']]
    save(base/'ignore_regions.json',{'schema':'r04-training-ignore-v1','images':sidecar,
        'scope':'TRAIN ONLY; negative-anchor classification exclusion, not pixel fill or test-GT deletion'})
    save(base/'views.json',{'schema':'r04-training-views-v1','arm':arm,'records':view_records})
    summary={'arm':arm,'training_eligible':True,'names':NAMES,'views':len(view_records),
        'view_types':dict(Counter(r['kind'] for r in view_records)),
        'original_targets':len(originals),'original_targets_with_complete_view':len(fully_seen),
        'supervised_label_exposures':dict(Counter(NAMES[o['class_id']] for r in view_records for o in r['targets'])),
        'ignored_regions_by_reason':dict(Counter(o['reason'] for r in view_records for o in r['ignore_regions'])),
        'new_human_labels':0,'new_instance_masks':0,'data_yaml':str((base/'data.yaml').resolve()),
        'source_ontology_unchanged':True,'known_nontarget_objects_retained_as_background':True,
        'sampling':'448 replacement draws/epoch; equal expected probability per board group; within-group 1.5x IC/connector view priority; not equal per-instance/gradient weights',
        'source_originals_sha256':sha(OLD/'component_assets/component_records.json')}
    save(base/'summary.json',summary)
    data={'path':str(base.resolve()),'train':'images/train','val':'images/val','names':NAMES,
          'r04_ignore_sidecar':str((base/'ignore_regions.json').resolve()),'r04_views_manifest':str((base/'views.json').resolve())}
    (base/'data.yaml').write_text(yaml.safe_dump(data,allow_unicode=True,sort_keys=False),encoding='utf-8')
    log(f"{arm}: {len(view_records)} views, {len(fully_seen)}/{len(originals)} original objects completely visible at least once; ignore {summary['ignored_regions_by_reason']}")
    return summary

def main():
    if (ROOT/'data').exists():raise SystemExit('Preserve existing frozen R04 data; choose a new revision to rerun')
    source=read(OLD/'component_assets/component_records.json')
    rows=freeze_split(source['images'])
    save(ROOT/'data/original_records.json',{'names':NAMES,'images':rows,'scope':'Source bbox ontology preserved; no mounted-only GT edits'})
    old_native=read(OLD/'component_assets/native_manifest.json')
    lookup={r['image_id']:r for r in rows}
    native=copy.deepcopy(old_native)
    native.update(schema='r04-native-source-bbox-v1',training_eligible=False,
        scope='Source target4 bbox development evaluation; general/Pi test already inspected in R03; NOT pristine final / NOT D455',
        validation_policy='New group stratification from old train+val pool only; same official pretrained initialization required')
    for record in native['records']:
        row=lookup[record['id']];s=row['split']
        record['r03_split']=row['r03_split'];record['split']='test' if s in ['test','pi_test'] else s
        record['cohort']='pi_test' if s=='pi_test' else 'general_test' if s=='test' else s
    native['counts']={}
    for split in ['train','val','test']:
        subset=[r for r in native['records'] if r['split']==split]
        native['counts'][split]={'images':len(subset),'groups':len({r['group_id'] for r in subset}),
            'label_instances':sum(len(r['objects']) for r in subset),'per_class':dict(Counter(NAMES[o['class_id']] for r in subset for o in r['objects']))}
    native['native_split_counts']={}
    for split in ['train','val','test','pi_test']:
        subset=[r for r in rows if r['split']==split]
        native['native_split_counts'][split]={'board_groups':len({r['board_group'] for r in subset}),'images':len(subset),
            'unique_instances':dict(Counter(o['class_name'] for r in subset for o in r['objects']))}
    native['counts_test_scope']='counts.test aggregates general_test and pi_test; native_split_counts reports them separately'
    native['source_weighting']='Native evaluation is unweighted. Paired training image-group sampling is specified in each arm summary; not inherited from R03.'
    save(ROOT/'data/native_manifest.json',native)
    summaries=[make_arm(rows,arm) for arm in ['baseline','improved']]
    save(ROOT/'data/summary.json',{'arms':summaries,'split':read(ROOT/'data/split.json')})
    # Frozen plan precedes all R04 optimizer updates and development test jobs.
    save(ROOT/'protocol.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'revision':'R04',
        'arms':['baseline','improved'],'initial_weights':str((OLD/'models/yolo11s.pt').resolve()),'initial_weights_sha256':sha(OLD/'models/yolo11s.pt'),
        'max_epochs_per_arm':20,'max_direct_optimizer_calls_per_arm':1165,'epoch_replacement_draws':448,
        'batch':4,'nbs':8,'amp':False,'optimizer':'AdamW','lr0':.001,'lrf':.01,'warmup_epochs':2,
        'transform_policy':'Shared LetterBox, HSV, horizontal/vertical flips; no RandomPerspective filtering, mosaic, mixup, scale, translation or random crops',
        'class_multipliers':[1.,1.,1.,1.],'save_epoch_candidates':list(range(2,21,2)),
        'inference_candidates':['native','dual'],'validation_rank':['bbox AP50-95 max300 descending','bbox AP50-95 max100 descending','offline p50 ascending','candidate id'],
        'operating_confidence':'val micro-F1 on .05:.95 grid; greatest threshold in a tie; same single global threshold on fixed test',
        'test_order':'All candidate validation for both arms, freeze all selections with hashes, then one general/Pi development-holdout evaluation per selected arm',
        'limitation':'Joint view/ignore intervention; not an individual-factor ablation. R03 historic comparisons differ in split/sampling/batch/augmentation. No fresh untouched final or actual D455 test available.',
        'data_summary_sha256':sha(ROOT/'data/summary.json'),'native_manifest_sha256':sha(ROOT/'data/native_manifest.json')})
    log('Data and protocol frozen. No R04 training or evaluation has run yet.')

if __name__=='__main__':main()
