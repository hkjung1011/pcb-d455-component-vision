"""Original-image COCO evaluation and validation-selected operating point.

Dataset-native coordinates are not a claim of D455 capture resolution.
"""
import argparse
from contextlib import redirect_stdout
from collections import Counter, defaultdict
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'runtime_config'))
sys.path.insert(0, str(ROOT / 'src' if (ROOT / 'src').exists() else ROOT.parent / 'pcb_components' / 'src'))


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def bbox_iou(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    lt = np.maximum(a[:2], b[:2]); rb = np.minimum(a[2:], b[2:])
    overlap = np.maximum(rb-lt, 0).prod()
    union = np.maximum(a[2:]-a[:2], 0).prod()+np.maximum(b[2:]-b[:2], 0).prod()-overlap
    return float(overlap / union) if union > 0 else 0.0


def encode_native_mask(binary, height, width):
    """Encode an already restored binary mask; never resize predictions for scoring."""
    from pycocotools import mask as mu
    binary = np.asarray(binary)
    if binary.shape != (height, width):
        raise ValueError(f'Native mask shape {binary.shape} differs from {(height, width)}')
    if not np.all(np.isfinite(binary)) or not np.all((binary == 0) | (binary == 1)):
        raise ValueError('Expected binary native mask values 0/1')
    rle = mu.encode(np.asfortranarray(binary, dtype=np.uint8))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def operate(samples, threshold, class_count):
    counts = np.zeros((class_count,3), dtype=np.int64) # TP FP FN
    bins = {name:[0,0] for name in ['lt8','8to16','16to32','ge32']}
    by_image = []; negatives = negative_fp = 0
    for sample in samples:
        truths = sample['truth']; found=set(); local=np.zeros_like(counts)
        predictions = sorted((p for p in sample['predictions'] if p['score']>=threshold), key=lambda p:-p['score'])
        if not truths:
            negatives += 1; negative_fp += bool(predictions)
        for p in predictions:
            candidates=[(bbox_iou(p['bbox_xyxy'],g['bbox_xyxy']),j) for j,g in enumerate(truths) if j not in found and g['class_id']==p['class_id']]
            quality,j=max(candidates, default=(0,-1))
            if quality>=.5:
                found.add(j); local[p['class_id'],0]+=1
            else:
                local[p['class_id'],1]+=1
        for j,g in enumerate(truths):
            if j not in found: local[g['class_id'],2]+=1
            box=g['bbox_xyxy']; short=min(box[2]-box[0],box[3]-box[1])
            bucket='lt8' if short<8 else '8to16' if short<16 else '16to32' if short<32 else 'ge32'
            bins[bucket][0]+=int(j in found); bins[bucket][1]+=1
        counts+=local
        by_image.append({'id':sample['id'],'group_id':sample['group_id'],'source':sample.get('source'),
                         'counts_tp_fp_fn':local.tolist(),'ground_truth_count':len(truths),'prediction_count':len(predictions)})
    tp,fp,fn=map(int,counts.sum(axis=0))
    precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else None
    return {'confidence':threshold,'match_box_iou':.5,'tp':tp,'fp':fp,'fn':fn,
        'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if recall is not None and precision+recall else 0.,
        'per_class_counts_tp_fp_fn':counts.tolist(),'size_recall':{k:{'detected':v[0],'total':v[1],'recall':v[0]/v[1] if v[1] else None} for k,v in bins.items()},
        'negative_images':negatives,'negative_images_with_false_positive':negative_fp,
        'negative_false_positive_image_rate':negative_fp/negatives if negatives else None,'per_image':by_image}


def coco_score(ground_truth, predictions, kind='bbox'):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with redirect_stdout(io.StringIO()):
        gt=COCO(); gt.dataset=copy.deepcopy(ground_truth); gt.createIndex()
        if predictions:
            # loadRes annotates its argument in place. Keep the saved evidence pure.
            dt=gt.loadRes(copy.deepcopy(predictions))
        else:
            dt=COCO(); dt.dataset={'images':ground_truth['images'],'categories':ground_truth['categories'],'annotations':[]}; dt.createIndex()
        evaluator=COCOeval(gt,dt,kind); evaluator.params.maxDets=[1,100,300]
        evaluator.evaluate(); evaluator.accumulate()
    p=evaluator.eval['precision']
    def avg(x):
        x=x[x>-1]; return float(x.mean()) if x.size else None
    return {'iou_type':kind,'ap50_95_max100':avg(p[:,:,:,0,1]),'ap50_max100':avg(p[0,:,:,0,1]),
        'ap75_max100':avg(p[5,:,:,0,1]),'ap50_95_max300':avg(p[:,:,:,0,2]),
        'per_class':{category['name']:{'ap50_95_max100':avg(p[:,:,evaluator.params.catIds.index(category['id']),0,1]),'ap50_max100':avg(p[0,:,evaluator.params.catIds.index(category['id']),0,1]),
                        'ap50_95_max300':avg(p[:,:,evaluator.params.catIds.index(category['id']),0,2])} for category in ground_truth['categories']},
        'area_ap50_95_max100':{name:avg(p[:,:,:,i,1]) for i,name in enumerate(['all','small','medium','large'])}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--split',choices=['val','test','external'],default='val')
    p.add_argument('--tile-size',type=int,default=0); p.add_argument('--imgsz',type=int,default=1024)
    p.add_argument('--operating-conf',type=float); p.add_argument('--bootstrap',type=int,default=0)
    p.add_argument('--group-contains'); p.add_argument('--source'); p.add_argument('--cohort'); args=p.parse_args()
    if args.split!='val' and args.operating_conf is None: p.error('Test requires confidence selected on val')
    if args.operating_conf is not None and not 0 <= args.operating_conf <= 1: p.error('Confidence must be in [0, 1]')
    if args.bootstrap < 0 or args.tile_size < 0: p.error('Bootstrap count and tile size must be nonnegative')
    if args.output.exists() and any(args.output.iterdir()): p.error('New output directory required')
    args.output.mkdir(parents=True,exist_ok=True)
    import cv2
    import torch
    from torchvision.ops import batched_nms
    from ultralytics import YOLO
    from pycocotools import mask as mu
    from pcb_components.geometry import generate_tiles
    manifest=json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    names=manifest['names']; rows=[r for r in manifest['records'] if r['split']==args.split]
    if args.group_contains: rows=[r for r in rows if args.group_contains.lower() in r['group_id'].lower() or args.group_contains.lower() in r.get('original_group_id','').lower()]
    if args.source: rows=[r for r in rows if r['source']==args.source]
    if args.cohort: rows=[r for r in rows if r.get('cohort')==args.cohort]
    if not rows: raise SystemExit('No selected images')
    model=YOLO(str(args.weights))
    if [model.names[i].lower() for i in range(len(model.names))] != [v.lower() for v in names]: raise SystemExit('Model and manifest ontology differ')
    segmentation=model.task=='segment'
    if segmentation and args.tile_size: raise SystemExit('This release benchmarks original-resolution whole-image masks only')
    gt={'info':{},'images':[],'annotations':[],'categories':[{'id':i+1,'name':name} for i,name in enumerate(names)]}
    predictions=[]; mask_predictions=[]; samples=[]; timings=[]; annotation_id=0
    for image_id,row in enumerate(rows,1):
        path=Path(row.get('source_image',row['image'])); path=path if path.is_absolute() else args.manifest.parent/path
        im=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
        if im is None: raise ValueError(f'Image decode failed: {path}')
        h,w=im.shape[:2]
        if ('width' in row and int(row['width']) != w) or ('height' in row and int(row['height']) != h):
            raise ValueError(f'Manifest/image native dimensions differ: {path}')
        gt['images'].append({'id':image_id,'width':w,'height':h,'file_name':str(path)})
        truths=[]
        for obj in row['objects']:
            c=int(obj['class_id']); x1,y1,x2,y2=map(float,obj['bbox_xyxy']); annotation_id+=1
            if not (0 <= c < len(names) and 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError(f'Invalid native GT class/box: {row["id"]}: {obj}')
            annotation={'id':annotation_id,'image_id':image_id,'category_id':c+1,'bbox':[x1,y1,x2-x1,y2-y1],
                        'area':(x2-x1)*(y2-y1),'iscrowd':0}
            if segmentation:
                parts=obj.get('polygon_parts_xy') or ([obj['polygon_xy']] if obj.get('polygon_xy') else [])
                if not parts: raise ValueError('Native polygon required for every target mask GT')
                seg=[np.asarray(part,dtype=float).reshape(-1).tolist() for part in parts]
                rle=mu.merge(mu.frPyObjects(seg,h,w)); annotation['segmentation']=seg; annotation['area']=float(mu.area(rle))
            gt['annotations'].append(annotation); truths.append({'class_id':c,'bbox_xyxy':[x1,y1,x2,y2]})
        tiles=[(im,0,0)] if not args.tile_size else [(t.extract(im),t.bounds.x1,t.bounds.y1) for t in generate_tiles(im.shape,args.tile_size,.2)]
        if image_id==1:
            model.predict(tiles[0][0],imgsz=args.imgsz,conf=.001,iou=.6,max_det=1000,retina_masks=segmentation,verbose=False,device=0)
        torch.cuda.synchronize(); began=time.perf_counter(); local=[]
        for crop,ox,oy in tiles:
            result=model.predict(crop,imgsz=args.imgsz,conf=.001,iou=.6,max_det=1000,retina_masks=segmentation,verbose=False,device=0)[0]
            if result.boxes is None: continue
            for j,(box,c,score) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),result.boxes.cls.cpu().numpy(),result.boxes.conf.cpu().numpy())):
                box=box+np.array([ox,oy,ox,oy]); box[[0,2]]=np.clip(box[[0,2]],0,w);box[[1,3]]=np.clip(box[[1,3]],0,h)
                item={'class_id':int(c),'score':float(score),'bbox_xyxy':box.tolist()}
                if segmentation:
                    if result.masks is None: raise ValueError('Mask predictions missing')
                    binary=result.masks.data[j].cpu().numpy()
                    item['segmentation']=encode_native_mask(binary,h,w)
                local.append(item)
        if local and args.tile_size:
            keep=batched_nms(torch.tensor([r['bbox_xyxy'] for r in local],dtype=torch.float32),
                torch.tensor([r['score'] for r in local]),torch.tensor([r['class_id'] for r in local]),.5).tolist()
            local=[local[i] for i in keep[:1000]]
        torch.cuda.synchronize(); timings.append((time.perf_counter()-began)*1000)
        for item in local:
            x1,y1,x2,y2=item['bbox_xyxy']; pred={'image_id':image_id,'category_id':item['class_id']+1,
                'bbox':[x1,y1,x2-x1,y2-y1],'score':item['score']}; predictions.append(pred)
            if segmentation: mask_predictions.append({'image_id':image_id,'category_id':item['class_id']+1,'score':item['score'],'segmentation':item['segmentation']})
        samples.append({'id':row['id'],'image_id':image_id,'group_id':row['group_id'],'source':row['source'],'truth':truths,
                        'predictions':[{k:v for k,v in x.items() if k!='segmentation'} for x in local]})
        if image_id<=3:
            overlay=im.copy()
            for item in sorted(local,key=lambda x:-x['score']):
                if item['score']<.25: continue
                x1,y1,x2,y2=map(int,item['bbox_xyxy']);cv2.rectangle(overlay,(x1,y1),(x2,y2),(40,210,230),max(1,w//700))
                cv2.putText(overlay,names[item['class_id']],(x1,max(15,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,max(.4,w/2200),(40,210,230),1)
            scale=min(1,1400/max(h,w));overlay=cv2.resize(overlay,(int(w*scale),int(h*scale)))
            ok,encoded=cv2.imencode('.jpg',overlay);encoded.tofile(args.output/f'preview_{image_id:03d}_conf025.jpg')
        if image_id%100==0: print(f'EVALUATED {image_id}/{len(rows)}',flush=True)
    save(args.output/'native_ground_truth.json',gt);save(args.output/'bbox_predictions.json',predictions)
    if segmentation:save(args.output/'mask_predictions.json',mask_predictions)
    save(args.output/'samples.json',samples)
    operating_threshold=args.operating_conf
    if operating_threshold is None:
        curve=[operate(samples,float(t),len(names)) for t in np.arange(.05,.951,.05)]
        operating_threshold=max(curve,key=lambda x:(x['f1'],x['confidence']))['confidence']
        save(args.output/'validation_threshold_curve.json',[{k:v for k,v in x.items() if k!='per_image'} for x in curve])
    operating=operate(samples,operating_threshold,len(names))
    report={'scope':manifest.get('scope'),'split':args.split,'cohort':args.cohort,'task':model.task,'class_names':names,
        'annotation_limitations':manifest.get('annotation_limitations',manifest.get('limitations')),
        'operating_metric_scope':'Confidence selected by box-IoU>=0.5 micro-F1 on validation only; mask AP is reported separately. Size bins use native GT bbox short side.',
        'checkpoint_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'images':len(rows),'groups':len({r['group_id'] for r in rows}),'native_gt_instances':len(gt['annotations']),
        'imgsz':args.imgsz,'tile_size':args.tile_size,'tile_overlap':.2 if args.tile_size else None,'tile_nms_iou':.5 if args.tile_size else None,
        'retina_masks':segmentation,'raw_confidence_floor':.001,'prediction_nms_iou':.6,
        'bbox':coco_score(gt,predictions),'mask':coco_score(gt,mask_predictions,'segm') if segmentation else None,
        'operating':operating,'latency':{'offline_predict_and_postprocess_ms_p50':float(np.percentile(timings,50)),
          'offline_predict_and_postprocess_ms_p95':float(np.percentile(timings,95)),'samples':len(timings),
          'warmup_excluded':True,'excludes':['camera acquisition','file decoding','visualization','disk write']},
        'd455_verified':False,'physical_specimen_independence_verified':False}
    if args.bootstrap:
        group_rows=defaultdict(list)
        for r in operating['per_image']:group_rows[r['group_id']].append(np.array(r['counts_tp_fp_fn']).sum(axis=0))
        matrix=np.array([np.sum(v,axis=0) for v in group_rows.values()]); rng=np.random.default_rng(42); values=[]
        for _ in range(args.bootstrap):
            tp,fp,fn=matrix[rng.integers(0,len(matrix),len(matrix))].sum(axis=0)
            if tp+fn:values.append(float(tp/(tp+fn)))
        report['group_bootstrap_recall95']={'resamples':args.bootstrap,'groups':len(matrix),
            'lower':float(np.percentile(values,2.5)) if values else None,'upper':float(np.percentile(values,97.5)) if values else None,
            'scope':'Fixed-threshold recall CI, NOT an AP confidence interval; source-family groups, physical identity may be unknown'}
    save(args.output/'metrics.json',report)
    print(json.dumps({k:report[k] for k in ['split','task','images','native_gt_instances','bbox','mask']},indent=2))


if __name__=='__main__':main()
