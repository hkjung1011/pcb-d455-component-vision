"""Read-only independent source/split/coordinate audit; no GPU or training."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import ast,hashlib,json,re,traceback,xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image,ImageOps
import yaml

ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports/independent_data_audit.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    report={'schema':'wafer-r01-independent-data-audit-v1','status':'RUNNING','created_utc':datetime.now(timezone.utc).isoformat(),'checks':[],'failures':[],'limitations':['VOC1-based inclusive convention is inferred from source ranges, not author-confirmed','Filename/appearance groups do not prove physical wafer or lot independence','No images have been obtained from D455; no masks are present','Training callback behavior is source-reviewed; completed runtime must be separately checked']}
    def check(condition,text):
        if not condition:raise ValueError(text)
        report['checks'].append(text)
    try:
        manifest=read(ROOT/'data/manifest.json');claimed=read(ROOT/'reports/data_audit.json')
        source_path=ROOT.parent/'wafer_research/wafer_datas_source_audit.json';source=read(source_path);download=read(ROOT.parent/'wafer_research/download_manifest.json')
        rows={r['id']:r for r in manifest['records']};source_by_id={r['id']:r for r in source['records']}
        names=['edge_bite','gray_line','open','scratch','short_circuit','stains_enbedded']
        check(manifest['names']==names,'Six canonical classes and ordered IDs match')
        check(manifest['class_mapping']=={'Short_circuit':'short_circuit','short_cricuit':'short_circuit'},'Only two short-circuit spellings normalized')
        check(len(source_by_id)==2132 and len(rows)==2034,'2132 original candidates;2034 admitted images')
        report['input_sha256']={str(p):sha(p) for p in [ROOT/'data/manifest.json',ROOT/'data/data.yaml',ROOT/'reports/data_audit.json',source_path,ROOT/'scripts/prepare_data.py',ROOT/'scripts/train_pilot.py',Path(__file__)]}
        check(sha(source_path)==claimed['source_audit_sha256'],'Preparation binds exact source audit')
        check(sha(ROOT/'scripts/prepare_data.py')==claimed['preparation_script_sha256'],'Prepared script hash matches execution record')
        download_by_path={r['path']:r for r in download['files']}
        all_records={};bytes_groups=defaultdict(list);pixels_groups=defaultdict(list);phashes={};thumbs={};raw_classes=Counter();source_boxes=0;max_roundtrip=0.;difficult=Counter()
        for index,(sid,sr) in enumerate(sorted(source_by_id.items())):
            image=ROOT/'raw/VOC2007/JPEGImages'/f'{sid}.jpg';xml=ROOT/'raw/VOC2007/Annotations'/f'{sid}.xml'
            for p in [image,xml]:
                original=download_by_path[str(p.relative_to(ROOT/'raw')).replace('\\','/')]
                data=p.read_bytes();digest=hashlib.sha256(data).hexdigest()
                if digest!=original['sha256'] or hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=original['git_blob_sha']:raise ValueError('Source no longer matches downloaded Git blob: '+sid)
            with Image.open(image) as raw:
                rgb=np.array(ImageOps.exif_transpose(raw).convert('RGB'))
                if raw.getexif().get(274,1)!=1:raise ValueError('EXIF orientation needs coordinate review')
            height,width=rgb.shape[:2];digest=sha(image);pixel=hashlib.sha256(f'{width}x{height}'.encode()+rgb.tobytes()).hexdigest()
            check_source_pixel=hashlib.sha256(rgb.tobytes()).hexdigest()
            if digest!=sr['sha256'] or check_source_pixel!=sr['pixel_sha256']:raise ValueError('Source pixel SHA differs')
            bytes_groups[digest].append(sid);pixels_groups[pixel].append(sid)
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);low=cv2.dct(cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32))[:8,:8].reshape(-1)[1:]
            phash=sum(int(bit)<<i for i,bit in enumerate(low>np.median(low)))
            if f'{phash:016x}'!=sr['phash63']:raise ValueError('pHash not reproducible')
            phashes[sid]=phash
            thumbs[sid]=np.array(Image.fromarray(rgb).convert('L').resize((64,64)),dtype=np.float32)
            tree=ET.parse(xml).getroot()
            if (width,height)!=(int(tree.findtext('size/width')),int(tree.findtext('size/height'))):raise ValueError('XML dimensions mismatch')
            objects=[]
            for obj in tree.findall('object'):
                raw_name=obj.findtext('name').strip();name={'Short_circuit':'short_circuit','short_cricuit':'short_circuit'}.get(raw_name,raw_name);raw_classes[raw_name]+=1;source_boxes+=1
                b=[float(obj.findtext('bndbox/'+k)) for k in ['xmin','ymin','xmax','ymax']]
                converted=[b[0]-1,b[1]-1,b[2],b[3]];difficult[obj.findtext('difficult','0')]+=1
                if not 0<=converted[0]<converted[2]<=width or not 0<=converted[1]<converted[3]<=height:raise ValueError('Invalid bounds')
                objects.append((name,converted,b,raw_name))
            all_records[sid]=objects
            if sid in rows:
                r=rows[sid]
                if r['decoded_sha256']!=pixel or r['image_sha256']!=digest or r['xml_sha256']!=sha(xml):raise ValueError('Admitted manifest hash mismatch')
                if len(r['objects'])!=len(objects):raise ValueError('Object count changed')
                prepared=Path(r['prepared_image']);label=Path(r['prepared_label'])
                if sha(prepared)!=digest:raise ValueError('Prepared image changed')
                if prepared.parent.name!=r['split'] or label.parent.name!=r['split']:raise ValueError('Prepared split path mismatch')
                lines=label.read_text(encoding='utf-8').splitlines()
                if len(lines)!=len(objects):raise ValueError('YOLO label count mismatch')
                for o,(name,box,b,raw_name),line in zip(r['objects'],objects,lines):
                    if o['bbox_xyxy']!=box or o['source_voc_bbox']!=b or o['class']!=name or o['raw_class']!=raw_name:raise ValueError('Manifest object differs from XML')
                    cls,cx,cy,w,h=map(float,line.split())
                    if cls!=names.index(name) or not all(0<=v<=1 for v in [cx,cy,w,h]):raise ValueError('Class or normalized label range changed')
                    reconstructed=[(cx-w/2)*width,(cy-h/2)*height,(cx+w/2)*width,(cy+h/2)*height]
                    error=max(abs(a-b) for a,b in zip(reconstructed,box));max_roundtrip=max(max_roundtrip,error)
                    if error>1e-6:raise ValueError('VOC-to-YOLO coordinate roundtrip exceeded tolerance')
            if (index+1)%500==0:print(f'AUDIT_SOURCE {index+1}/2132',flush=True)
        check(source_boxes==3693 and dict(raw_classes)==source['class_counts'],'All3693 source objects and raw seven-name counts preserved')
        check(all(v==0 or str(k)=='0' for k,v in difficult.items()),'No VOC difficult labels require ignored-GT policy')
        conflicting=[];conflict_groups=[]
        for members in pixels_groups.values():
            if len(members)>1:
                signatures={json.dumps(sorted((o[0],o[1]) for o in all_records[s])) for s in members}
                if len(signatures)>1:conflicting.extend(members);conflict_groups.append(members)
        check(len(conflict_groups)==49 and len(conflicting)==98,'49 exact-pixel groups have conflicting annotations;98 images')
        check(set(conflicting)==set(claimed['quarantined_conflicting_exact_duplicate_images'])==set(source_by_id)-set(rows),'All and only conflicting duplicate images quarantined')
        for groups in [bytes_groups,pixels_groups]:
            check(all(len({rows[s]['split'] for s in ids if s in rows})<=1 for ids in groups.values()),'No source-byte/decoded-hash crossing between splits')
        # Independently reconstruct every pHash<=4 pair rather than trust the screening list.
        ids=sorted(phashes);pairs=[]
        for i,a in enumerate(ids):
            for b in ids[i+1:]:
                d=(phashes[a]^phashes[b]).bit_count()
                if d<=4:pairs.append((a,b,d))
        provided={(min(p['a'],p['b']),max(p['a'],p['b']),p['phash_hamming']) for p in source['phash_candidates_hamming_le4']}
        check(set(pairs)==provided,'All-pairs pHash<=4 candidate list independently reproduced')
        edges=defaultdict(set)
        def connect(members):
            for s in members[1:]:edges[members[0]].add(s);edges[s].add(members[0])
        for members in pixels_groups.values():connect([s for s in members if s in rows])
        families=defaultdict(list)
        for sid in rows:
            match=re.fullmatch(r'(waferImg\d+)_\d+',sid)
            if match:families[match.group(1)].append(sid)
        for members in families.values():connect(members)
        kept_edges=[]
        for a,b,d in pairs:
            if a in rows and b in rows:
                connect([a,b]);kept_edges.append((a,b,d))
                if rows[a]['split']!=rows[b]['split']:raise ValueError('pHash candidate crosses split')
        buckets=defaultdict(list)
        for sid in rows:
            small=np.array(Image.fromarray(thumbs[sid].astype('uint8')).resize((9,8)))
            buckets[np.packbits(small[:,1:]>small[:,:-1]).tobytes()].append(sid)
        near_count=0
        for members in buckets.values():
            for i,a in enumerate(members):
                for b in members[i+1:]:
                    if float(np.mean(np.abs(thumbs[a]-thumbs[b])))<=2:
                        connect([a,b]);near_count+=1
        remaining=set(rows);components=[]
        while remaining:
            seed=min(remaining);todo=[seed];comp=set()
            while todo:
                sid=todo.pop()
                if sid in comp:continue
                comp.add(sid);todo.extend(edges[sid]-comp)
            remaining-=comp;components.append(comp)
        check(all({rows[s]['group_id'] for s in c}=={min(c)} for c in components),'Exact/family/pHash/dHash connected components exactly match manifest group IDs')
        check(all(len({rows[s]['split'] for s in c})==1 for c in components),'No reconstructed group crossing')
        split_summary={}
        for split in ['train','val','test']:
            rr=[r for r in rows.values() if r['split']==split];counts=Counter(o['class'] for r in rr for o in r['objects'])
            split_summary[split]={'images':len(rr),'groups':len({r['group_id'] for r in rr}),'bbox':sum(counts.values()),'classes':dict(counts)}
            check(set(counts)==set(names) and min(counts.values())>=5,f'{split}: all6 classes have at least5 boxes')
            check({p.stem for p in (ROOT/'data/labels'/split).glob('*.txt')}=={r['id'] for r in rr},f'{split}: exact label file inventory')
            check({p.stem for p in (ROOT/'data/images'/split).iterdir() if p.is_file()}=={r['id'] for r in rr},f'{split}: exact prepared image inventory')
        check(split_summary==claimed['split'],'Frozen split image/group/bbox/class counts reproduced')
        check(sum(x['bbox'] for x in split_summary.values())==3394==claimed['source_bbox'],'3394 retained source boxes')
        for row in claimed['files']:
            if sha(row['path'])!=row['sha256']:raise ValueError('Prepared audit file hash mismatch')
        config=yaml.safe_load((ROOT/'data/data.yaml').read_text(encoding='utf-8'))
        check(config['names']==names and Path(config['path']).resolve()==(ROOT/'data').resolve(),'YAML class IDs/data root match manifest')
        check(not manifest['physical_wafer_lot_ids_available'] and not manifest['same_design_independence_verified'],'Unknown physical independence explicitly retained')
        # Read installed trainer source, no model initialization or GPU use.
        installed=Path('C:/Users/hkjun/Documents/mcu-vision/.venv-yolo11/Lib/site-packages/ultralytics/engine/trainer.py')
        engine=installed.read_text(encoding='utf-8');local=(ROOT/'scripts/train_pilot.py').read_text(encoding='utf-8');tree=ast.parse(local)
        funcs={n.name:ast.get_source_segment(local,n) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        check('register_step_post_hook' in funcs['on_start'] and 't.direct_calls += 1' in funcs['on_start'],'Actual optimizer post-step hook increments direct calls')
        check('if t.in_final_eval: return' in funcs['on_epoch'] and 'self.in_final_eval = True' in funcs['final_eval'] and 'finally: self.in_final_eval = False' in funcs['final_eval'],'Extra final-best validation callback excluded from epoch records')
        check(engine.index('self._build_train_pipeline()')<engine.index('self.run_callbacks("on_pretrain_routine_end")'),'Optimizer/train pipeline precedes hook registration')
        check('self.epoch += 1' in engine and 'self.run_callbacks("on_fit_epoch_end")' in engine,'Installed Ultralytics has extra final_eval callback; wrapper guard necessary')
        check('amp=False' in local and 'epochs=10' in local and 'register_step_post_hook' in local,'FP32 ten-epoch pilot with measured steps')
        check('selected_on' in local and local.index("'reports/selection_frozen.json'")<local.index("metrics = evaluator.val"),'Best weight SHA frozen before test evaluation')
        check("split='test'" in local and "conf=.001" in local and "max_det=300" in local,'Explicit test inference protocol and no test operating-threshold report')
        report.update(status='PASS_INDEPENDENT_DATA_AUDIT',source_files_sha_and_git_blob_verified=4264,prepared_audit_files_sha_verified=len(claimed['files']),source_images=2132,retained_images=2034,source_boxes=3693,retained_boxes=3394,quarantined_images=98,conflicting_exact_pixel_groups=49,split=split_summary,phash_all_pairs_reproduced=len(pairs),retained_phash_edges=len(kept_edges),reconstructed_groups=len(components),near_dhash_low_mae_pairs=near_count,maximum_yolo_roundtrip_error_pixels=max_roundtrip,canonical_names=names,training_review={'status':'PASS_SOURCE_REVIEW_NOT_RUNTIME_EXECUTION','installed_trainer_sha256':sha(installed),'pilot_script_sha256':sha(ROOT/'scripts/train_pilot.py'),'optimizer_hook':'direct post-step','extra_final_epoch_callback_guard':True,'epochs_configured':10,'amp':False})
    except Exception as exc:
        report['status']='FAIL_INDEPENDENT_DATA_AUDIT';report['failures'].append(f'{type(exc).__name__}: {exc}');traceback.print_exc()
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:report.get(k) for k in ['status','source_images','retained_images','retained_boxes','reconstructed_groups','maximum_yolo_roundtrip_error_pixels','failures']}))
    if report['failures']:raise SystemExit(1)
if __name__=='__main__':main()
