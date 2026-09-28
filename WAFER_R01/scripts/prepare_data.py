"""Prepare source-original VOC photos only; hash/family groups precede splitting."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json, os, re, shutil
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageOps
import yaml

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'raw/VOC2007'; DATA=ROOT/'data'; REPORTS=ROOT/'reports'
ORIGINAL=re.compile(r'^(edge_bite\d+|gray_line_\d+|open_\d+|scratch_\d+|waferImg_\d+|waferImg\d+_\d+)$')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    if DATA.exists(): raise RuntimeError('Preserve existing prepared data')
    records=[]; classes=Counter(); decoded_groups=defaultdict(list); thumbnails={}
    xmls=sorted((SOURCE/'Annotations').glob('*.xml'))
    if len(xmls)!=2132: raise RuntimeError(f'Expected downloaded2132 originals; found {len(xmls)}')
    for p in xmls:
        stem=p.stem
        if not ORIGINAL.fullmatch(stem): raise RuntimeError(f'Unexpected augmented name {stem}')
        candidates=[q for q in (SOURCE/'JPEGImages').glob(stem+'.*') if q.suffix.lower() in {'.jpg','.jpeg','.png'}]
        if len(candidates)!=1: raise RuntimeError(f'Missing/ambiguous source image {stem}')
        image=candidates[0]; tree=ET.parse(p).getroot()
        with Image.open(image) as im:
            im=ImageOps.exif_transpose(im).convert('RGB'); width,height=im.size
            decoded=hashlib.sha256(f'{width}x{height}'.encode()+im.tobytes()).hexdigest()
            thumb=np.asarray(im.convert('L').resize((64,64)),dtype=np.float32)
            thumbnails[stem]=thumb
        if (width,height)!=(int(tree.findtext('size/width')),int(tree.findtext('size/height'))):
            raise RuntimeError(f'Image/XML shape mismatch: {stem}')
        objects=[]
        for i,o in enumerate(tree.findall('object')):
            raw_name=o.findtext('name').strip()
            name={'Short_circuit':'short_circuit','short_cricuit':'short_circuit'}.get(raw_name,raw_name)
            b=o.find('bndbox')
            original=[float(b.findtext(k)) for k in ['xmin','ymin','xmax','ymax']]
            x1,y1,x2,y2=original; x1-=1; y1-=1
            if not (0<=x1<x2<=width and 0<=y1<y2<=height):
                raise RuntimeError(f'Invalid VOC1-based inclusive box {stem}: {original}')
            objects.append({'instance_id':f'{stem}:{i}','class':name,'raw_class':raw_name,'bbox_xyxy':[x1,y1,x2,y2],
                            'source_voc_bbox':original,'difficult':o.findtext('difficult','0')})
            classes[name]+=1
        if not objects: raise RuntimeError(f'Empty source label requires manual check {stem}')
        records.append({'id':stem,'image':str(image.resolve()),'xml':str(p.resolve()),'width':width,'height':height,
                        'image_sha256':sha(image),'xml_sha256':sha(p),'decoded_sha256':decoded,'objects':objects})
        decoded_groups[decoded].append(stem)
    acquired_count=len(records); acquired_boxes=sum(len(r['objects']) for r in records)
    # Identical pixels with incompatible annotations are quarantined, not relabeled.
    by_id={r['id']:r for r in records}; quarantined=[]
    for members in decoded_groups.values():
        if len(members)<2: continue
        signatures={json.dumps(sorted((o['class'],o['bbox_xyxy']) for o in by_id[s]['objects'])) for s in members}
        if len(signatures)>1: quarantined.extend(members)
    quarantined_set=set(quarantined)
    records=[r for r in records if r['id'] not in quarantined_set]
    decoded_groups={h:[s for s in ss if s not in quarantined_set] for h,ss in decoded_groups.items()}
    decoded_groups={h:ss for h,ss in decoded_groups.items() if ss}
    names=sorted(classes)
    if len(names)!=6: raise RuntimeError(f'Expected six source classes, found {names}')
    ids=[r['id'] for r in records]; parent={s:s for s in ids}
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(x,y):
        x,y=find(x),find(y)
        if x!=y: parent[max(x,y)]=min(x,y)
    for members in decoded_groups.values():
        for member in members[1:]: union(members[0],member)
    families=defaultdict(list)
    for s in ids:
        m=re.match(r'^(waferImg\d+)_\d+$',s)
        if m: families[m.group(1)].append(s)
    for members in families.values():
        for member in members[1:]: union(members[0],member)
    source_audit_path=ROOT.parent/'wafer_research/wafer_datas_source_audit.json'
    source_audit=json.loads(source_audit_path.read_text(encoding='utf-8'))
    appearance_edges=[]
    for pair in source_audit['phash_candidates_hamming_le4']:
        if pair['a'] in parent and pair['b'] in parent:
            union(pair['a'],pair['b']); appearance_edges.append(pair)
    # Candidate near-duplicates: same coarse dHash AND low full thumbnail error.
    buckets=defaultdict(list)
    for s in ids:
        v=np.asarray(Image.fromarray(thumbnails[s].astype('uint8')).resize((9,8)))
        buckets[np.packbits(v[:,1:]>v[:,:-1]).tobytes()].append(s)
    near_pairs=[]
    for members in buckets.values():
        for i,s in enumerate(members):
            for t in members[:i]:
                if find(s)==find(t): continue
                a,b=thumbnails[s],thumbnails[t]
                mae=float(np.mean(np.abs(a-b)))
                if mae<=2.0:
                    union(s,t); near_pairs.append({'first':s,'second':t,'gray64_mae':mae})
    groups=defaultdict(list)
    for r in records:
        r['group_id']=find(r['id']); groups[r['group_id']].append(r)
    keys=sorted(groups); mat=np.zeros((len(keys),len(names)+1),dtype=int)
    for i,key in enumerate(keys):
        mat[i,-1]=len(groups[key])
        for r in groups[key]:
            for o in r['objects']: mat[i,names.index(o['class'])]+=1
    total=mat.sum(0); rng=np.random.default_rng(42); best=None
    cut1=int(.70*len(keys)); cut2=int(.85*len(keys))
    for trial in range(2000):
        order=rng.permutation(len(keys)); portions=[order[:cut1],order[cut1:cut2],order[cut2:]]
        counts=np.array([mat[x].sum(0) for x in portions])
        if np.any(counts[:,:-1]<5): continue
        target=np.array([.7,.15,.15])[:,None]*total
        cost=float(np.mean(((counts-target)/np.maximum(target,1))**2))
        key=(cost,trial)
        if best is None or key<best[0]: best=(key,portions,counts)
    if best is None: raise RuntimeError('No grouped split with all classes represented')
    split_groups={split:[keys[i] for i in part] for split,part in zip(['train','val','test'],best[1])}
    assignment={g:s for s,gg in split_groups.items() for g in gg}
    files=[]; summary={}; annotations=0
    for r in records:
        split=assignment[r['group_id']]; r['split']=split
        folder=DATA/'images'/split; folder.mkdir(parents=True,exist_ok=True)
        image=Path(r['image']); output=folder/image.name
        try: os.link(image,output)
        except OSError: shutil.copy2(image,output)
        label=DATA/'labels'/split/(r['id']+'.txt'); label.parent.mkdir(parents=True,exist_ok=True)
        lines=[]
        for o in r['objects']:
            cid=names.index(o['class']); x1,y1,x2,y2=o['bbox_xyxy']; w,h=r['width'],r['height']
            lines.append(f'{cid} {(x1+x2)/2/w:.10f} {(y1+y2)/2/h:.10f} {(x2-x1)/w:.10f} {(y2-y1)/h:.10f}')
        label.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        r['prepared_image']=str(output.resolve()); r['prepared_label']=str(label.resolve())
        for file in [output,label,Path(r['xml'])]: files.append({'path':str(file.resolve()),'sha256':sha(file)})
        annotations+=len(lines)
    for split in split_groups:
        rr=[r for r in records if r['split']==split]
        counter=Counter(o['class'] for r in rr for o in r['objects'])
        summary[split]={'images':len(rr),'groups':len(split_groups[split]),'bbox':sum(counter.values()),'classes':dict(counter)}
    data=DATA/'data.yaml'
    data.write_text(yaml.safe_dump({'path':str(DATA.resolve()),'train':'images/train','val':'images/val','test':'images/test',
                                   'names':names},sort_keys=False,allow_unicode=True),encoding='utf-8')
    manifest={'names':names,'records':records,'split_policy':'seed42,2000 group-balanced candidates minimizing class/image proportion error; labels only, no prediction selection',
              'group_policy':'decoded exact duplicates; explicit waferImgN filename families; pHash Hamming<=4 appearance families conservatively together; same dHash and gray64 MAE<=2 near-copies',
              'physical_wafer_lot_ids_available':False,'same_design_independence_verified':False,
              'bbox_convention':'VOC 1-based inclusive -> zero-based continuous xyxy xmin-1,ymin-1,xmax,ymax',
              'source_augmentation_excluded':True,'class_mapping':{'Short_circuit':'short_circuit','short_cricuit':'short_circuit'},
              'other_class_names_preserve_source_spelling':True}
    save(DATA/'manifest.json',manifest)
    files.extend({'path':str(p.resolve()),'sha256':sha(p)} for p in [data,DATA/'manifest.json'])
    save(REPORTS/'data_audit.json',{'status':'PASS_DATA_AUDIT','utc':datetime.now(timezone.utc).isoformat(),
         'source_original_candidates_acquired':acquired_count,'source_boxes_acquired':acquired_boxes,
         'source_images':len(records),'source_bbox':annotations,'names':names,'split':summary,'files':files,
         'quarantined_conflicting_exact_duplicate_images':sorted(quarantined),'source_audit_sha256':sha(source_audit_path),
         'appearance_group_edges':len(appearance_edges),'appearance_groups_are_not_physical_wafer_ids':True,
         'data_yaml_sha256':sha(data),'source_commit':'17dbc7e19f4e354f9b2b436af66e5abdaf2ea4b1',
         'no_group_overlap':True,'no_decoded_exact_hash_overlap':True,'near_copy_grouped_pairs':near_pairs,
         'source_family_counts':{k:len(v) for k,v in families.items()},'split_cost':best[0][0],
         'physical_wafer_independence':'NOT VERIFIED; source files have no physical wafer/lot identifiers',
         'new_human_labels':0,'instance_mask_labels':0,'preparation_script_sha256':sha(__file__)})
    print(json.dumps({'status':'PASS_DATA_AUDIT','classes':names,'source_images':len(records),'bbox':annotations,'splits':summary,'near_pairs':len(near_pairs)}))


if __name__=='__main__': main()
