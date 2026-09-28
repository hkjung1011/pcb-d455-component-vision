"""Prepare byte-bound, globally grouped board data without changing source images.

Creates candidate manifests; visual review must explicitly enable training later.
Native instance geometry is preserved for independent COCO evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import imagehash
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['raspberry_pi_sbc']
SPLITS = ('train', 'val', 'test')


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stable_id(prefix, values):
    return prefix + hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:20]


def normalize(micro, iot):
    rows = []
    for original in micro:
        row = dict(original)
        row['source_image'] = original['image']
        row['source_group_id'] = original['group_id']
        row['source_class_ids'] = [original['source_model_code']]
        row['anchor_split'] = original.get('split')
        row['native_source_objects'] = [{'class_name': original['source_model'],
            'source_category_id': original['source_model_code'], 'bbox_xyxy': original['bbox_xyxy'],
            'annotation_type': 'native_bbox'}]
        row['objects'] = []
        if row['is_rpi']:
            row['objects'] = [{'class_id': 0, 'class_name': NAMES[0],
                'source_class_name': original['source_model'], 'source_category_id': original['source_model_code'],
                'canonical_model_code': original['source_model'], 'bbox_xyxy': original['bbox_xyxy'],
                'polygon_parts_xy': [], 'native_segmentation': [], 'annotation_type': 'native_bbox',
                'annotation_status': 'source_provided', 'bbox_provenance': 'Micro-PCB native Left/Top/Width/Height CSV',
                'source_url': original['source_url']}]
        row['group_basis'] = 'Existing immutable pHash component' if row['anchor_split'] else 'Complete five-frame source condition group'
        rows.append(row)
    for original in iot:
        row = dict(original)
        row['source_image'] = original['image']
        row['source_group_id'] = stable_id('iot_family_', [original['source'], original['source_parent_filename'],
            sorted(set(o['source_category_id'] for o in original['objects']))])
        row['source_class_ids'] = sorted(set(o['source_category_id'] for o in original['objects']))
        row['native_source_objects'] = original['objects']
        row['objects'] = []
        for source_object in original['objects']:
            if source_object['canonical_candidate'] == NAMES[0]:
                obj = dict(source_object)
                obj['source_class_name'] = obj.pop('class_name')
                obj['class_id'] = 0
                obj['class_name'] = NAMES[0]
                obj['bbox_provenance'] = 'Native COCO bbox xywh converted to xyxy without prediction'
                obj['polygon_provenance'] = 'Native COCO segmentation; original parts retained'
                row['objects'].append(obj)
        row['is_rpi'] = bool(row['objects'])
        row['source_model'] = ','.join(sorted(set(o['class_name'] for o in original['objects'])))
        row['anchor_split'] = None
        row['physical_specimen_independence_verified'] = False
        row['group_basis'] = 'Source + parent filename + source class set; SHA256 and pHash64<=4 union'
        rows.append(row)
    assert len({r['id'] for r in rows}) == len(rows), 'Duplicate record id'
    return rows


def contact_sheet(rows, path, title, columns=4):
    w, h = 340, 280
    canvas = Image.new('RGB', (columns*w, 44+math.ceil(len(rows)/columns)*h), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), title, fill='black')
    evidence = []
    for index, row in enumerate(rows):
        x, y = (index % columns)*w, 44+(index//columns)*h
        with Image.open(row['source_image']) as source:
            picture = ImageOps.contain(source.convert('RGB'), (w-10, h-58))
        ox, oy = x+(w-picture.width)//2, y
        canvas.paste(picture, (ox, oy))
        sx, sy = picture.width/row['width'], picture.height/row['height']
        for obj in row['native_source_objects']:
            bx1, by1, bx2, by2 = obj['bbox_xyxy']
            draw.rectangle((ox+bx1*sx, oy+by1*sy, ox+bx2*sx, oy+by2*sy), outline='red', width=2)
        draw.text((x+5,y+h-54), f'{index+1}. {row["id"]}', fill='black')
        name = row['source_model']
        draw.text((x+5,y+h-37), name[:50], fill='black')
        draw.text((x+5,y+h-21), f'RPi target: {row["is_rpi"]}; native source box shown', fill='black')
        evidence.append({'cell':index+1,'id':row['id'],'source_image':row['source_image'],
                         'source_model':name,'target_positive':row['is_rpi']})
    canvas.save(path)
    save(path.with_suffix('.json'), evidence)


def create_contact_sheets(rows):
    folder = ROOT/'reports'/'contact_sheets'
    folder.mkdir(parents=True, exist_ok=True)
    micro_by_model = defaultdict(list)
    iot_by_model = defaultdict(list)
    for row in rows:
        if row['source'] == 'micro_pcb' and not row['is_rpi']:
            micro_by_model[row['source_model']].append(row)
        elif row['source'] == 'iotkits_v1':
            for obj in row['native_source_objects']:
                iot_by_model[obj['class_name']].append(row)
    sample = []
    for model, candidates in sorted(micro_by_model.items()):
        chosen = sorted(candidates, key=lambda r:r['id'])
        sample.extend([chosen[0], chosen[len(chosen)//2]])
    contact_sheet(sample, folder/'micro_negative_candidates.png', 'Micro-PCB: two sampled candidates per non-RPi source model')
    positive, negative = [], []
    for model, candidates in sorted(iot_by_model.items()):
        chosen = sorted(candidates, key=lambda r:r['id'])[0]
        (positive if chosen['is_rpi'] else negative).append(chosen)
    contact_sheet(positive, folder/'iot_rpi_source_classes.png', 'IoTKIT: one RPi target sample per native source class')
    contact_sheet(negative, folder/'iot_negative_source_classes.png', 'IoTKIT: one non-RPi candidate per native source class')
    print('CONTACT_SHEETS_READY '+str(folder), flush=True)


def fingerprint(row):
    actual_sha = digest(row['source_image'])
    if actual_sha != row['sha256']:
        raise ValueError(f'Source bytes changed after collection: {row["id"]}')
    with Image.open(row['source_image']) as source:
        if source.size != (row['width'], row['height']):
            raise ValueError(f'Image dimension mismatch: {row["id"]}')
        cached = row.get('phash64_from_hash_verified_prior_audit')
        phash = cached or str(imagehash.phash(source.convert('RGB'), hash_size=8))
    return {'id':row['id'],'sha256':actual_sha,'phash64':phash,'width':row['width'],'height':row['height'],
            'phash_source':'reused_after_current_sha256_validation' if cached else 'computed_pillow_imagehash_phash64'}


class UnionFind:
    def __init__(self,n): self.parent=list(range(n)); self.size=[1]*n
    def find(self,a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a=self.parent[a]
        return a
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b: return
        if self.size[a]<self.size[b]: a,b=b,a
        self.parent[b]=a; self.size[a]+=self.size[b]


def make_edges(rows):
    edges=[]; provenance=Counter(); seen_sha={}; seen_group={}
    bands=[defaultdict(list) for _ in range(5)]
    for i,row in enumerate(rows):
        for key,seen,kind in [(row['sha256'],seen_sha,'sha256'),(row['source_group_id'],seen_group,'source_group')]:
            if key in seen:
                edges.append((i,seen[key])); provenance[kind]+=1
            else: seen[key]=i
        value=int(row['phash64'],16)
        candidates=set()
        for b in range(5):
            key=(value>>(13*b)) & ((1<<(12 if b==4 else 13))-1)
            candidates.update(bands[b][key])
            bands[b][key].append(i)
        for j in sorted(candidates):
            if (value ^ int(rows[j]['phash64'],16)).bit_count()<=4:
                edges.append((i,j)); provenance['phash64_distance_lte_4']+=1
    return edges,dict(provenance)


def components(rows, edges, excluded):
    uf=UnionFind(len(rows))
    for a,b in edges:
        if a not in excluded and b not in excluded: uf.union(a,b)
    groups=defaultdict(list)
    for i in range(len(rows)):
        if i not in excluded: groups[uf.find(i)].append(i)
    return list(groups.values())


def anchor_conflicts(rows, groups):
    return [g for g in groups if len({rows[i]['anchor_split'] for i in g if rows[i]['anchor_split']})>1]


def choose_split(rows, groups):
    assignments={}; strata=defaultdict(list)
    for g in groups:
        anchors={rows[i]['anchor_split'] for i in g if rows[i]['anchor_split']}
        if anchors:
            assert len(anchors)==1
            split=next(iter(anchors))
            for i in g: assignments[i]=split
        else:
            strata['positive' if any(rows[i]['is_rpi'] for i in g) else 'negative'].append(g)
    rng=random.Random(42)
    for label, candidates in sorted(strata.items()):
        candidates=sorted(candidates,key=lambda g:min(rows[i]['id'] for i in g))
        rng.shuffle(candidates)
        total=sum(len(g) for g in candidates)
        targets=dict(zip(SPLITS,[total*.7,total*.15,total*.15])); counts=Counter()
        # Allocate full components to the split with greatest normalized deficit.
        for g in sorted(candidates,key=lambda g:-len(g)):
            split=max(SPLITS,key=lambda s:(targets[s]-counts[s])/max(targets[s],1))
            for i in g: assignments[i]=split
            counts[split]+=len(g)
    return assignments


def area(points):
    return abs(sum(points[i][0]*points[(i+1)%len(points)][1]-points[(i+1)%len(points)][0]*points[i][1]
                   for i in range(len(points))))*.5


def validate_object(obj, width, height, record_id):
    box=obj['bbox_xyxy']
    if len(box)!=4 or not all(math.isfinite(v) for v in box) or not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):
        raise ValueError(f'Invalid native bbox: {record_id}: {box}')
    for part in obj.get('polygon_parts_xy',[]):
        if len(part)<3 or any(len(p)!=2 or not all(math.isfinite(v) for v in p) for p in part):
            raise ValueError(f'Invalid native polygon: {record_id}')
        if any(not (0<=x<=width and 0<=y<=height) for x,y in part) or area(part)<=0:
            raise ValueError(f'Out of bounds or degenerate native polygon: {record_id}')


def label_text(row, task):
    lines=[]; width,height=row['width'],row['height']
    for obj in row['objects']:
        validate_object(obj,width,height,row['id'])
        if task=='detect':
            x1,y1,x2,y2=obj['bbox_xyxy']
            coords=[(x1+x2)/2/width,(y1+y2)/2/height,(x2-x1)/width,(y2-y1)/height]
        else:
            parts=obj.get('polygon_parts_xy') or []
            if len(parts)!=1:
                raise ValueError(f'YOLO native single contour required, preserve multipart in COCO: {row["id"]}')
            coords=[v for x,y in parts[0] for v in (x/width,y/height)]
        lines.append('0 '+' '.join(f'{v:.8f}' for v in coords))
    return '\n'.join(lines)+('\n' if lines else '')


def count_records(rows):
    counts={}
    for split in SPLITS:
        values=[r for r in rows if r['split']==split]
        sources={}
        for source in sorted({r['source'] for r in values}):
            subset=[r for r in values if r['source']==source]
            sources[source]={'images':len(subset),'positive_images':sum(bool(r['objects']) for r in subset),
                'negative_images':sum(not r['objects'] for r in subset),'instances':sum(len(r['objects']) for r in subset),
                'native_polygon_instances':sum(bool(o.get('polygon_parts_xy')) for r in subset for o in r['objects'])}
        counts[split]={'images':len(values),'groups':len({r['group_id'] for r in values}),
            'positive_images':sum(bool(r['objects']) for r in values),'negative_images':sum(not r['objects'] for r in values),
            'instances':sum(len(r['objects']) for r in values),
            'native_polygon_instances':sum(bool(o.get('polygon_parts_xy')) for r in values for o in r['objects']),
            'by_source':sources}
    return counts


def materialize(rows, task, audit, suffix=''):
    folder=ROOT/'data'/(('board_detect' if task=='detect' else 'board_segment')+suffix)
    if folder.exists() and any(folder.iterdir()): raise ValueError(f'Refuse overwriting prepared dataset: {folder}')
    for split in SPLITS:
        (folder/'images'/split).mkdir(parents=True,exist_ok=True)
        (folder/'labels'/split).mkdir(parents=True,exist_ok=True)
    output=[]; io_counts=Counter()
    for row in rows:
        target_id=row['id']+'_'+row['sha256'][:12]
        image=folder/'images'/row['split']/(target_id+Path(row['source_image']).suffix.lower())
        label=folder/'labels'/row['split']/(target_id+'.txt')
        try: os.link(row['source_image'],image); io_counts['hardlinks']+=1
        except OSError: shutil.copy2(row['source_image'],image); io_counts['copies']+=1
        label.write_text(label_text(row,task),encoding='utf-8')
        record=dict(row)
        record.update(image=str(image.resolve()),label=str(label.resolve()), task=task,
            annotation_status='source_provided' if row['is_rpi'] else 'negative_candidate_pending_visual_review',
            native_geometry_preserved=True, training_polygon_conversion='none: original single contour' if task=='segment' and row['is_rpi'] else None)
        output.append(record)
    data=folder/'data.yaml'
    data.write_text('path: '+json.dumps(str(folder.resolve()))+'\n'+
        ''.join(f'{split}: images/{split}\n' for split in SPLITS)+'names:\n  0: raspberry_pi_sbc\n',encoding='utf-8')
    manifest={'schema':'r03-board-dataset-v1','task':task,'names':NAMES,'training_eligible':False,
        'eligibility_reason':'Pending parent visual review of source classes and negative candidate contact sheets',
        'review_scope':'Contact-sheet sampling is not exhaustive instance adjudication',
        'source_weighting':'uniform image sampling; source loss factor1.0, no oversampling',
        'evaluation_scope':'Internal source holdout only; physical board/capture independence unverified; not D455 evidence',
        'physical_specimen_independence_verified':False,'dataset_yaml':str(data.resolve()),
        'split_protocol':audit,'counts':count_records(output),'materialization':dict(io_counts),'records':output}
    save(folder/'manifest.json',manifest)
    print('DATASET_READY '+json.dumps({'task':task,'manifest':str(folder/'manifest.json'),'counts':manifest['counts']}),flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contacts-only',action='store_true')
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--micro-records',default='micro_records_v2.json')
    parser.add_argument('--dataset-suffix',default='_v2')
    args=parser.parse_args()
    started=time.time()
    micro_blob=json.loads((ROOT/args.micro_records).read_text(encoding='utf-8'))
    if micro_blob.get('semantic_mapping_status',{}).get('status')=='INVALID_RPI_CLASS_MAPPING_DISCOVERED_BY_VISUAL_AUDIT':
        raise ValueError('Source class mapping is visually disproven; rebuild reviewed micro records before preparation')
    micro=micro_blob['records']
    iot=json.loads((ROOT/'assets'/'board_records.json').read_text(encoding='utf-8'))
    rows=normalize(micro,iot)
    create_contact_sheets(rows)
    if args.contacts_only: return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fingerprints=list(pool.map(fingerprint,rows))
    for row, fp in zip(rows,fingerprints): row.update(fp)
    save(ROOT/'reports'/('board_fingerprints'+args.dataset_suffix+'.json'),fingerprints)
    print(f'FINGERPRINTS_READY {len(rows)} in {time.time()-started:.1f}s',flush=True)
    edges, edge_counts=make_edges(rows)
    groups=components(rows,edges,set()); conflicts=anchor_conflicts(rows,groups)
    # Only existing positive micro splits are immutable anchors. Newly sampled
    # micro negatives are unanchored, just like all newly added IoTKIT images.
    quarantine={i for g in conflicts for i in g if not rows[i]['anchor_split']}
    initial_conflict_details=[{'ids':[rows[i]['id'] for i in g], 'anchor_splits':sorted({rows[i]['anchor_split'] for i in g if rows[i]['anchor_split']})} for g in conflicts]
    groups=components(rows,edges,quarantine)
    remaining=anchor_conflicts(rows,groups)
    if remaining:
        save(ROOT/'reports'/'board_split_conflict_STOP.json',{'reason':'Existing micro anchor splits connected after all unanchored conflict members were quarantined',
            'components':[[{'id':rows[i]['id'],'anchor_split':rows[i]['anchor_split'],'source_group_id':rows[i]['source_group_id']} for i in g] for g in remaining]})
        raise ValueError('Immutable micro anchor conflict; see board_split_conflict_STOP.json')
    # Exact duplicates with disagreeing target presence are unsafe annotation candidates.
    by_sha=defaultdict(list)
    for i,row in enumerate(rows):
        if i not in quarantine: by_sha[row['sha256']].append(i)
    contradictions=[indices for indices in by_sha.values() if len({rows[i]['is_rpi'] for i in indices})>1]
    if contradictions:
        save(ROOT/'reports'/'board_annotation_conflict_STOP.json',[[rows[i]['id'] for i in g] for g in contradictions])
        raise ValueError('Exact same bytes have contradictory RPi presence labels')
    assignments=choose_split(rows,groups)
    discarded=[]; kept=[]; group_summary=[]
    for g in groups:
        group_id=stable_id('globalcc_',sorted(rows[i]['id'] for i in g))
        group_summary.append({'group_id':group_id,'members':[rows[i]['id'] for i in g],
            'split':assignments[g[0]],'anchor_splits':sorted({rows[i]['anchor_split'] for i in g if rows[i]['anchor_split']})})
        unique={}
        for i in sorted(g,key=lambda i:(rows[i]['source']!='micro_pcb', rows[i]['id'])):
            row=rows[i]
            if row['sha256'] in unique:
                discarded.append({'discarded_id':row['id'],'representative_id':unique[row['sha256']]['id'],'reason':'Exact SHA256 duplicate'})
                continue
            row['group_id']=group_id
            row['split']=assignments[i]
            row['existing_split_preserved']=row['split']==row['anchor_split'] if row['anchor_split'] else None
            unique[row['sha256']]=row
            kept.append(row)
    audit={'seed':42,'unanchored_target_ratios':{'train':.70,'val':.15,'test':.15},
        'micro_anchor_splits':'None: source ontology corrected to reviewed G/H/M; all source-family splits freshly assigned' if not any(r['anchor_split'] for r in rows) else 'Immutable existing phash_v2 split assignments',
        'source_class_review':micro_blob.get('semantic_mapping_status'),
        'group_edges':['source condition/family','SHA256 exact bytes','pHash64 Hamming distance <=4'],
        'iot_family_key':['source','source_parent_filename','sorted source class IDs'],
        'edge_counts':edge_counts,'input_images':len(rows),'quarantined_images':len(quarantine),
        'exact_duplicates_removed':len(discarded),'retained_images':len(kept),'groups':len(groups),
        'independence':'Deduplication does not establish physical specimen or capture independence',
        'split_source':'Assigned globally before making detect/segment views; original IoTKIT export split not used'}
    segmentation=[]; seg_excluded=[]
    for row in kept:
        if row['source']!='iotkits_v1': continue
        missing=[o.get('annotation_id') for o in row['objects'] if not o.get('polygon_parts_xy')]
        if missing:
            seg_excluded.append({'id':row['id'],'reason':'RPi target has bbox but no native polygon','annotation_ids':missing})
        else: segmentation.append(row)
    save(ROOT/'reports'/('board_global_split_audit'+args.dataset_suffix+'.json'),dict(audit,
        initial_anchor_conflicts=initial_conflict_details,
        quarantine=[{'id':rows[i]['id'],'source':rows[i]['source'],'reason':'New unanchored member of a conflicting immutable-anchor component'} for i in sorted(quarantine)],
        exact_duplicates=discarded,segmentation_excluded=seg_excluded,components=group_summary))
    materialize(sorted(kept,key=lambda r:r['id']),'detect',audit,args.dataset_suffix)
    materialize(sorted(segmentation,key=lambda r:r['id']),'segment',audit,args.dataset_suffix)
    print(f'COMPLETE elapsed_seconds={time.time()-started:.1f}',flush=True)


if __name__=='__main__': main()
