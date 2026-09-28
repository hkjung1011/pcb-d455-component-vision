import hashlib
import json
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'component_assets'
raw=json.loads((folder/'component_records.json').read_text(encoding='utf-8'))
tiles=json.loads((folder/'component_tile_records.json').read_text(encoding='utf-8'))['tiles']
native=[]; training=[]
for row in raw['images']:
    source=Path(row['image']); split='test' if row['split']=='pi_test' else row['split']
    record={'id':row['image_id'],'source':'wacv2019','source_image':str(source),'image':str(source),
        'width':row['width'],'height':row['height'],'sha256':row['image_sha256'],
        'group_id':'wacv:'+row['board_group'],'original_group_id':row['board_group'],
        'split':split,'cohort':row['split'],'objects':row['objects'],
        'excluded_source_objects':row['excluded_source_objects'], 'annotation_status':'source_provided'}
    native.append(record)
    if split!='train':
        item=dict(record)
        staged_split=row['split']
        staged=folder/'component_dataset'/'images'/staged_split/source.name
        if not staged.exists():
            candidates=list((folder/'component_dataset'/'images'/staged_split).glob(row['image_id']+'.*'))
            if len(candidates)!=1:raise ValueError(('Missing staged image',row['image_id']))
            staged=candidates[0]
        item['image']=str(staged)
        item['label']=str(folder/'component_dataset'/'labels'/staged_split/(staged.stem+'.txt'))
        training.append(item)
for tile in tiles:
    image=Path(tile['image'])
    training.append({'id':tile['tile_id'],'source':'wacv2019','image':str(image),
        'label':str(folder/'component_dataset'/'labels'/'train'/(image.stem+'.txt')),
        'width':tile['width'],'height':tile['height'],'sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
        'group_id':'wacv:'+tile['board_group'],'split':'train','cohort':'train','objects':tile['objects'],
        'original_image_id':tile['source_image_id'],'crop_xyxy':tile['crop_xyxy'],'annotation_status':'source_provided_tiled'})
def package(records,training_eligible):
    counts={}
    for split in ['train','val','test']:
        rows=[r for r in records if r['split']==split]
        counts[split]={'images':len(rows),'groups':len({r['group_id'] for r in rows}),
            'label_instances':sum(len(r['objects']) for r in rows),
            'per_class':dict(Counter(raw['class_names'][o['class_id']] for r in rows for o in r['objects']))}
    return {'schema':'r03-model-dataset-v1','names':raw['class_names'],'training_eligible':training_eligible,
        'scope':'WACV source-provided target4 bounding boxes; RPI3B held out entirely; unknown labels not reclassified; not D455',
        'source_url':raw['source_url'],'license_status':raw['license_status'],
        'source_weighting':'Uniform sampling per native training tile; source factor 1.0, no class oversampling; original instances repeat across overlapping tiles',
        'native_split_counts':raw['native_split_counts'],'counts':counts,'records':records}
(folder/'training_manifest.json').write_text(json.dumps(package(training,True),ensure_ascii=False,indent=2),encoding='utf-8')
(folder/'native_manifest.json').write_text(json.dumps(package(native,False),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(package(training,True)['counts'],indent=2))
