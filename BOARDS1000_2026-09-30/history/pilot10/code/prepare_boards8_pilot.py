from pathlib import Path
from collections import Counter
import hashlib
import json
import shutil
import sys

SOURCE=Path(r'C:\Users\hkjun\Documents\Codex\2026-09-27\new-chat\work\boards_ports')
TRIAL=Path(__file__).resolve().parent/'boards8_pilot_20260929'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if TRIAL.exists():
    raise SystemExit(f'Existing trial preserved: {TRIAL}')
original=SOURCE/'datasets/boards_v1'
verified=read(SOURCE/'datasets/verification.json')['boards_v1']
assert verified['structure_pass'] and verified['class_map_sha256']==sha(SOURCE/'class_map.json')
TRIAL.mkdir()
for file in ['verify_datasets.py','assemble.py']:
    shutil.copyfile(SOURCE/file,TRIAL/file)
common=(SOURCE/'common.py').read_text(encoding='utf-8')
holdout=SOURCE.parent/'r03/assets/commons_holdout/images'
common=common.replace("HOLDOUT_IMAGES = WORK / 'r03' / 'assets' / 'commons_holdout' / 'images'",f'HOLDOUT_IMAGES = Path({str(holdout)!r})')
(TRIAL/'common.py').write_text(common,encoding='utf-8')
mapping=read(SOURCE/'class_map.json')
old_names=mapping['models']['boards']['classes']
excluded='stm32_nucleo'
names=[n for n in old_names if n!=excluded]
mapping['models']={'boards':{'purpose':'8-class board pilot; Nucleo excluded pending independent evaluation data','classes':names}}
mapping['pilot_scope']={'excluded_class':excluded,'parent_class_map_sha256':sha(SOURCE/'class_map.json'),'reason':'Nucleo has no validation/test support; exclude entire images rather than creating missing-label negatives'}
for spec in mapping['sources'].values():
    for native,value in spec.get('map',{}).items():
        if value[0] in [excluded,'boards:'+excluded]:
            spec['map'][native]=['drop','excluded_from_eight_class_pilot']
write(TRIAL/'class_map.json',mapping)
root=TRIAL/'datasets/boards_v1'
for kind in ['images','labels']:
    for split in ['train','val','test']:
        (root/kind/split).mkdir(parents=True)
index=read(original/'dataset_index.json')
included=[]
exclusions=[]
counts=Counter()
box_counts={s:Counter() for s in ['train','val','test']}
for r in index['records']:
    lines=(original/r['label']).read_text().splitlines()
    if any(old_names[int(line.split()[0])]==excluded for line in lines):
        exclusions.append({'id':r['id'],'split':r['split'],'reason':'contains excluded Nucleo class'})
        continue
    new_lines=[]
    for line in lines:
        cid,*coords=line.split()
        cls=old_names[int(cid)]
        new_lines.append(' '.join([str(names.index(cls)),*coords]))
        box_counts[r['split']][cls]+=1
    shutil.copyfile(original/r['image'],root/r['image'])
    (root/r['label']).write_text('\n'.join(new_lines)+'\n',encoding='utf-8')
    record={**r,'label_sha256':sha(root/r['label'])}
    included.append(record)
    counts[r['split']]+=1
index.update({'classes':names,'class_map_sha256':sha(TRIAL/'class_map.json'),'records':included})
write(root/'dataset_index.json',index)
(root/'data.yaml').write_text(f'path: {root.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(names)),encoding='utf-8')
write(TRIAL/'derivation.json',{'source':str(original),'source_dataset_fingerprint':verified['dataset_fingerprint'],'source_verification_sha256':sha(SOURCE/'datasets/verification.json'),'parent_class_map_sha256':sha(SOURCE/'class_map.json'),'excluded_class':excluded,'excluded_images':len(exclusions),'exclusions':exclusions,'images':dict(counts),'boxes':{k:dict(v) for k,v in box_counts.items()},'split_groups_preserved':True,'train_selection_uses_test_metrics':False})
sys.path.insert(0,str(TRIAL))
from verify_datasets import check
result=check(root)
write(TRIAL/'datasets/verification.json',{'boards_v1':result})
print(json.dumps({'trial':str(TRIAL),'images':dict(counts),'excluded_images':len(exclusions),'structure_pass':result['structure_pass'],'training_ready':result['training_ready'],'readiness_problems':result['readiness_problems'],'label_problem_count':result['label_problem_count']},ensure_ascii=False,indent=2))
assert result['structure_pass'] and result['training_ready']
