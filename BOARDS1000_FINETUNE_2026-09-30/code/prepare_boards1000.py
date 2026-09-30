from pathlib import Path
from collections import Counter,defaultdict
import hashlib
import json
import shutil
import sys

import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix

HERE=Path(__file__).resolve().parent
SOURCE=Path(r'C:\Users\hkjun\Documents\Codex\2026-09-27\new-chat\work\boards_ports')
PREVIOUS=HERE/'boards8_pilot_20260929'
TRIAL=HERE/'boards1000_s650_v150_t200_e50_20260929'
ORIGINAL=SOURCE/'datasets/boards_v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if TRIAL.exists(): raise SystemExit(f'Preserving existing experiment: {TRIAL}')
index=read(ORIGINAL/'dataset_index.json')
old_names=index['classes']
names=[n for n in old_names if n!='stm32_nucleo']
groups=defaultdict(list)
for r in index['records']:
    if r['source']!='iotkits_v1': continue
    labs=[line.split() for line in (ORIGINAL/r['label']).read_text().splitlines()]
    classes={old_names[int(line[0])] for line in labs}
    assert 'stm32_nucleo' not in classes
    groups[r['group']].append({**r,'_classes':classes,'_labels':labs})
gids=sorted(groups)
N=len(gids)
SPLITS=['train','val','test','excluded']
targets=[650,150,200,53]
sizes=np.array([len(groups[g]) for g in gids])
totals=Counter(c for rs in groups.values() for r in rs for c in r['_classes'])
assert sum(sizes)==1053
class_images=np.array([[sum(c in r['_classes'] for r in groups[g]) for g in gids] for c in names])
# Four binary assignments per whole connected group, plus absolute-deviation slacks.
binary=N*4
slacks=len(names)*3*2
count=binary+slacks
c=np.zeros(count)
lo=np.zeros(count)
hi=np.concatenate([np.ones(binary),np.full(slacks,np.inf)])
rng=np.random.default_rng(20260929)
for i,g in enumerate(gids):
    original_split=groups[g][0]['split']
    assert {r['split'] for r in groups[g]}=={original_split}
    for s,name in enumerate(SPLITS):
        c[4*i+s]=rng.uniform(0,1e-7)
        if name!=original_split: c[4*i+s]+=0.0001
        if name=='test' and original_split=='train': c[4*i+s]+=0.002
    # Retain every earlier test group in test, never move it back into training.
    if original_split=='test':
        lo[4*i+2]=hi[4*i+2]=1

rows=N+4+len(names)*3*2
A=lil_matrix((rows,count),dtype=float)
lower=[];upper=[];row=0
for i in range(N):
    A[row,4*i:4*i+4]=1;lower.append(1);upper.append(1);row+=1
for s,target in enumerate(targets):
    for i,size in enumerate(sizes): A[row,4*i+s]=size
    lower.append(target);upper.append(target);row+=1
for ci,cls in enumerate(names):
    for s in range(3):
        target=totals[cls]*targets[s]/1053
        slack=binary+(ci*3+s)*2
        for i,value in enumerate(class_images[ci]): A[row,4*i+s]=value
        A[row,slack]=1;A[row,slack+1]=-1
        c[slack]=c[slack+1]=1/max(target,1)
        lower.append(target);upper.append(target);row+=1
        # Several groups in every split; group count does not prove real scene independence.
        for i,value in enumerate(class_images[ci]):
            if value: A[row,4*i+s]=1
        lower.append(3 if s==0 else 2);upper.append(np.inf);row+=1
solution=milp(c,integrality=np.concatenate([np.ones(binary),np.zeros(slacks)]),bounds=Bounds(lo,hi),constraints=LinearConstraint(A.tocsr(),lower,upper),options={'time_limit':30,'mip_rel_gap':0.002})
if solution.x is None: raise RuntimeError(f'No feasible group split: {solution.message}')
x=np.rint(solution.x[:binary]).astype(int).reshape(N,4)
assert np.all(x.sum(axis=1)==1)
assignment={g:SPLITS[int(np.argmax(x[i]))] for i,g in enumerate(gids)}
assert [int(sum(sizes*x[:,s])) for s in range(4)]==targets
TRIAL.mkdir()
for name in ['common.py','assemble.py','verify_datasets.py']:
    shutil.copyfile(PREVIOUS/name,TRIAL/name)
mapping=read(PREVIOUS/'class_map.json')
mapping.pop('pilot_scope',None)
mapping['models']['boards']['purpose']='8-class board detector; 1000-image development experiment'
mapping['experiment_scope']={'selected_source':'iotkits_v1','requested_total':1000,'split_targets':dict(zip(SPLITS[:3],targets[:3])),'parent_class_map_sha256':sha(SOURCE/'class_map.json'),'nucleo_excluded':True,'selection_uses_model_predictions':False,'test_role':'development evaluation; includes previously evaluated test groups and prior train/val groups; not an untouched final test'}
write(TRIAL/'class_map.json',mapping)
root=TRIAL/'datasets/boards_v1'
for kind in ['images','labels']:
    for split in SPLITS[:3]: (root/kind/split).mkdir(parents=True)
records=[];selection=[];box_counts={s:Counter() for s in SPLITS[:3]};class_groups={s:defaultdict(set) for s in SPLITS[:3]}
for g in gids:
    split=assignment[g]
    for r in groups[g]:
        selection.append({'id':r['id'],'group':g,'original_split':r['split'],'assigned_split':split})
        if split=='excluded': continue
        new_image=f'images/{split}/{Path(r["image"]).name}'
        new_label=f'labels/{split}/{Path(r["label"]).name}'
        shutil.copyfile(ORIGINAL/r['image'],root/new_image)
        lines=[]
        for cid,*coords in r['_labels']:
            cls=old_names[int(cid)]
            lines.append(' '.join([str(names.index(cls)),*coords]))
            box_counts[split][cls]+=1
            class_groups[split][cls].add(g)
        (root/new_label).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        records.append({k:v for k,v in r.items() if not k.startswith('_')}|{'split':split,'image':new_image,'label':new_label,'label_sha256':sha(root/new_label),'prior_split':r['split']})
new_index={**index,'classes':names,'class_map_sha256':sha(TRIAL/'class_map.json'),'records':records}
write(root/'dataset_index.json',new_index)
(root/'data.yaml').write_text(f'path: {root.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(names)),encoding='utf-8')
summary={'source':str(ORIGINAL),'source_verification_sha256':sha(SOURCE/'datasets/verification.json'),'source_dataset_fingerprint':read(SOURCE/'datasets/verification.json')['boards_v1']['dataset_fingerprint'],'source_used':'iotkits_v1','candidate_images':1053,'selected_images':1000,'images':dict(zip(SPLITS[:3],targets[:3])),'excluded_images':53,'boxes':{s:dict(v) for s,v in box_counts.items()},'groups':dict(Counter(assignment.values())),'class_group_counts':{s:{c:len(gs) for c,gs in value.items()} for s,value in class_groups.items()},'optimizer':'scipy.optimize.milp; assignment based on counts/group provenance only','optimization_status':solution.message,'optimization_gap':float(solution.mip_gap),'prior_test_images_preserved':106,'test_prior_split_counts':dict(Counter(r['prior_split'] for r in records if r['split']=='test')),'scope':mapping['experiment_scope'],'group_selection':selection}
write(TRIAL/'derivation.json',summary)
sys.path.insert(0,str(TRIAL))
from verify_datasets import check
result=check(root)
write(TRIAL/'datasets/verification.json',{'boards_v1':result})
print(json.dumps({k:v for k,v in summary.items() if k!='group_selection'}|{'structure_pass':result['structure_pass'],'training_ready':result['training_ready'],'label_problem_count':result['label_problem_count'],'cross_split_candidates':result['cross_split_near_duplicate_count']},indent=2,ensure_ascii=False))
assert result['training_ready'] and result['structure_pass']
