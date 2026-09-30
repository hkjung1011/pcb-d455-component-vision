"""Verify the restored portable snapshot with stdlib only; no inference."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, hashlib, json, math

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def verify(root):
    root=root.resolve();m=json.loads((root/'SNAPSHOT_MANIFEST.json').read_text(encoding='utf-8'))
    listed=set()
    for r in m['files']:
        p=(root/r['path']).resolve();assert p.is_relative_to(root) and p.is_file() and not p.is_symlink(),r['path']
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['path'];listed.add(r['path'])
    assert listed=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='SNAPSHOT_MANIFEST.json' and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.pyo','.cache'}}
    dataset=root/'datasets/boards_v1';index=json.loads((dataset/'dataset_index.json').read_text(encoding='utf-8'))
    provenance=json.loads((root/'PORTABLE_DATASET.json').read_text(encoding='utf-8'))
    counts=Counter();groups=defaultdict(set);hashes=defaultdict(set);expected=set();boxes={s:Counter() for s in ['train','val','test']}
    for r in index['records']:
        s=r['split'];counts[s]+=1;groups[r['group']].add(s);hashes[r['image_sha256']].add(s)
        for field in ['image','label']:
            p=(dataset/r[field]).resolve();assert p.is_relative_to(dataset.resolve()) and sha(p)==r[field+'_sha256'];expected.add(r[field])
        for line in (dataset/r['label']).read_text().splitlines():
            ts=line.split();assert len(ts)==5;cid=int(ts[0]);x,y,w,h=map(float,ts[1:])
            assert 0<=cid<len(index['classes']) and all(math.isfinite(v) for v in [x,y,w,h])
            assert 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1
            assert x-w/2>=-1e-5 and y-h/2>=-1e-5 and x+w/2<=1+1e-5 and y+h/2<=1+1e-5
            boxes[s][index['classes'][cid]]+=1
    assert dict(counts)=={'train':650,'val':150,'test':200}
    assert all(len(s)==1 for s in groups.values()) and all(len(s)==1 for s in hashes.values())
    actual={p.relative_to(dataset).as_posix() for d in ['images','labels'] for p in (dataset/d).rglob('*') if p.is_file() and p.suffix!='.cache'}
    assert actual==expected
    verification=json.loads((root/'evidence/dataset_verification.json').read_text(encoding='utf-8'))['boards_v1']
    assert {s:dict(c) for s,c in boxes.items()}==verification['boxes']
    h=hashlib.sha256()
    for p in sorted(p for p in dataset.rglob('*') if p.is_file() and p.suffix!='.cache'):
        h.update(p.relative_to(dataset).as_posix().encode()+b'\0'+sha(p).encode()+b'\n')
    assert h.hexdigest()==provenance['portable_dataset_fingerprint']
    plan=json.loads((root/'evidence/experiment_plan.json').read_text(encoding='utf-8'))
    for r in plan['evaluation_inventory']['records']:
        assert sha(dataset/r['image'])==r['image_sha256'] and sha(dataset/r['label'])==r['label_sha256']
    assert len(plan['evaluation_inventory']['records'])==350
    return {'status':'PASS','files_hashed':len(m['files']),'images':dict(counts),'portable_dataset_fingerprint':h.hexdigest(),'original_training_dataset_fingerprint':provenance['original_training_dataset_fingerprint'],'inference_executed':False,'scope':'Frozen bytes, labels, groups and evaluation hashes; original pHash/Commons findings preserved, no physical-scene independence claim.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    print(json.dumps(verify(a.root),indent=2))
