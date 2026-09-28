"""Prepare a local private-archive addition; never commit, push, or publish.

Run after WAFER_Surface_R01 is packaged and its English-named verification
receipt is staged in outputs with identical bytes to the Korean receipt.
Preserves all existing archive files except the root README/CURRENT_STATUS.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess

BASE=Path(__file__).resolve().parents[3];OUT=BASE/'outputs'
SOURCE=OUT/'WAFER_Surface_R01';CHECKOUT=BASE/'work/github_private_archive'
REPO='hkjung1011/pcb-d455-component-vision';TAG='wafer-r01-2026-09-29'
R04_COMMIT='0af9cfcc78a9940baaf8f84ae8080bca79aa6b43'
R03_COMMIT='fe1c2949a6d1d9f92b3d68ec577acd8d5822524c'
RELEASE=f'https://github.com/{REPO}/releases/tag/{TAG}'
ZIP=OUT/'WAFER_Surface_R01.zip'
RECEIPT=OUT/'WAFER_Surface_R01_검증기록.json'
ENGLISH_RECEIPT=OUT/'WAFER_Surface_R01_verification.json'
SUMS_NAME='WAFER_R01_SHA256SUMS.txt'

def command(args,cwd=None):return subprocess.check_output(args,cwd=cwd,text=True,encoding='utf-8').strip()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git_blob(p):
    data=Path(p).read_bytes();return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
def write(p,text):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text.strip()+'\n',encoding='utf-8')
def save(p,obj):write(p,json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def tracked_tree(commit):
    raw=subprocess.check_output(['git','ls-tree','-rz','--full-tree',commit],cwd=CHECKOUT)
    out={}
    for line in raw.split(b'\0'):
        if not line:continue
        header,path=line.split(b'\t',1);mode,kind,digest=header.decode().split()
        if kind!='blob':raise RuntimeError('Unexpected non-blob in preserved archive')
        out[path.decode('utf-8')]=digest
    return out
def tag_commit(tag):
    obj=json.loads(command(['gh','api',f'repos/{REPO}/git/ref/tags/{tag}']))['object']
    for _ in range(5):
        if obj['type']=='commit':return obj['sha']
        if obj['type']!='tag':break
        obj=json.loads(command(['gh','api',f'repos/{REPO}/git/tags/{obj["sha"]}']))['object']
    raise RuntimeError('Cannot resolve preserved tag '+tag)

def main():
    if command(['git','status','--porcelain'],CHECKOUT):raise RuntimeError('Archive checkout dirty; inspect before preparing')
    if command(['git','rev-parse','HEAD'],CHECKOUT)!=R04_COMMIT:raise RuntimeError('Expected exact completed R04 base commit')
    repo=json.loads(command(['gh','api',f'repos/{REPO}']))
    if not repo['private']:raise RuntimeError('Archive must remain PRIVATE')
    if tag_commit('r04-2026-09-29')!=R04_COMMIT or tag_commit('r03-2026-09-28')!=R03_COMMIT:raise RuntimeError('Prior tag changed')
    remote=json.loads(command(['gh','api',f'repos/{REPO}/commits/main']))['sha']
    if remote!=R04_COMMIT:raise RuntimeError('Remote main moved; inspect before preparing')
    if (CHECKOUT/'WAFER_R01').exists():raise RuntimeError('WAFER_R01 already exists; preserve it')
    originals=tracked_tree(R04_COMMIT)
    for rel,digest in originals.items():
        if not (CHECKOUT/rel).is_file() or git_blob(CHECKOUT/rel)!=digest:raise RuntimeError('Original tracked bytes differ: '+rel)
    checks=read(SOURCE/'SHA256SUMS.json');status=read(SOURCE/'CURRENT_STATUS.json');verification=read(RECEIPT)
    if status.get('revision')!='WAFER_SURFACE_R01' or status.get('status')!='TRAINED_AND_PUBLIC_IMAGE_TESTED':raise RuntimeError('Wafer package not completed')
    if status.get('d455_verified') is not False or status.get('instance_masks_trained') is not False or status.get('source_data_redistributed') is not False:raise RuntimeError('Unexpected scope claims')
    audit=read(SOURCE/'reports/independent_final_results_audit.json')
    if audit.get('status')!='PASS_INDEPENDENT_FINAL_AUDIT_NOT_D455':raise RuntimeError('Final independent audit must pass')
    expected={r['path'] for r in checks}|{'SHA256SUMS.json'}
    actual={p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob('*') if p.is_file()}
    if expected!=actual:raise RuntimeError('Package inventory has missing/unlisted files')
    for row in checks:
        p=SOURCE/row['path']
        if sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise RuntimeError('Package changed: '+row['path'])
    if sha(ZIP)!=verification['zip_sha256'] or ZIP.stat().st_size!=verification['zip_bytes']:raise RuntimeError('ZIP differs from verified receipt')
    if not ENGLISH_RECEIPT.is_file() or sha(ENGLISH_RECEIPT)!=sha(RECEIPT):raise RuntimeError('Root must stage byte-identical WAFER_Surface_R01_verification.json first')
    # Only package-owned, hash-listed scientific PNG/SVG plots may enter Git.
    figure_inventory=read(SOURCE/'figures/inventory.json')
    allowed_plots={f'figures/{f["file"]}':f['sha256'] for row in figure_inventory for f in row['files']}
    for p in SOURCE.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(SOURCE).as_posix()
        if p.suffix.lower() in {'.png','.svg','.jpg','.jpeg','.bmp','.webp','.tif','.tiff'}:
            if rel not in allowed_plots or sha(p)!=allowed_plots[rel]:raise RuntimeError('Unapproved/source image in package: '+rel)
        if p.suffix.lower() in {'.xml','.npy','.npz','.pkl'} or any(part in {'raw','data','JPEGImages','Annotations'} for part in p.relative_to(SOURCE).parts):raise RuntimeError('Raw data/annotation asset in package: '+rel)
    # Everything above is preflight. Mutations below add one new archive snapshot.
    included=[];omitted=[]
    for p in sorted(SOURCE.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(SOURCE);row={'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
        if rel.parts[0]=='models' or p.suffix.lower() in {'.pt','.onnx','.engine','.safetensors'} or p.stat().st_size>40*1024*1024:omitted.append(row)
        else:copy(p,CHECKOUT/'WAFER_R01'/rel);included.append(row)
    old_readme=(CHECKOUT/'README.md').read_text(encoding='utf-8')
    copy(CHECKOUT/'README.md',CHECKOUT/'archive/README_before_wafer_r01.md')
    copy(CHECKOUT/'CURRENT_STATUS.json',CHECKOUT/'archive/CURRENT_STATUS_before_wafer_r01.json')
    # Use actual generated package status byte-for-byte, rather than inventing fields.
    copy(SOURCE/'CURRENT_STATUS.json',CHECKOUT/'CURRENT_STATUS.json')
    copy(ENGLISH_RECEIPT,CHECKOUT/'archive/WAFER_Surface_R01_verification.json')
    split_rows='\n'.join(f'| {name} | {r["images"]} | {r["bbox"]} |' for name,r in status['split'].items())
    heading=f'''# PCB · 웨이퍼 비전 연구 기록

## 2026-09-29 · 웨이퍼 표면 결함 R01 추가

현미경 공개자료의 **YOLO11n {status['epochs']}에폭** bbox 파일럿을 별도로 학습했다. 실제 optimizer 호출 **{status['optimizer_calls']:,}회**, 학습 미사용 이미지 {status['split']['test']['images']}장·bbox {status['split']['test']['bbox']}개에서 **mAP50 {status['test_map50']*100:.2f}% / mAP50–95 {status['test_map50_95']*100:.2f}%**다.

| 분할 | 원본 이미지 | bbox |
|---|---:|---:|
{split_rows}

원본 후보 {status['source_images_acquired']:,}장·bbox {status['source_bbox']:,}개 중 픽셀이 같고 라벨이 충돌하는 98장을 격리해 {status['admitted_images']:,}장·bbox {status['admitted_bbox']:,}개를 사용했다. 여섯 결함 종류의 기존 공개 라벨을 사용했고 신규 수작업 라벨·mask는 0개다. **D455 실측 성능, 물리 웨이퍼/lot 독립성, instance segmentation은 검증하지 않았다.** PCB 점수와 직접 우열 비교하지 않는다.

- [웨이퍼 결과·설정·알고리즘·그래프](WAFER_R01/README.md) · [실제 최신 상태](CURRENT_STATUS.json)
- [시험 지표](WAFER_R01/reports/test_metrics.json) · [독립 결과 감사와 고정 confidence0.25 보조 분석](WAFER_R01/reports/independent_final_results_audit.md)
- [데이터 독립 감사](WAFER_R01/reports/independent_data_audit.json) · [모델이 포함된 비공개 Release]({RELEASE})
- [보존된 PCB R04](R04/README.md) · [PCB R04 상태](R04/CURRENT_STATUS.json) · [PCB R03](R03/README.md)

Git의 `WAFER_R01/`에는 가중치가 생략돼 있다. 가중치는 Release의 `WAFER_Surface_R01.zip`에서 복원한다. 원본 사진·VOC XML·변환 라벨·예측 사진은 업로드하지 않는다. 데이터의 명시적 라이선스가 없어 원본 재배포는 하지 않았다.

---

## 보존된 PCB R04 완료 당시 기록

'''
    write(CHECKOUT/'README.md',heading+old_readme)
    manifest={'snapshot':'WAFER_SURFACE_R01','prepared_utc':datetime.now(timezone.utc).isoformat(),'base_commit':R04_COMMIT,'git_included':included,'release_only':omitted,'release':RELEASE,'zip_bytes':verification['zip_bytes'],'zip_sha256':verification['zip_sha256'],'package_sha_manifest_sha256':sha(SOURCE/'SHA256SUMS.json'),'status_sha256':sha(SOURCE/'CURRENT_STATUS.json'),'source_data_uploaded':False,'prior_preservation_exceptions':['README.md','CURRENT_STATUS.json'],'preserved_tag_commits':{'r04-2026-09-29':R04_COMMIT,'r03-2026-09-28':R03_COMMIT}}
    save(CHECKOUT/'ARCHIVE_MANIFEST_WAFER_R01.json',manifest)
    write(CHECKOUT/'archive'/SUMS_NAME,f'{sha(ZIP)}  WAFER_Surface_R01.zip\n{sha(ENGLISH_RECEIPT)}  WAFER_Surface_R01_verification.json')
    write(CHECKOUT/'RELEASE_NOTES_WAFER_R01.md',f'''# 웨이퍼 표면 결함 R01 · 공개 현미경 이미지 파일럿

YOLO11n {status['epochs']}에폭·실제 optimizer {status['optimizer_calls']:,}회. bbox6종, 이미지2,034장/라벨3,394개. 검증셋의 AP로 best 가중치를 선택하고 SHA를 고정한 뒤 학습 미사용287장/496개를 평가했다.

**mAP50 {status['test_map50']*100:.2f}% · mAP50–95 {status['test_map50_95']*100:.2f}%** (Ultralytics8.4.120, confidence floor0.001, NMS IoU0.7, 이미지 전체 max_det300).

고정 confidence0.25의 별도 TP/FP/FN 분석은 test 전에 기준과 코드를 기록했으며 현장 보정값이 아니다. COCO 보조 AP는 다른 matching·보간·JSON 반올림을 사용하므로 원래 AP와 구분한다.

ZIP에는 best/last 가중치, 코드, 분할/라벨 수량, 해시, 평가·감사와 그래프가 있다. 원본 사진·VOC XML·변환 라벨·예측 사진은 포함하지 않는다. D455 실사용, 새 물리 웨이퍼/lot 일반화 및 mask 성능은 미검증이다. 기존 R04/R03 태그와 파일은 보존했다.

SHA256 `{sha(ZIP)}`
''')
    for rel,digest in originals.items():
        if rel not in {'README.md','CURRENT_STATUS.json'} and git_blob(CHECKOUT/rel)!=digest:raise RuntimeError('Existing archive modified: '+rel)
    for row in included:
        if sha(CHECKOUT/'WAFER_R01'/row['path'])!=row['sha256']:raise RuntimeError('Archive copy changed')
    print(json.dumps({'status':'LOCAL_ARCHIVE_PREPARED_NOT_PUBLISHED','git_files':len(included),'release_only_files':len(omitted),'release_tag':TAG,'release_notes':str(CHECKOUT/'RELEASE_NOTES_WAFER_R01.md'),'sha_asset':str(CHECKOUT/'archive'/SUMS_NAME),'no_commit_push_release_performed':True}))

if __name__=='__main__':main()
