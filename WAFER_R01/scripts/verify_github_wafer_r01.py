"""Read-only remote verification after root publishes the private wafer archive."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess

BASE=Path(__file__).resolve().parents[3];OUT=BASE/'outputs';CHECKOUT=BASE/'work/github_private_archive'
REPO='hkjung1011/pcb-d455-component-vision';TAG='wafer-r01-2026-09-29'
PRESERVED={'r04-2026-09-29':'0af9cfcc78a9940baaf8f84ae8080bca79aa6b43','r03-2026-09-28':'fe1c2949a6d1d9f92b3d68ec577acd8d5822524c'}
def command(args,cwd=None):return subprocess.check_output(args,cwd=cwd,text=True,encoding='utf-8').strip()
def api(endpoint):return json.loads(command(['gh','api',endpoint]))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def git_blob(p):
    data=Path(p).read_bytes();return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
def tag_commit(tag):
    obj=api(f'repos/{REPO}/git/ref/tags/{tag}')['object']
    for _ in range(5):
        if obj['type']=='commit':return obj['sha']
        if obj['type']!='tag':break
        obj=api(f'repos/{REPO}/git/tags/{obj["sha"]}')['object']
    raise RuntimeError('Cannot resolve tag '+tag)
def remote_blobs(commit):
    tree=api(f'repos/{REPO}/git/trees/{commit}?recursive=1')
    if tree.get('truncated'):raise RuntimeError('Remote tree truncated')
    return {r['path']:r['sha'] for r in tree['tree'] if r['type']=='blob'}

def main():
    repo=api(f'repos/{REPO}')
    if not repo['private']:raise RuntimeError('Repository is not PRIVATE')
    if command(['git','status','--porcelain'],CHECKOUT):raise RuntimeError('Local archive checkout must be clean after publish')
    local=command(['git','rev-parse','HEAD'],CHECKOUT);remote=api(f'repos/{REPO}/commits/main')['sha'];tag=tag_commit(TAG)
    if local!=remote or tag!=remote:raise RuntimeError('Local/main/wafer tag mismatch')
    blobs=remote_blobs(remote)
    tracked={p.decode('utf-8') for p in subprocess.check_output(['git','ls-files','-z'],cwd=CHECKOUT).split(b'\0') if p}
    if tracked!=set(blobs):raise RuntimeError('Local tracked file inventory differs from remote')
    for rel,digest in blobs.items():
        if not (CHECKOUT/rel).is_file() or git_blob(CHECKOUT/rel)!=digest:raise RuntimeError('Remote Git blob differs from local bytes: '+rel)
    preserved={}
    for name,expected in PRESERVED.items():
        actual=tag_commit(name)
        if actual!=expected:raise RuntimeError('Historical tag changed: '+name)
        preserved[name]=actual
    prior=remote_blobs(PRESERVED['r04-2026-09-29'])
    for rel,digest in prior.items():
        if rel not in {'README.md','CURRENT_STATUS.json'} and blobs.get(rel)!=digest:raise RuntimeError('Historical archive file changed: '+rel)
    manifest=read(CHECKOUT/'ARCHIVE_MANIFEST_WAFER_R01.json')
    included={r['path'] for r in manifest['git_included']}
    actual={p[len('WAFER_R01/'):] for p in blobs if p.startswith('WAFER_R01/')}
    if included!=actual:raise RuntimeError('Wafer archive inventory mismatch')
    for row in manifest['git_included']:
        p=CHECKOUT/'WAFER_R01'/row['path']
        if sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise RuntimeError('Wafer archive source-copy SHA mismatch')
    if any(p.startswith('WAFER_R01/models/') or p.endswith('.pt') and p.startswith('WAFER_R01/') for p in blobs):raise RuntimeError('Wafer weights unexpectedly included in Git')
    if sha(CHECKOUT/'CURRENT_STATUS.json')!=manifest['status_sha256'] or sha(CHECKOUT/'WAFER_R01/CURRENT_STATUS.json')!=manifest['status_sha256']:raise RuntimeError('Latest status differs from actual packaged status')
    release=api(f'repos/{REPO}/releases/tags/{TAG}')
    if release['draft'] or not release['prerelease']:raise RuntimeError('Expected published prerelease')
    paths={'WAFER_Surface_R01.zip':OUT/'WAFER_Surface_R01.zip','WAFER_Surface_R01_verification.json':OUT/'WAFER_Surface_R01_verification.json','WAFER_R01_SHA256SUMS.txt':CHECKOUT/'archive/WAFER_R01_SHA256SUMS.txt'}
    if {a['name'] for a in release['assets']}!=set(paths) or len(release['assets'])!=3:raise RuntimeError('Expected exactly ZIP, verification receipt and SHA release assets')
    if sha(paths['WAFER_Surface_R01_verification.json'])!=sha(OUT/'WAFER_Surface_R01_검증기록.json'):raise RuntimeError('English verification receipt differs from packaged receipt')
    receipt=read(paths['WAFER_Surface_R01_verification.json'])
    if sha(paths['WAFER_Surface_R01.zip'])!=receipt['zip_sha256'] or paths['WAFER_Surface_R01.zip'].stat().st_size!=receipt['zip_bytes']:raise RuntimeError('ZIP differs from package receipt')
    assets=[]
    for name,p in paths.items():
        a=next(a for a in release['assets'] if a['name']==name)
        if a['size']!=p.stat().st_size or a.get('digest')!='sha256:'+sha(p):raise RuntimeError('Release asset server size/SHA mismatch: '+name)
        assets.append({'name':name,'bytes':a['size'],'sha256':sha(p),'url':a['browser_download_url']})
    record={'status':'VERIFIED_PRIVATE_WAFER_ARCHIVE_AND_RELEASE','utc':datetime.now(timezone.utc).isoformat(),'repo':repo['html_url'],'private':True,'main_commit':remote,'tag':TAG,'tag_commit':tag,'remote_blob_count':len(blobs),'all_remote_blobs_match_local_bytes':True,'wafer_git_files':len(included),'historical_tags_unchanged':preserved,'historical_files_unchanged_except_root_readme_status':True,'prior_blob_count':len(prior),'release':release['html_url'],'prerelease':True,'assets':assets,'source_data_redistributed':False,'d455_verified':False,'physical_wafer_independence_verified':False,'note':'This verifies storage and uploaded bytes, not field or production performance.'}
    output=OUT/'GitHub_WAFER_R01_업로드기록.json';output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'GitHub_WAFER_R01_업로드안내.md').write_text(f'''# 웨이퍼 R01 비공개 보관 확인

[저장소]({repo['html_url']}) · [웨이퍼 Release]({release['html_url']})

PRIVATE 상태, main/tag `{remote}`, 원격 blob {len(blobs)}개의 실제 바이트 해시, Release 첨부3개의 크기·SHA256을 확인했다. 기존 R04/R03 태그와 파일은 유지했고 루트 README·CURRENT_STATUS만 최신 웨이퍼 기록을 반영했다.

모델은 `WAFER_Surface_R01.zip`에 포함되며 Git clone에는 웨이퍼 가중치가 없다. 원본 사진·VOC XML·학습 라벨은 업로드하지 않았다. 현미경 공개자료 실험으로 D455 실측이나 물리 웨이퍼/lot 일반화를 검증한 결과는 아니다.
''',encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))

if __name__=='__main__':main()
