"""Read-only GitHub verification after the authorized R04 archive publish."""
from pathlib import Path
import hashlib
import json
import subprocess
from datetime import datetime, timezone

BASE=Path(__file__).resolve().parents[3]
CHECKOUT=BASE/'work/github_private_archive'
OUT=BASE/'outputs'
REPO='hkjung1011/pcb-d455-component-vision'
TAG='r04-2026-09-28'

def command(args,cwd=None):return subprocess.check_output(args,cwd=cwd,text=True,encoding='utf-8')
def api(endpoint):return json.loads(command(['gh','api',endpoint]))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    repo=api(f'repos/{REPO}')
    if not repo['private']:raise RuntimeError('Repository is not PRIVATE')
    local=command(['git','rev-parse','HEAD'],CHECKOUT).strip()
    remote=api(f'repos/{REPO}/commits/main')['sha']
    tag=api(f'repos/{REPO}/git/ref/tags/{TAG}')['object']
    tag_sha=tag['sha']
    if tag['type']=='tag':tag_sha=api(f'repos/{REPO}/git/tags/{tag_sha}')['object']['sha']
    if local!=remote or tag_sha!=local:raise RuntimeError('Local/main/tag mismatch')
    tree=api(f'repos/{REPO}/git/trees/{remote}?recursive=1')
    if tree.get('truncated'):raise RuntimeError('Remote tree was truncated')
    mismatch=[];count=0
    for item in tree['tree']:
        if item['type']!='blob':continue
        count+=1
        path=CHECKOUT/item['path']
        if not path.is_file():mismatch.append(item['path']);continue
        data=path.read_bytes()
        git_sha=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        if git_sha!=item['sha']:mismatch.append(item['path'])
    if mismatch:raise RuntimeError(f'Remote file mismatches {mismatch}')
    release=api(f'repos/{REPO}/releases/tags/{TAG}')
    if release['draft'] or not release['prerelease']:raise RuntimeError('Expected published prerelease record')
    paths={'PCB_D455_R04.zip':OUT/'PCB_D455_R04.zip',
           'PCB_D455_R04_verification.json':OUT/'PCB_D455_R04_검증기록.json',
           'R04_SHA256SUMS.txt':CHECKOUT/'archive/R04_SHA256SUMS.txt'}
    assets=[]
    for name,path in paths.items():
        matches=[a for a in release['assets'] if a['name']==name]
        if len(matches)!=1:raise RuntimeError(f'Missing/duplicate release asset {name}')
        asset=matches[0]
        if asset['size']!=path.stat().st_size or asset.get('digest')!='sha256:'+sha(path):
            raise RuntimeError(f'Asset size/digest mismatch {name}')
        assets.append({'name':name,'bytes':asset['size'],'sha256':sha(path),'url':asset['browser_download_url']})
    # R03 history must still resolve to its original immutable snapshot.
    old=api(f'repos/{REPO}/git/ref/tags/r03-2026-09-28')['object']
    old_sha=old['sha']
    if old['type']=='tag':old_sha=api(f'repos/{REPO}/git/tags/{old_sha}')['object']['sha']
    if old_sha!='fe1c2949a6d1d9f92b3d68ec577acd8d5822524c':raise RuntimeError('R03 tag changed')
    data={'status':'VERIFIED_PRIVATE_REMOTE_AND_RELEASE','utc':datetime.now(timezone.utc).isoformat(),
          'repo':repo['html_url'],'private':True,'main_commit':remote,'tag':TAG,'tag_commit':tag_sha,
          'remote_blob_count':count,'all_remote_blobs_match_local':True,'r03_tag_unchanged':True,
          'release':release['html_url'],'release_prerelease':True,'assets':assets,
          'd455_verified':False,'note':'Upload verification does not establish camera or production performance.'}
    target=OUT/'GitHub_R04_업로드기록.json'
    target.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'GitHub_R04_업로드안내.md').write_text(f'''# R04 비공개 보관 확인

[GitHub 저장소]({repo['html_url']}) · [R04 Release]({release['html_url']})

PRIVATE 상태, main/tag `{remote}`, 원격 파일 {count}개 blob SHA, 첨부 {len(assets)}개 SHA256을 확인했다. R03 태그와 Release는 보존했다.

최신 결과는 [R04 보고서](PCB_D455_R04/README.md), 전체 모델·증거는 `PCB_D455_R04.zip`에 있다. Git에는 가중치가 생략되어 Release ZIP에서 받아야 한다. 원본 데이터 사진·가상환경·인증정보는 포함하지 않았다.

일반/Pi 결과는 공개 개발 holdout 성능이며 D455 실물 또는4종 instance mask 검증은 아니다. 상세 원격 검증은 [JSON](GitHub_R04_업로드기록.json)에 있다.
''',encoding='utf-8')
    print(json.dumps(data,ensure_ascii=False))

if __name__=='__main__':main()
