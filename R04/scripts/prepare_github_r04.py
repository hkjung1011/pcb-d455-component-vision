"""Prepare the authorized R04 private archive update; no network mutations."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

BASE=Path(__file__).resolve().parents[3]
OUT=BASE/'outputs'
SOURCE=OUT/'PCB_D455_R04'
DEST=BASE/'work/github_private_archive'
REPO='hkjung1011/pcb-d455-component-vision'
TAG='r04-2026-09-29'
RELEASE=f'https://github.com/{REPO}/releases/tag/{TAG}'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s.strip()+'\n',encoding='utf-8')
def save(p,x):write(p,json.dumps(x,ensure_ascii=False,indent=2))
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)

def main():
    if subprocess.check_output(['git','status','--porcelain'],cwd=DEST,text=True).strip():
        raise RuntimeError('Archive checkout has changes; inspect first')
    repo=json.loads(subprocess.check_output(['gh','repo','view',REPO,'--json','visibility'],text=True))
    if repo['visibility']!='PRIVATE':raise RuntimeError('Expected private archive repository')
    if (DEST/'R04').exists():raise RuntimeError('R04 Git view already exists; preserve it')
    original=read(SOURCE/'SHA256SUMS.json')
    expected={row['path'] for row in original}|{'SHA256SUMS.json'}
    actual={p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob('*') if p.is_file()}
    if actual!=expected:raise RuntimeError('Package contains unlisted or missing files')
    for row in original:
        p=SOURCE/row['path']
        if sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise RuntimeError(f'Package changed {p}')
    omitted=[];included=[]
    for p in sorted(SOURCE.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(SOURCE)
        row={'path':rel.as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
        if p.suffix=='.pt' or p.stat().st_size>40*1024*1024:
            omitted.append(row)
        else:
            copy(p,DEST/'R04'/rel);included.append(row)
    copy(SOURCE/'CURRENT_STATUS.json',DEST/'CURRENT_STATUS.json')
    for name in ['models.csv','test_metrics.csv','class_metrics.csv','label_weight_ledger.csv','group_sampling_ledger.csv']:
        copy(SOURCE/name,DEST/'metrics'/f'r04_{name}')
    copy(OUT/'PCB_D455_R04_검증기록.json',DEST/'archive/PCB_D455_R04_verification.json')
    status=read(SOURCE/'CURRENT_STATUS.json')
    selected=read(SOURCE/'selected_models.json')
    rows='\n'.join(f'| {r["arm"]} / {r["cohort"]} | {r["ap50_95_max100"]*100:.2f}% | {r["ap50_95_max300"]*100:.2f}% | {r["precision"]*100:.2f}% / {r["recall"]*100:.2f}% |' for r in status['metrics'])
    chosen=selected['arms'][selected['recommended_arm']]
    write(DEST/'README.md',f'''# Raspberry Pi PCB · D455 인식 개발 기록

**2026-09-29 · 최신 R04 · 비공개 연구 기록**

Raspberry Pi는 촬영·인식 대상이다. Claude 검토를 원본·코드와 대조하고, RTX5060 Laptop8GB에서 기준/개선 YOLO11s를 같은 예산으로 학습했다. 각 모델 최대20에폭, 실제 optimizer1,165회이며 마지막epoch는일부만진행했다. R03/R02와 과거 Release는 보존했다.

## 최신 실제 지표

| 모델 / 개발 holdout | bbox mAP50–95@100 | bbox mAP50–95@300 | 고정 confidence P / R |
|---|---:|---:|---:|
{rows}

@100/@300은 이미지 전체 최대 1,000개 검출을 보존한 뒤 적용한 COCO 영상·클래스별 검출 한도다. 추천은 **{selected['recommended_arm']}, epoch{chosen['epoch']}, {chosen['pipeline']}, confidence{chosen['confidence']:.2f}**로 검증셋에서만 결정했다. 40개 checkpoint·추론조합을 검증하고 선택 SHA를 고정한 뒤 일반11장/5그룹, Pi3B2장/1그룹을 평가했다. 이 그룹은 학습·검증에서 제외했지만 R03에서 이미 오류를 살펴본 **개발 holdout**이며 새 최종시험은 아니다.

신규 D455 사진0장, 신규 사람이 만든 라벨0개, 신규target4 mask0개다. **D455 실사용 성능과 저항 포함4종 부품 instance segmentation은 미검증/미완료**다. 공개자료 점수나 축소영상 점수를 카메라 실측으로 표현하지 않는다.

## R04에서 확인하고 바꾼 내용

- 원본 목표5,896개 중 학습3,307개·검증1,052개·일반시험1,348개·Pi시험189개. 새 라벨과 crop 반복 노출을 구분했다.
- 기준418view 대비 개선968view로 큰 부품 전체를 담았다. 완전포함원본부품3,215→3,307개.
- 50%미만 목표조각·unknown 영역에는 실제 음성 classification loss 제외를 구현했다. 양성 GT와 box/DFL은 유지했다.
- 두 arm 동일분할·공식초기가중치·그룹가중샘플링·FP32·batch4. 새val의원본8–16px목표98개. 클래스 loss각1.0, 그룹동일기대확률, 그룹내IC/connectorview1.5배우선순위.
- native/dual 검증과 confidence를 동결하고 저장 예측을 독립 재계산했다. 실제draw·직접optimizer호출·parameterstep·선택checkpoint를 별도 기록했다.

## 자료 찾아보기

- [R04 결과와 알고리즘·그래프](R04/README.md) · [Claude 주장별 반영/정정](R04/reports/CLAUDE_REVIEW_RESPONSE.md)
- [현재 상태](CURRENT_STATUS.json) · [모델/예산 CSV](metrics/r04_models.csv) · [성능 CSV](metrics/r04_test_metrics.csv) · [종류별 결과](metrics/r04_class_metrics.csv)
- [데이터 독립 감사](R04/reports/data_independent_audit.md) · [최종 결과 감사](R04/reports/final_results_audit.md)
- [고정 선택](R04/selected_models.json) · [모든 검증·시험 지표](R04/reports/evaluation_suite/aggregate.json) · [오프라인 추론](R04/INFERENCE.md)
- [R03 공개자료4개모델](R03/README.md) · [R03 Release](https://github.com/{REPO}/releases/tag/r03-2026-09-28) · [R02 보조semantic기록](history/PCB_D455_R02_개발결과.md)

모델20개 후보와 초기 가중치는 [R04 비공개 Release]({RELEASE})의 `PCB_D455_R04.zip`에 있다. Git은 가중치를 제외한 탐색용 기록이며 clone만으로 추론 가중치가 복원되지 않는다. 원본 데이터 사진·타일이미지·다운로드ZIP·가상환경·인증 설정은 업로드하지 않았다. metadata의 절대경로는 당시 실행 증거이며 새PC에서는 자료와 경로 설정이 필요하다.

다음은 실제D455 30장광학파일럿과 별도200장수집·실물/세션분할·검수된4종bbox/mask다. 이 수량은 계획이며 확보된 실적이 아니다. 출처와 라이선스는원자료/R03기록을따르며 비공개보관이추가재배포권한을부여하지않는다.
''')
    old=(DEST/'CHANGELOG.md').read_text(encoding='utf-8')
    entry='''## R04 · 2026-09-29

두 번째 Claude 검토의 타일 잘림·GT 정의·실제 optimizer count·validation 분포·카메라 픽셀 가정을 원본과 대조했다. 동일 조건의 기준/개선 YOLO11s 각각20에폭이내/1,165회 직접갱신을 수행하고40개val후보의고정선택 후 개발holdout을 평가했다. 원본시험GT와R03Release는보존했다. D455실물/target4mask는미완료이며30+200장 계획을 기록했다.

'''
    write(DEST/'CHANGELOG.md',old.replace('# 진행 이력\n\n','# 진행 이력\n\n'+entry,1))
    verification=read(OUT/'PCB_D455_R04_검증기록.json')
    save(DEST/'ARCHIVE_MANIFEST_R04.json',{'snapshot':'R04','r04_git_included':included,'r04_release_only':omitted,
         'release':RELEASE,'zip_bytes':verification['zip_bytes'],'zip_sha256':verification['zip_sha256'],
         'raw_source_photos_uploaded':False,'prior_archive_manifest':'ARCHIVE_MANIFEST.json'})
    write(DEST/'archive/R04_SHA256SUMS.txt',f"{verification['zip_sha256']}  PCB_D455_R04.zip\n{sha(OUT/'PCB_D455_R04_검증기록.json')}  PCB_D455_R04_verification.json")
    notes=f'''# R04 · Claude 검토 반영과 동일예산 비교

기준/개선 YOLO11s 각20에폭이내·실제optimizer1,165회. 새groupedvalidation으로40후보선택을동결하고학습미사용일반/Pi개발holdout평가를완료했다.

| 모델 / 개발 holdout | bbox mAP50–95@100 | bbox mAP50–95@300 | 고정 confidence P / R |
|---|---:|---:|---:|
{rows}

D455실물/target4instance mask는미완료. R03에서이미살펴본holdout이며미관측최종시험이아니다. 신규사진/사람라벨/mask모두0. 원본사진/타일이미지/가상환경은포함하지않는다.

`PCB_D455_R04.zip`에는모델후보20개·초기가중치·코드·label/weight기록·평가·그래프·검토표가포함된다. R03Release는유지한다.

SHA256 `{verification['zip_sha256']}`
'''
    write(DEST/'RELEASE_NOTES_R04.md',notes)
    print(json.dumps({'git_files':len(included),'release_only':len(omitted),'release':RELEASE}))

if __name__=='__main__':main()
