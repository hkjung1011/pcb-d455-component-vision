"""Copy addendum evidence into the existing report package; no core mutation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,shutil

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent/'r04'
DEST=ROOT/'reports/analysis_addendum_v1_1'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    if DEST.exists():raise RuntimeError('Evidence destination already exists; do not overwrite')
    for name in ['pre_holdout_audit','supplementary_val','supplementary_holdout']:
        record=read(HERE/'reports'/f'{name}.json')
        if not record['status'].startswith('PASS'):raise RuntimeError('Missing successful '+name)
    frozen=read(HERE/'freeze_record.json')
    for rel,expected in frozen['core_sha256'].items():
        if sha(ROOT/rel)!=expected:raise RuntimeError('Frozen core changed: '+rel)
    DEST.mkdir(parents=True)
    pairs=[]
    for p in HERE.iterdir():
        if p.is_file() and p.suffix in {'.md','.json','.sha256','.py','.log'}:pairs.append((p,DEST/p.name))
    for p in (HERE/'reports').iterdir():
        if p.is_file():pairs.append((p,DEST/'outputs'/p.name))
    draft=HERE.parent/'r04_review/analysis_addendum_v1_1_draft'
    for name in ['ANALYSIS_ADDENDUM_v1.1_DRAFT.md','REVIEW_APPENDIX.md']:
        pairs.append((draft/name,DEST/name))
    for p in (draft/'repro').rglob('*'):
        if p.is_file() and p.suffix in {'.py','.ps1','.json','.md'}:pairs.append((p,DEST/'repro'/p.relative_to(draft/'repro')))
    ledger=[]
    for source,destination in pairs:
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        if sha(source)!=sha(destination):raise RuntimeError('Copy mismatch')
        ledger.append({'path':str(destination.relative_to(DEST)).replace('\\','/'),'source_path':str(source),'sha256':sha(destination),'bytes':destination.stat().st_size})
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_ANALYSIS_EVIDENCE_COPY','files':ledger,'core_hashes_unchanged':True,'source_images_included':False,'analysis_selection_unchanged':True}
    (DEST/'EVIDENCE_INVENTORY.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    (DEST/'README.md').write_text('# R04 분석 부록 v1.1 실행 기록\n\n'
        '사용자 진행 지시에 따라 Claude 초안을 확정하고, 기존 R04 학습·평가 프로토콜은 유지했다.\n\n'
        '- `ANALYSIS_ADDENDUM_v1.1.md`, `freeze_record.json`: holdout 전 분석 규칙·해시·시각. 일부 val/R03 holdout을 본 뒤 작성한 계획이다.\n'
        '- `outputs/pre_holdout_audit.json`: 학습·val40·선택 동결을 독립 CPU 재계산한 holdout 전 감사.\n'
        '- `outputs/supplementary_val.*`: 그룹별/그룹 제외 순위·20쌍 AP·FP 분류. 주 선택을 변경하지 않았다.\n'
        '- `outputs/supplementary_holdout.*`: 개발 holdout 그룹·클래스/크기와 FP 진단.\n'
        '- `repro/`, `REVIEW_APPENDIX.md`: 사용자가 전달한 Claude 재현 자료를 원형 보존했다.\n\n'
        '이 폴더의 Python 파일은 실행 당시 소스의 바이트 사본이다. 재현은 원래 workspace의 `work/r04_addendum/` 배치와 `work/r04`/`work/r03` 로컬 데이터·가중치를 전제로 한다. 이미 완료된 holdout을 다시 실행하거나 결과를 덮어쓰는 용도가 아니다. 원본 사진은 포함하지 않는다.\n',encoding='utf-8')
    print(json.dumps({'status':record['status'],'files':len(ledger),'destination':str(DEST)}))
if __name__=='__main__':main()
