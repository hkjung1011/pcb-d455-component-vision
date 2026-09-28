"""Finalize the review's analysis definitions without changing frozen R04 core."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent/'r04'
DRAFT = BASE.parent/'r04_review/analysis_addendum_v1_1_draft/ANALYSIS_ADDENDUM_v1.1_DRAFT.md'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
    destination=BASE/'ANALYSIS_ADDENDUM_v1.1.md'
    if destination.exists(): raise RuntimeError('Final addendum already exists; do not overwrite')
    jobs=[read(p) for p in (ROOT/'reports/evaluation_suite/jobs').glob('*.json')]
    if any(j['binding']['split']=='test' for j in jobs): raise RuntimeError('Holdout already started')
    stamp=datetime.now(timezone.utc).isoformat()
    body=DRAFT.read_text(encoding='utf-8-sig')
    body=body.replace('# R04 분석 부록 v1.1 — 초안 (DRAFT · 미동결)', '# R04 분석 부록 v1.1 — 확정')
    body=body.replace('| 상태 | 초안. 사용자 확인 전이며 해시로 동결하지 않았다. |', f'| 상태 | 확정. 사용자 2026-09-29 진행 지시에 따라 기준을 결정함. 동결 UTC: {stamp}. |')
    body=body.replace('work/r04_review/analysis_addendum_v1_1/outputs/', 'work/r04_addendum/reports/')
    body=body.replace('work/r04_review/pre_holdout_audit/', 'work/r04_addendum/reports/')
    body=body.replace('계산량은 batch 4, 입력 1024, 1,165회로 두 팔이 같다. 라벨 노출량과 배율 구성까지 같게 맞춘 것은 아니다.', '명목 입력 예산은 batch 4, 입력 1024, 1,165회로 같다. 라벨 수에 따른 할당 연산·실제 실행 시간·메모리는 같다고 주장하지 않는다. 라벨 노출량과 배율 구성까지 같게 맞춘 것은 아니다.')
    body=body.replace('| 4대 이상 | 기존 200장 예시(train 3 · val 1 · test-A 1 · test-B 1) | test-A/B 각각 보고 |', '| 6대 이상 | 기존 200장 예시(train 3 · val 1 · test-A 1 · test-B 1) | test-A/B 각각 보고 |\n| 4–5대 | 보유 설계에 따라 train/val/test를 배정. 6대 예시를 그대로 적용하지 않음 | 확보된 독립 test 범위만 보고 |')
    body=body.replace('코드는 부록 확정 후 별도로 작성하고 검토한다.', '코드는 별도 새 파일로 구현한다. 사전 감사와 CPU 보조 분석을 통과한 뒤 기존 runner를 재개한다.')
    body += '\n\n## 실행 확정 메모 (2026-09-29)\n\n'
    body += '- 추가 seed 학습, 요인 분리, sampler 변경, 평가 v2와 stress v2는 이번 실행에 포함하지 않는다. 기존 학습을 재사용한다.\n'
    body += '- FP 소계 목록은 §3.6 그대로다. `jumper`, `testpoint`, `led`는 나머지에 둔다. source_type 원문별 집계가 주 보고다.\n'
    body += '- FP 판정용 v1 TP 매칭은 기존 코드와 같이 IoU 동률에서 **높은 GT 인덱스**를 우선한다. §3.5의 낮은 GT 인덱스 동률 규칙은 이미 FP인 검출의 보조 관계 설명에만 적용하며 TP 집합을 바꾸지 않는다.\n'
    body += '- 원안의 같은 계산량 주장은 명목 입력 예산으로 한정했고, train3/val1/testA1/testB1 예시의 최소 보드 수는 6대로 정정했다. 원본 초안은 보존한다.\n'
    body += '- 일부 R04 validation과 R03 holdout을 이미 본 뒤 작성한 보조 분석 계획이다. R04 holdout 결과는 이 문서 동결 시점에 생성되지 않았다.\n'
    body += '- 결과상 특정 지표가 높거나 낮았다는 관측은 수치와 함께 보고할 수 있다. 단일 seed에서 통계적으로 우월하거나 실사용 일반화가 입증됐다는 주장은 하지 않는다.\n'
    destination.write_text(body,encoding='utf-8')
    (BASE/'ANALYSIS_ADDENDUM_v1.1.sha256').write_text(sha(destination)+'  '+destination.name+'\n',encoding='ascii')
    files=['protocol.json','stress_protocol.json','scripts/evaluate_r04.py','scripts/run_r04_evaluation.py','scripts/audit_r04_results.py','scripts/train_r04.py','scripts/ignore_adapter.py','data/native_manifest.json','reports/evaluation_suite/protocol.json']
    record={'schema':'r04-analysis-addendum-freeze-v1.1','frozen_utc':stamp,'document':str(destination),'document_sha256':sha(destination),'source_draft_sha256':sha(DRAFT),'source_review':'Claude draft provided by user','authorization':'User instructed proceed now on 2026-09-29','holdout_jobs_at_freeze':0,'completed_val_jobs_at_freeze':sum(j['binding']['split']=='val' and j['status']=='completed' for j in jobs),'prior_seen':'R03 holdout and initial 22/40 R04 validation; supplemental definitions are not pre-results registration','core_sha256':{f:sha(ROOT/f) for f in files},'primary_selection_unchanged':True}
    (BASE/'freeze_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))
if __name__=='__main__': main()
