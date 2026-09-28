"""Archive only the current paused metadata in the existing private Git checkout."""
from pathlib import Path
import csv
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]
REPO=BASE/'work/github_private_archive'
def copy(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
def main():
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise RuntimeError('Inspect pre-existing Git changes first')
    snapshot=json.loads((ROOT/'PAUSED_STATE.json').read_text(encoding='utf-8'))
    destination=REPO/'R04_PAUSED'
    if destination.exists():raise RuntimeError('Paused record already exists')
    copy(ROOT/'PAUSED_STATE.json',destination/'snapshot.json')
    copy(BASE/'outputs/PCB_D455_R04_재개안내.md',destination/'RESUME.md')
    copy(BASE/'outputs/Resume-PCB-R04.ps1',destination/'Resume-PCB-R04.ps1')
    for name in ['protocol.json','stress_protocol.json']:
        copy(ROOT/name,destination/name)
    for arm in ['baseline','improved']:
        for name in ['training_summary.json','start.json','epochs.json','candidates.json','sampled_exposures.json']:
            copy(ROOT/'runs'/arm/name,destination/'runs'/arm/name)
    for name in ['CLAUDE_REVIEW_RESPONSE.md','MODEL_FAMILY.md','data_independent_audit.md','data_independent_audit.json','ignore_adapter_tests.json','dense_batch_smoke.json']:
        copy(ROOT/'reports'/name,destination/'reports'/name)
    for source in (ROOT/'scripts').glob('*.py'):
        copy(source,destination/'scripts'/source.name)
    rows=[]
    for item in snapshot['completed_jobs']:
        path=ROOT/'reports/evaluation_suite/evaluations'/item['job']/'metrics.json'
        metric=json.loads(path.read_text(encoding='utf-8'))
        copy(path,destination/'validation'/item['job']/'metrics.json')
        rows.append({'candidate':item['job'],'split':'val','ap50_95_max100':metric['bbox']['ap50_95_max100'],
                     'ap50_95_max300':metric['bbox']['ap50_95_max300'],
                     'notice':'Partial validation only; final selection and holdout evaluation not completed'})
    with (destination/'partial_validation.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    status={'revision':'R04_PAUSED','status':'TRAINING_COMPLETE_VALIDATION_22_OF_40_USER_PAUSED',
            'saved_utc':snapshot['saved_utc'],'trained_arms':2,'direct_optimizer_calls_each':1165,
            'checkpoints_saved_locally':20,'checkpoints_uploaded_in_this_pause_record':False,
            'validation_complete':22,'validation_planned':40,'final_selection_frozen':False,
            'new_general_pi_holdout_completed':False,'d455_verified':False,'target4_instance_masks_trained':False,
            'last_completed_release':'r03-2026-09-28','resume_record':'R04_PAUSED/RESUME.md',
            'note':'This is a metadata/code/partial-metric record. R04 weights and raw data remain on the local PC; no final R04 release yet.'}
    (REPO/'CURRENT_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    old=(REPO/'README.md').read_text(encoding='utf-8')
    banner='''# R04 중간 저장 · 2026-09-28

**사용자 요청으로 PC 종료 전 중단·저장했습니다.** 기준/개선 YOLO11s 학습은 각각20에폭이내·실제1,165회갱신으로완료됐고, 검증은22/40조합까지완료했습니다. 최종선택·일반/Pi시험은아직미완료입니다.

[중간 상태·재개 안내](R04_PAUSED/RESUME.md) · [저장 검증](R04_PAUSED/snapshot.json) · [현재까지 검증 지표](R04_PAUSED/partial_validation.csv) · [검토 반영](R04_PAUSED/reports/CLAUDE_REVIEW_RESPONSE.md)

이 중간 기록에는 코드·설정·학습상태·검증지표가 있습니다. **R04 가중치20개와 원본자료는 로컬PC에 저장되어 있으며, 이번 중간 Git 기록에는 업로드하지 않았습니다.** 재부팅 후 학습을 다시 하지 않고 남은 평가부터 이어갑니다. 아래 R03는 이전에 완료한 기록입니다.

---

'''
    (REPO/'README.md').write_text(banner+old,encoding='utf-8')
    print(json.dumps({'paused_metadata_files':sum(p.is_file() for p in destination.rglob('*')),'weights_uploaded':False}))
if __name__=='__main__':main()
