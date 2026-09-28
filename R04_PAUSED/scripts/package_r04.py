"""Package verified R04 evidence, models and plots without source PCB photographs."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import zipfile
import importlib.metadata
import platform
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[1]
OUT = BASE/'outputs'
DEST = OUT/'PCB_D455_R04'
NAMES = ['resistor', 'capacitor', 'ic', 'connector']

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(x.strip()+'\n', encoding='utf-8')
def copy(p, q):
    q.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, q)
def percent(x): return '해당 없음' if x is None else f'{100*x:.2f}%'


def main():
    aggregate = read(ROOT/'reports/evaluation_suite/aggregate.json')
    audit = read(ROOT/'reports/final_results_audit.json')
    if aggregate['status'] != 'COMPLETED_DEVELOPMENT_HOLDOUT_NOT_D455_VALIDATED':
        raise RuntimeError('Evaluation is not complete')
    if not str(audit['status']).startswith('PASS'):
        raise RuntimeError('Final evidence audit must pass')
    if DEST.exists():
        raise RuntimeError('Existing R04 package must be preserved; do not overwrite')
    figures = read(ROOT/'reports/figures/figure_inventory.json')
    # A plot allowlist prevents a later inference preview from accidentally
    # redistributing WACV source pixels with the archived reports.
    allowed_plot_hashes = {a['file']:a['sha256'] for figure in figures['figures']
                           for a in figure['artifacts'].values()}
    if len(figures['figures']) != 9:
        raise RuntimeError('All nine actual-result figures must be generated before packaging')
    allowed_plot_names = set(allowed_plot_hashes)
    for name, expected in allowed_plot_hashes.items():
        if sha(ROOT/'reports/figures'/name) != expected:
            raise RuntimeError('Plot differs from reviewed figure inventory')
    media_extensions = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp','.gif','.svg','.mp4','.avi'}
    for p in (ROOT/'reports').rglob('*'):
        if p.is_file() and p.suffix.lower() in media_extensions:
            if p.parent != ROOT/'reports/figures' or p.name not in allowed_plot_names:
                raise RuntimeError(f'Unreviewed media in reports: {p}')
    DEST.mkdir(parents=True)
    for folder in ['scripts', 'reports', 'runs']:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.tmp'}:
                copy(p, DEST/p.relative_to(ROOT))
    # All metadata/YOLO annotations, but no raw photographs, materialized crops or source archive.
    for p in sorted((ROOT/'data').rglob('*')):
        if p.is_file() and 'images' not in p.relative_to(ROOT/'data').parts and p.suffix in {'.json','.yaml','.txt'}:
            copy(p, DEST/p.relative_to(ROOT))
    for p in sorted((ROOT.parent/'r04_review').iterdir()):
        if p.is_file() and p.suffix in {'.json','.md','.py','.log'}:
            copy(p, DEST/'review'/p.name)
    for arm in ['baseline','improved']:
        for name in ['args.yaml', 'results.csv']:
            copy(ROOT/'training_logs'/arm/name, DEST/'runs'/arm/name)
    copy(ROOT/'protocol.json', DEST/'protocol.json')
    if (ROOT/'stress_protocol.json').exists():
        copy(ROOT/'stress_protocol.json', DEST/'stress_protocol.json')
    protocol = read(ROOT/'protocol.json')
    copy(Path(protocol['initial_weights']), DEST/'models/yolo11s_initial.pt')
    registry = {'schema':'r04-inference-selection-v1', 'recommended_arm':aggregate['recommended_arm'],
                'selection_sha256':aggregate['selection_sha256'], 'selected_on':'validation_only',
                'd455_verified':False, 'component_instance_masks_trained':False, 'arms':{}}
    model_rows, metrics_rows, comparison_rows, class_rows = [], [], [], []
    label_rows, group_rows = [], []
    data=read(ROOT/'data/summary.json')
    for arm, payload in aggregate['arms'].items():
        selected = payload['selection']
        candidate = selected['candidate']
        source_weights = Path(candidate['weights'])
        relative = source_weights.relative_to(ROOT).as_posix()
        registry['arms'][arm] = {'checkpoint':relative, 'sha256':candidate['checkpoint_sha256'],
                                'pipeline':candidate['pipeline'], 'confidence':selected['operating_confidence'],
                                'epoch':candidate['epoch'], 'optimizer_calls':candidate['optimizer_calls'],
                                'imgsz':1024, 'tile_overlap':.2, 'tile_nms_iou':.5}
        if sha(DEST/relative) != candidate['checkpoint_sha256']:
            raise RuntimeError('Packaged checkpoint mismatch')
        summary = read(ROOT/'runs'/arm/'training_summary.json')
        exposures = read(ROOT/'runs'/arm/'sampled_exposures.json')
        view_summary=next(s for s in data['arms'] if s['arm']==arm)
        for cid,name in enumerate(NAMES):
            label_rows.append({'arm':arm,'class':name,'original_train_bbox':data['split']['train']['class_counts'][name],
                'stored_view_label_exposures':view_summary['supervised_label_exposures'][name],
                'actual_sampled_label_exposures':exposures['class_exposures'].get(str(cid),0),
                'additional_class_loss_multiplier':1.0,'new_human_labels':0,
                'note':'View/sample exposures include repeated original instances; not unique labels or gradient-share weights'})
        for group,count in sorted(exposures['groups'].items()):
            group_rows.append({'arm':arm,'group':group,'actual_draws':count,
                'actual_draw_fraction':count/exposures['draws'],'expected_group_probability':1/17,
                'within_group_ic_connector_view_priority':1.5})
        model_rows.append({'arm':arm,'epoch_reached':summary['actual_epochs_reached'],
                           'last_epoch_partial':summary['last_epoch_partial'],
                           'direct_optimizer_calls':summary['actual_direct_optimizer_calls'],
                           'actual_draws':exposures['draws'],'parameters':read(ROOT/'runs'/arm/'start.json')['parameters'],
                           'selected_epoch':candidate['epoch'], 'selected_optimizer_calls':candidate['optimizer_calls'],
                           'pipeline':candidate['pipeline'], 'confidence':selected['operating_confidence'],
                           'checkpoint_sha256':candidate['checkpoint_sha256']})
        for cohort, metric in payload['tests'].items():
            row = {'arm':arm, 'cohort':cohort, 'images':metric['images'], 'groups':metric['groups'],
                   'gt':metric['native_gt_instances'], 'ap50_95_max100':metric['bbox']['ap50_95_max100'],
                   'ap50_95_max300':metric['bbox']['ap50_95_max300'], 'ap50_max300':metric['bbox']['ap50_max300'],
                   'precision':metric['operating']['precision'], 'recall':metric['operating']['recall'],
                   'confidence':selected['operating_confidence'], 'tp':metric['operating']['tp'],
                   'fp':metric['operating']['fp'], 'fn':metric['operating']['fn'],
                   'scope':'development_holdout_not_pristine_final', 'd455_verified':False}
            metrics_rows.append(row)
            comparison_rows.append(f'| {arm} / {cohort} | {metric["images"]} / {metric["native_gt_instances"]} | {percent(row["ap50_95_max100"])} | {percent(row["ap50_95_max300"])} | {percent(row["precision"])} / {percent(row["recall"])} |')
            for name,(tp,fp,fn) in zip(NAMES,metric['operating']['per_class_counts_tp_fp_fn']):
                class_rows.append({'arm':arm,'cohort':cohort,'class':name,'tp':tp,'fp':fp,'fn':fn,
                                   'total':tp+fn,'recall':tp/(tp+fn) if tp+fn else None})
    save(DEST/'selected_models.json',registry)
    if read(ROOT/'selected_models.json')!=registry or sha(ROOT/'selected_models.json')!=sha(DEST/'selected_models.json'):
        raise RuntimeError('Packaged registry differs from the tested offline inference registry')
    for name, rows in [('models',model_rows),('test_metrics',metrics_rows),('class_metrics',class_rows),
                       ('label_weight_ledger',label_rows),('group_sampling_ledger',group_rows)]:
        with (DEST/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    status={'revision':'R04','utc':datetime.now(timezone.utc).isoformat(),
            'state':'MATCHED_PUBLIC_SOURCE_TRAINING_EVALUATION_COMPLETE',
            'recommended_arm':aggregate['recommended_arm'],'recommended_by':'VALIDATION_ONLY',
            'd455_verified':False,'new_d455_photos':0,'new_human_labels':0,'new_target4_masks':0,
            'component_instance_masks_trained':False,'production_readiness':'NOT_ESTABLISHED',
            'test_scope':'Previously inspected development holdouts; no fresh final test',
            'training_models':model_rows,'metrics':metrics_rows,
            'historical_tasks':'R03 board detector/board mask and parts n/s, R02 source3 semantic retained separately'}
    save(DEST/'CURRENT_STATUS.json',status)
    chosen=aggregate['arms'][aggregate['recommended_arm']]['selection']
    pi=aggregate['arms'][aggregate['recommended_arm']]['tests']['pi_test']
    model_table='\n'.join(f'| {r["arm"]} | {r["epoch_reached"]}번째 진입 / 마지막 일부 | {r["direct_optimizer_calls"]:,} | {r["actual_draws"]:,} | {r["selected_epoch"]} / {r["selected_optimizer_calls"]} | {r["pipeline"]} / {r["confidence"]:.2f} |' for r in model_rows)
    label_table='\n'.join(f'| {split} | {data["split"][split]["images"]} | {data["split"][split]["groups"]} | '+ ' / '.join(str(data['split'][split]['class_counts'].get(n,0)) for n in NAMES)+' |' for split in ['train','val','test','pi_test'])
    pi_table='\n'.join(f'| {r["arm"]} | {r["class"]} | {r["tp"]}/{r["total"]} | {percent(r["recall"])} |' for r in class_rows if r['cohort']=='pi_test')
    write(DEST/'README.md',f'''# PCB D455 · R04 검토 반영과 비교 실험

**공개자료에서 기준/개선 YOLO11s 두 모델을 실제 학습·평가했다. 검증셋 기준 선택은 `{aggregate['recommended_arm']}`, {chosen['candidate']['epoch']}번째 epoch, `{chosen['candidate']['pipeline']}`, confidence {chosen['operating_confidence']:.2f}다.** Raspberry Pi 개발 holdout에서 AP50–95@300은 {percent(pi['bbox']['ap50_95_max300'])}, 고정 confidence recall은 {percent(pi['operating']['recall'])}다. 실제 D455와 저항 포함 4종 부품 mask는 아직 검증하지 못했다.

## 성능과 평가 범위

| 모델 / 시험 | 사진 / GT | bbox mAP50–95@100 | bbox mAP50–95@300 | 고정 confidence P / R |
|---|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

@100/@300은 COCO의 영상·클래스별 최대 검출 수다. 원본 좌표와 원본 target4 정답을 그대로 사용했다. 운영 P/R은 box IoU≥0.5와 검증셋에서 고정한 전역 confidence 기준이다. AP는 낮은 confidence부터의 점수 곡선이므로 운영 P/R과 목적이 다르다.

일반 holdout은 11장·5그룹, Raspberry Pi는 3B 앞/뒤 2장·1그룹이다. 학습·검증에는 이 그룹이 들어가지 않았지만 R03에서 결과를 이미 살펴보고 R04 설계를 개선했으므로 **미관측 최종 시험이 아닌 개발 holdout**이다. 물리적으로 서로 다른 실물인지 증명하지 못했고 새 설계·D455·전체 BOM 성능으로 확장하지 않는다. Pi 1그룹에는 일반화 신뢰구간을 붙이지 않는다. 일반 holdout의 그룹 bootstrap은 고정 threshold recall의 기술적 불확실성만 나타낸다.

## 검토 후 바뀐 점

기존 native 타일에서 부품이 잘리는 문제를 확인했다. 같은 새 train/val 분할, 초기 공식 가중치, 증강, 그룹 샘플링과 optimizer 예산으로 두 실험을 구성했다.

- **baseline:** native1024 타일 418개. 보이는 목표 조각을 모두 양성으로 사용한다. 전체 모습이 있는 원본 부품은 3,215/3,307개다.
- **improved:** native1024 + context2048 + 부품 중심 crop 총968개. 원본3,307개 모두 완전한 모습의 view가 있다. 50% 미만 목표 조각2,722회와 unknown1,399회의 영역에서는 음성 classification loss를 실제로 제외한다. 양성 GT/할당 anchor와 box/DFL loss는 유지한다.
- 알려진 비대상은 4종 closed-set 배경으로 유지했다. 회색 픽셀 채움이나 라벨 삭제만으로 ignore 처리를 했다고 기록하지 않는다.
- 이전 train+val 안에서만 그룹을 다시 나눴다. 8–16px 검증 GT는 24/875에서98/1,052로 늘었다. 이 수치가 D455 대표성을 보장하지 않는다.
- 두 모델의 2에폭 간격10개 checkpoint × native/dual을 모두 검증하고, AP300→AP100→지연→ID 순으로 선택했다. 선택과 confidence의 SHA를 고정한 후 두 holdout을 각각 평가했다.

여러 변경을 함께 적용한 비교이며 타일/ignore 각각의 단독 효과나 통계적 우월성을 증명하는 실험은 아니다. R03는 분할·샘플링·batch·증강이 달라 직접적인 동일조건 기준선으로 쓰지 않는다. R03 공개 점수와 원본 GT를 고치지 않고 보존했다.

## 라벨 수와 가중치

| 분할 | 원본 사진 | 이름 기준 그룹 | 저항 / 커패시터 / IC / 커넥터 |
|---|---:|---:|---|
{label_table}

새 사람이 만든 라벨은0개, 새 mask는0개다. 기존 공개 bbox5,896개를 재사용했고 그중 학습원본3,307개를 여러 crop으로 노출했다. 영상 annotation, crop 중복 노출, 실제 draw 노출을 구분해 저장했다.

클래스 loss multiplier는 각각1.0, loss gain은 box7.5/cls0.5/dfl1.5다. 17개 학습 보드그룹은 기대 샘플 확률이 각각1/17이며 그룹 내 IC/커넥터 포함 view에1.5배 우선순위를 줬다. 그룹별 weight 합을1로 정규화하고 epoch당448회 복원 추출했다. 이는 클래스 gradient 비율을1.5배로 보장하지 않는다. 실제 draw와 클래스 노출은 `runs/*/sampled_exposures.json`, 모든 view의 weight는 `data/*/views.json`에 있다.

| 모델 | 학습 epoch | 실제 optimizer 호출 | 실제 view draw | 선택 epoch / 호출 | 추론 / confidence |
|---|---|---:|---:|---:|---|
{model_table}

각 모델 최대20에폭·1,165회 직접 optimizer 호출에서 멈췄다. 20번째 epoch는90/112배치까지만 처리했다. AdamW parameter별 step과 직접 호출 수를 별도로 기록했다. FP32, 1024px, batch4/nbs8, AdamW lr0.001/lrf0.01, warmup2, seed42, RTX5060 Laptop8GB를 사용했다. 3에폭은 동작 점검에 유용하지만 실사용 성능을 입증하는 기준이 아니다.

## Raspberry Pi 클래스별 인식

| 모델 | 종류 | 찾은 GT / 전체 | Recall |
|---|---|---:|---:|
{pi_table}

원본 connector 정의에는 미실장 RUN 패드 annotation2개가 포함된다. 점수를 높이려고 기존 GT에서 삭제하지 않았다. 새 D455 실장부품 정의는 별도 annotation 버전과 검수로 관리한다. unknown 객체가 있는 데이터이므로 오검출의 의미와 전체 부품 coverage에 한계가 있다.

## 알고리즘과 실제 그래프

[그래프 목록](reports/figures/README.md) · [Claude 검토 반영표](reports/CLAUDE_REVIEW_RESPONSE.md) · [원본 데이터 독립 감사](reports/data_independent_audit.md) · [최종 결과 감사](reports/final_results_audit.md)

[모델별 구조와 역할](reports/MODEL_FAMILY.md) · [어떤 부품을 찾고 놓쳤는지](reports/error_analysis.md) · [실행 코드의 고정 예측 재현 확인](reports/inference_equivalence.json)

[합성 해상도 민감도 시험](reports/resolution_stress/README.md)은 별도 사전 고정 조건의 진단이다. 사진별 배율 근사값으로 축소했으며 실제 D455 광학·초점·노이즈를 재현한 결과나 성능 하한이 아니다.

[모델·예산 CSV](models.csv) · [전체 지표 CSV](test_metrics.csv) · [클래스 지표 CSV](class_metrics.csv) · [검증 선택/평가 원본](reports/evaluation_suite/aggregate.json)

[원본 라벨·반복 노출·클래스 가중치 장부](label_weight_ledger.csv) · [그룹별 확률과 실제 학습 횟수](group_sampling_ledger.csv)

## 실사용 준비와 재현

[오프라인 실행법](INFERENCE.md)의 고정 모델·추론 파이프라인으로 새 이미지를 처리할 수 있다. 실제 D455 연결이나 실시간 검증을 수행한 것은 아니다. 합성 축소를 D455 성능 하한이라고 부르지 않는다. Depth Min-Z와 RGB 초점/부품 판독성을 구분하며, 실제 영상의 intrinsics·거리·초점·조명·원본 부품 픽셀을 측정해야 한다.

다음은 실제D455 30장 파일럿, 조건부200장 수집·원본4종 bbox/instance mask 검수, 실물/세션별 독립분할이다. 수량은 계획이며 아직 확보한 데이터가 아니다. 한 대만 있으면 새로운 실물 일반화가 아닌 미관측 촬영세션 평가로 명시한다. 상세 조건과 예산은 검토 반영표에 있다.

이 패키지는 체크포인트20개, 공식 초기 가중치, 코드·라벨·weight·평가·그래프를 담는다. 원본 WACV 사진·타일이미지·다운로드 ZIP·가상환경·인증정보는 포함하지 않는다. 학습 재현에는 원 출처 자료와 기록된 환경이 필요하다. metadata의 절대경로는 실행 당시 증거이며 다른 PC에서 자동 유효하지 않다. 오프라인 추론의 선택 registry는 패키지 상대경로를 쓴다.

R03/R02 기록과 GitHub 이전 Release는 유지한다. 공개자료·Ultralytics 코드/가중치의 출처와 라이선스는 R03 기록 및 원 프로젝트를 따른다. 비공개 보관이 새로운 재배포 권한을 만들지는 않는다.
''')
    write(DEST/'INFERENCE.md','''# R04 오프라인 실행

이 명령은 이미지 파일 또는 폴더를 읽어 4종 bbox를 그린 PNG와 원본좌표 JSON을 새 폴더에 저장한다. 카메라를 열지 않는다. 예측은 정확도 측정이 아니며 4종 부품 mask 출력은 없다.

```powershell
python scripts/infer_r04.py --input 'C:/images/board.jpg' --output 'C:/results/r04_01' --device 0
```

패키지 루트에서 실행한다. `--arm baseline` 또는 `--arm improved`로 비교 모델을 선택할 수 있다. 기본값은 검증셋으로 고정한 recommended 모델이다. checkpoint SHA, confidence, native/dual pipeline은 selected_models.json에서 읽는다. CPU는 `--device cpu`다.

```powershell
.\Run-Inference.ps1 -InputPath 'C:/images' -OutputPath 'C:/results/r04_02'
```

PowerShell wrapper는 현재 PC의 기존 Python을 기본 사용한다. 다른 PC에서는 `-PythonPath`를 지정한다. 필요한 버전은 environment.json에 있다. 별도 설치는 자동 수행하지 않는다.

native는1024px 원본 타일, dual은1024/2048px 원본 타일을 모델 입력1024px로 처리한다. overlap0.2, 모델 내부NMS0.6, 클래스별 병합NMS0.5, raw confidence0.001, 최대1000검출이다. 모델 반환 xyxy는 이미 원본 crop 좌표라서 crop offset만 더한다. 운영 threshold는 병합 후 적용한다.

재학습은 원자료를 확보한 후 prepare_r04.py의 source 경로를 새 작업폴더에 맞추고 두 arm을 구성해야 한다. 기록 폴더에 덮어쓰지 않는다. train_r04.py는 독립 데이터 감사·loss 검증과 hash를 확인한 뒤에만 실행한다. run_r04_evaluation.py는 두 학습이 완료되어야 실행된다. 기존 완료 holdout은 입력·출력 hash가 같은 경우 재사용하며 무조건 다시 추론하지 않는다.
''')
    write(DEST/'Run-Inference.ps1',r'''param(
 [Parameter(Mandatory=$true)][string]$InputPath,
 [Parameter(Mandatory=$true)][string]$OutputPath,
 [string]$PythonPath='C:\Users\hkjun\Documents\mcu-vision\.venv-yolo11\Scripts\python.exe',
 [string]$Device='0',
 [ValidateSet('recommended','baseline','improved')][string]$Arm='recommended'
)
$ErrorActionPreference='Stop'
& $PythonPath -B (Join-Path $PSScriptRoot 'scripts/infer_r04.py') --input $InputPath --output $OutputPath --selection (Join-Path $PSScriptRoot 'selected_models.json') --device $Device --arm $Arm
if ($LASTEXITCODE -ne 0) { throw "Inference exited $LASTEXITCODE" }
''')
    dependencies = {name:importlib.metadata.version(name) for name in
                    ['torch','torchvision','ultralytics','numpy','opencv-python','Pillow','PyYAML','pycocotools','matplotlib']}
    save(DEST/'environment.json', {'python':platform.python_version(), 'platform':platform.platform(),
                                  'dependencies':dependencies,
                                  'training':{arm:read(ROOT/'runs'/arm/'start.json') for arm in ['baseline','improved']}})
    write(DEST/'requirements-recorded.txt','\n'.join(f'{name}=={version}' for name,version in dependencies.items()))
    files=[{'path':p.relative_to(DEST).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(DEST.rglob('*')) if p.is_file()]
    for item in files:
        p=Path(item['path'])
        if p.suffix.lower() in media_extensions and (p.parent.as_posix()!='reports/figures' or p.name not in allowed_plot_names):
            raise RuntimeError('Unexpected source media in final package')
    save(DEST/'SHA256SUMS.json',files)
    for item in files:
        assert sha(DEST/item['path'])==item['sha256']
    archive=OUT/'PCB_D455_R04.zip'
    if archive.exists(): raise RuntimeError('Existing R04 ZIP preserved')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():z.write(p,Path(DEST.name)/p.relative_to(DEST))
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad: raise RuntimeError(f'ZIP CRC failed: {bad}')
    save(OUT/'PCB_D455_R04_검증기록.json',{'status':'PASS_PACKAGE_HASH_CRC','folder':str(DEST),'files':len(files)+1,
         'zip_bytes':archive.stat().st_size,'zip_sha256':sha(archive),'final_result_audit_sha256':sha(ROOT/'reports/final_results_audit.json'),
         'source_photos_included':False,'d455_verified':False})
    print(json.dumps({'package':str(DEST),'files':len(files)+1,'zip_bytes':archive.stat().st_size,'zip_sha256':sha(archive)}))


if __name__ == '__main__': main()
