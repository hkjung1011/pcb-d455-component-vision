from pathlib import Path
import csv
import hashlib
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parents[1]
TRIAL=HERE/'work/boards8_pilot_20260929'
RUN=TRIAL/'runs/boards8_yolo11s_10ep_attempt2'
OUT=HERE/'outputs/BOARDS8_PILOT_2026-09-29'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
state=read(TRIAL/'execution_state_attempt2.json')
assert state['status']=='complete' and state['test_invocations']==1
summary=read(RUN/'training-summary.json')
test=read(RUN/'test-evaluation.json')
verified=read(TRIAL/'datasets/verification.json')['boards_v1']
assert sha(Path(summary['best']))==summary['best_sha256']==test['checkpoint_sha256']
OUT.mkdir(exist_ok=True)
(OUT/'evidence').mkdir(exist_ok=True)
(OUT/'code').mkdir(exist_ok=True)
shutil.copyfile(summary['best'],OUT/'boards8-yolo11s-best.pt')
for file in ['training-summary.json','test-evaluation.json','test-evaluation-attempt.json','results.csv','args.yaml','epoch-accounting-correction.json','training-summary-before-epoch-accounting-correction.json']:
    shutil.copyfile(RUN/file,OUT/'evidence'/file)
for file in ['derivation.json','execution_state_attempt2.json','execution_state_before_epoch_accounting_correction.json','failure.json','class_map.json']:
    shutil.copyfile(TRIAL/file,OUT/'evidence'/file)
shutil.copyfile(TRIAL/'datasets/verification.json',OUT/'evidence/dataset_verification.json')
for file in ['prepare_boards8_pilot.py','run_boards8_pilot.py','package_boards8_training.py','correct_pilot_epoch_accounting.py']:
    shutil.copyfile(HERE/'work'/file,OUT/'code'/file)
for file in ['results.png','BoxPR_curve.png','confusion_matrix_normalized.png']:
    if (RUN/file).exists():
        shutil.copyfile(RUN/file,OUT/f'validation_{file}')
for file in ['BoxPR_curve.png','confusion_matrix_normalized.png','confusion_matrix.png']:
    if (RUN/'test'/file).exists():
        shutil.copyfile(RUN/'test'/file,OUT/f'test_{file}')
rows=list(csv.DictReader((RUN/'results.csv').open(encoding='utf-8-sig')))
rows=[{k.strip():float(v) for k,v in r.items()} for r in rows]
epoch=[int(r['epoch']) for r in rows]
plt.rcParams.update({'figure.dpi':150,'font.size':10})
fig,axes=plt.subplots(1,2,figsize=(12,4.3))
for key in ['train/box_loss','train/cls_loss','train/dfl_loss']:
    axes[0].plot(epoch,[r[key] for r in rows],marker='o',markersize=3,label=key.removeprefix('train/'))
axes[0].set(xlabel='Epoch',ylabel='Training loss',title='Training loss (FP32, YOLO11s)')
axes[0].legend()
for key,label in [('metrics/mAP50(B)','mAP50'),('metrics/mAP50-95(B)','mAP50-95')]:
    axes[1].plot(epoch,[100*r[key] for r in rows],marker='o',markersize=3,label=label)
axes[1].set(xlabel='Epoch',ylabel='Validation AP (%)',title='Validation used for checkpoint selection',ylim=(0,100))
axes[1].legend()
for ax in axes:
    ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(OUT/'learning_curves.png',bbox_inches='tight')
plt.close(fig)
classes=summary['classes']
x=np.arange(len(classes))
fig,ax=plt.subplots(figsize=(12,4.7))
for i,split in enumerate(['train','val','test']):
    ax.bar(x+(i-1)*.25,[verified['boxes'][split][c] for c in classes],width=.25,label=split)
ax.set_xticks(x,classes,rotation=25,ha='right')
ax.set(ylabel='Bounding boxes',title='8-class dataset label distribution')
ax.legend()
ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(OUT/'class_distribution.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5))
names=[x['class_name'] for x in test['per_class']]
ap=[100*x['map50_95'] if x['map50_95'] is not None else 0 for x in test['per_class']]
ax.barh(names,ap,color='#2a6f97')
for i,value in enumerate(ap):
    ax.text(value+1,i,f'{value:.1f}%',va='center')
ax.set(xlabel='Test AP50-95 (%)',xlim=(0,110),title='One test evaluation: 106 public-source images')
ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
fig.tight_layout();fig.savefig(OUT/'test_per_class_ap.png',bbox_inches='tight');plt.close(fig)
with (OUT/'test_per_class.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(test['per_class'][0]));writer.writeheader();writer.writerows(test['per_class'])
best_epoch=max(state['epochs'],key=lambda r:(r['fitness'],r['epoch']))['epoch']
val50=summary['val_metrics']['metrics/mAP50(B)']*100
val95=summary['val_metrics']['metrics/mAP50-95(B)']*100
test50=test['overall']['metrics/mAP50(B)']*100
test95=test['overall']['metrics/mAP50-95(B)']*100
table='\n'.join(f"| {r['class_name']} | {r['ground_truth_boxes']} | {100*r['map50']:.2f}% | {100*r['map50_95']:.2f}% |" for r in test['per_class'])
report=f'''# 8클래스 보드 검출 파일럿 — 2026-09-29

**YOLO11s를 실제 {summary['epochs_completed']}에폭 학습하고, 검증 성능으로 선택한 가중치를 고정한 뒤 test 106장으로 한 번 평가했습니다. test mAP50는 {test50:.2f}%, mAP50–95는 {test95:.2f}%입니다.**

이 모델은 Nucleo를 제외한 8클래스 보드 검출입니다. Nucleo의 검증·평가 자료가 없어 별도 8클래스 데이터셋을 만들었고, 기존 9클래스 자료·포트 자료·기존 모델은 보존했습니다. 포트 학습은 실행하지 않았습니다.

## 실제 실행

| 항목 | 결과 |
|---|---|
| 모델 | YOLO11s, 공식 COCO 초기 가중치 |
| GPU | {state['gpu']} |
| 학습 | {summary['epochs_completed']}에폭, optimizer 실제 갱신 {summary['actual_optimizer_steps']}회 |
| 설정 | 입력 640, batch 8, FP32, AdamW lr 0.001, seed 20260929 |
| 학습 시간 | {summary['training_seconds']/60:.1f}분 |
| PyTorch 최대 할당 메모리 | {summary['peak_cuda_memory_gib']:.2f} GiB |
| 데이터 이미지 | train 1,308 / val 206 / test 106 |
| Nucleo 제외 | Nucleo가 포함된 train 이미지 393장 전체 제외 |
| 선택 epoch | 기록된 val fitness 기준 {best_epoch} |
| 검증 mAP50 / mAP50–95 | {val50:.2f}% / {val95:.2f}% |
| test mAP50 / mAP50–95 | {test50:.2f}% / {test95:.2f}% |
| test 실행 횟수 | 가중치 해시 고정 후 1회 |

3에폭 시점에 손실과 모델 값의 유한성, 실제 optimizer 갱신을 확인하고 총 10에폭까지 진행했습니다. 초기 첫 시도는 상태 점검 코드와 라이브러리 손실값 형식 차이로 optimizer 갱신 0회에서 중단됐습니다. 점검 코드를 수정한 뒤 동일 초기 가중치로 다시 시작했으며 실패 기록도 보존했습니다.

학습 종료 후 라이브러리가 마지막 검증 callback을 한 번 더 호출해 임시 로그에 epoch 11이 생겼습니다. 실제 `results.csv`의 10개 학습 행과 동일 optimizer 갱신 수를 근거로 완료 에폭 집계만 정정했습니다. 원래 기록과 [정정 근거](evidence/epoch-accounting-correction.json)를 보존했으며, 가중치·test 지표는 그대로이고 test를 반복하지 않았습니다. 동봉 실행 코드에는 이후의 기록 오류를 막는 수정도 포함했습니다.

## 학습 곡선과 데이터 분포

![학습 곡선](learning_curves.png)

![클래스별 라벨 분포](class_distribution.png)

## 클래스별 test AP

| 클래스 | test 정답 박스 | AP50 | AP50–95 |
|---|---:|---:|---:|
{table}

![클래스별 test AP](test_per_class_ap.png)

이번 결과에서 `other_board` AP50–95는 0.16%, Pi4는 26.00%로 낮습니다. 특히 기타 보드 클래스는 학습 자료가 Embedded Hardware의 AURIX에 집중되어 있어 다양한 보드로 일반화되는지 보완 검토가 필요합니다. 이 파일럿 점수만으로 모든 보드에 사용할 준비가 끝났다고 판단하지 않습니다.

## 사용 범위

공개자료의 그룹 분할에서 얻은 파일럿 결과입니다. Nucleo·포트 인식 성능, 실제 D455 카메라 정확도, 실물 보드·촬영 세션의 통계적 독립성을 입증하지 않습니다. 각 클래스의 평가 사진과 그룹 수가 작으므로 여러 촬영 환경으로 일반화되는지 추가 확인해야 합니다. test 점수로 재학습 설정을 선택하지 않았습니다.

파이프라인: 기존 분할 유지 → Nucleo 포함 이미지 전체 제외 → 8개 클래스 ID 재매핑 → 파일·라벨·그룹·holdout 재검증 → 10에폭 학습 → val fitness로 checkpoint 선택 → SHA-256 고정 → test 1회.

가중치: [boards8-yolo11s-best.pt](boards8-yolo11s-best.pt)

SHA-256: `{summary['best_sha256']}`

```python
from ultralytics import YOLO
model = YOLO("boards8-yolo11s-best.pt")
model.predict(source="board_photo.jpg", imgsz=640, conf=0.25, save=True)
```

위 conf 0.25는 예시이며 test에서 최적화한 운영 임계값이 아닙니다. 클래스 이름은 가중치에 포함되어 있습니다. 원본 사진·ZIP은 기존 로컬 데이터 폴더에 있고 이 결과 묶음에는 포함하지 않았습니다.

[학습 기록](evidence/training-summary.json) · [test 결과](evidence/test-evaluation.json) · [데이터 검증](evidence/dataset_verification.json) · [클래스별 CSV](test_per_class.csv) · [소스 코드](code/run_boards8_pilot.py)
'''
(OUT/'README.md').write_text(report,encoding='utf-8')
manifest={'scope':'all files except this manifest','files':[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='ARTIFACT_MANIFEST.json']}
(OUT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'output':str(OUT),'epochs':summary['epochs_completed'],'optimizer_steps':summary['actual_optimizer_steps'],'test_map50':test50,'test_map50_95':test95,'best_epoch':best_epoch},ensure_ascii=False,indent=2))
