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
TRIAL=HERE/'work/boards1000_s650_v150_t200_e50_20260929'
RUN=TRIAL/'runs/boards1000_yolo11s_50ep'
OUT=HERE/'outputs/BOARDS1000_TRAIN650_VAL150_TEST200'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
state=read(TRIAL/'execution_state.json')
summary=read(RUN/'training-summary.json')
test=read(RUN/'test-evaluation.json')
verification=read(TRIAL/'datasets/verification.json')['boards_v1']
derivation=read(TRIAL/'derivation.json')
assert state['status']=='complete' and state['test_invocations']==1
assert verification['images']=={'train':650,'val':150,'test':200}
assert summary['epochs_completed']==len(state['epochs'])
assert sha(Path(summary['best']))==summary['best_sha256']==test['checkpoint_sha256']
OUT.mkdir(exist_ok=True)
(OUT/'evidence').mkdir(exist_ok=True)
(OUT/'code').mkdir(exist_ok=True)
shutil.copyfile(summary['best'],OUT/'boards8-1000images-yolo11s-best.pt')
for f in ['training-summary.json','test-evaluation.json','test-evaluation-attempt.json','results.csv','args.yaml']:
    shutil.copyfile(RUN/f,OUT/'evidence'/f)
for f in ['derivation.json','training_plan.json','execution_state.json','class_map.json']:
    shutil.copyfile(TRIAL/f,OUT/'evidence'/f)
shutil.copyfile(TRIAL/'datasets/verification.json',OUT/'evidence/dataset_verification.json')
shutil.copyfile(TRIAL/'datasets/boards_v1/dataset_index.json',OUT/'evidence/dataset_index.json')
for f in ['prepare_boards1000.py','run_boards1000.py','package_boards1000.py']:
    shutil.copyfile(HERE/'work'/f,OUT/'code'/f)
for f in ['common.py','assemble.py','verify_datasets.py']:
    shutil.copyfile(TRIAL/f,OUT/'code'/f)
for f in ['BoxPR_curve.png','confusion_matrix_normalized.png','confusion_matrix.png']:
    if (RUN/'test'/f).exists(): shutil.copyfile(RUN/'test'/f,OUT/f'test_{f}')
for f in ['results.png','BoxPR_curve.png','confusion_matrix_normalized.png']:
    if (RUN/f).exists(): shutil.copyfile(RUN/f,OUT/f'validation_{f}')
rows=[{k.strip():float(v) for k,v in r.items()} for r in csv.DictReader((RUN/'results.csv').open(encoding='utf-8-sig'))]
assert len(rows)==summary['epochs_completed'] and len(rows)<=50
epochs=[int(r['epoch']) for r in rows]
best_epoch=max(state['epochs'],key=lambda r:(r['fitness'],r['epoch']))['epoch']
plt.rcParams.update({'figure.dpi':150,'font.size':10})
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for key in ['train/box_loss','train/cls_loss','train/dfl_loss']:
    axes[0].plot(epochs,[r[key] for r in rows],label=key.removeprefix('train/'))
axes[0].set(title='Training loss: 650 images',xlabel='Epoch',ylabel='Loss')
for key,label in [('metrics/mAP50(B)','mAP50'),('metrics/mAP50-95(B)','mAP50-95')]:
    axes[1].plot(epochs,[r[key]*100 for r in rows],label=label)
axes[1].axvline(best_epoch,color='gray',ls='--',alpha=.6,label=f'Selected epoch {best_epoch}')
axes[1].set(title='Validation: 150 images',xlabel='Epoch',ylabel='AP (%)',ylim=(0,100))
for ax in axes: ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(OUT/'learning_curves.png',bbox_inches='tight');plt.close(fig)
classes=summary['classes'];x=np.arange(len(classes))
fig,ax=plt.subplots(figsize=(12,4.5))
for i,split in enumerate(['train','val','test']):
    ax.bar(x+(i-1)*.25,[verification['boxes'][split][c] for c in classes],.25,label=f'{split}: {verification["images"][split]} images')
ax.set_xticks(x,classes,rotation=25,ha='right');ax.set(ylabel='Ground-truth boxes',title='Class distribution (whole groups kept together)');ax.legend();ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(OUT/'class_distribution.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5))
aps=[100*r['map50_95'] for r in test['per_class']]
ax.barh(classes,aps,color='#286f94')
for i,ap in enumerate(aps): ax.text(ap+1,i,f'{ap:.1f}%',va='center')
ax.set(xlim=(0,110),xlabel='AP50-95 (%)',title='Development test: 200 images; one evaluation')
ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
fig.tight_layout();fig.savefig(OUT/'test_per_class_ap.png',bbox_inches='tight');plt.close(fig)
with (OUT/'test_per_class.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(test['per_class'][0]));w.writeheader();w.writerows(test['per_class'])
with (OUT/'split_class_counts.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['class','train_boxes','val_boxes','test_boxes','train_groups','val_groups','test_groups'])
    for c in classes:w.writerow([c,*[verification['boxes'][s][c] for s in ['train','val','test']],*[derivation['class_group_counts'][s][c] for s in ['train','val','test']]])
test50=test['overall']['metrics/mAP50(B)']*100;test95=test['overall']['metrics/mAP50-95(B)']*100
val50=summary['val_metrics']['metrics/mAP50(B)']*100;val95=summary['val_metrics']['metrics/mAP50-95(B)']*100
epoch10=next((r['metrics/mAP50-95(B)']*100 for r in rows if int(r['epoch'])==10),None)
table='\n'.join(f"| {r['class_name']} | {r['ground_truth_boxes']} | {derivation['class_group_counts']['test'][r['class_name']]} | {100*r['map50']:.2f}% | {100*r['map50_95']:.2f}% |" for r in test['per_class'])
report=f'''# 보드 검출 — 총 1,000장, 650 / 150 / 200 분할

**YOLO11s를 최대 50에폭 설정으로 실행해 실제 {summary['epochs_completed']}에폭 학습했습니다. 검증 성능으로 epoch {best_epoch} 모델을 선택한 뒤 개발용 test 200장을 한 번 평가했습니다. mAP50는 {test50:.2f}%, mAP50–95는 {test95:.2f}%입니다.**

## 데이터 분할

| 용도 | 이미지 | 정답 박스 | 역할 |
|---|---:|---:|---|
| train | 650 | {sum(verification['boxes']['train'].values())} | 가중치 학습 |
| val | 150 | {sum(verification['boxes']['val'].values())} | best 모델 선택과 조기 종료 |
| test | 200 | {sum(verification['boxes']['test'].values())} | checkpoint 고정 후 이번 실험에서 1회 개발 평가 |
| 합계 | 1,000 | {sum(sum(b.values()) for b in verification['boxes'].values())} | 8개 보드 클래스 |

IoTKITs 후보 1,053장에서 1,000장을 골랐습니다. 클래스별 사진 수를 고려하면서 기존 원본·증강·유사도 연결 그룹 전체를 한 분할에만 배정했습니다. 반복 영상 960장이 큰 그룹으로 연결된 Embedded Hardware는 이 1,000장 실험에서 제외했습니다. 기존 9클래스·포트 데이터와 10에폭 가중치는 보존했습니다. Nucleo와 포트는 이번 모델 범위에 포함하지 않았습니다.

정확한 이미지 수와 클래스별 그룹 지원을 정수 최적화로 맞췄습니다. 배정에는 예측이나 AP 수치를 사용하지 않았습니다. 모든 클래스가 각 분할에 있고, train은 클래스마다 최소 3개, val/test는 최소 2개 연결 그룹을 포함합니다. 파일·라벨 오류와 설정된 회전·반전 pHash 기준 split/Commons 유사 후보는 0건입니다. 연결 그룹 수가 실제 실물·촬영 세션의 독립성을 입증하지는 않습니다.

**test 200장은 새 최종시험이 아닙니다.** 이전 test 106장, 이전 val 86장, 이전 train 8장으로 구성한 개발 평가입니다. 이전 test를 이번 train/val로 옮기지는 않았습니다. 이번 모델은 공식 초기 가중치에서 다시 시작했고, 새 test 200장은 이번 모델의 학습·checkpoint 선택에는 사용하지 않았습니다. 이전 10에폭 실험과 데이터·분할이 달라 점수 차이를 에폭 증가만의 효과로 해석할 수 없습니다.

![라벨 분포](class_distribution.png)

## 학습 설정과 실제 실행

| 항목 | 값 |
|---|---|
| 모델 | YOLO11s, 8클래스 bbox 검출 |
| 초기화 | 공식 COCO 가중치; 이전 10에폭 모델에서 이어 학습하지 않음 |
| 실행 GPU | {state['gpu']} |
| 에폭 | 최대 50 / 실제 {summary['epochs_completed']} |
| 조기 종료 | 검증 fitness가 12에폭 동안 개선되지 않을 때 |
| optimizer 갱신 | 실제 {summary['actual_optimizer_steps']}회 |
| 입력·배치 | 640 × 640 / batch 8 / FP32 |
| 최적화 | AdamW, 초기 lr 0.001, cosine schedule, warmup 3 epochs |
| 증강 | mosaic 포함, 마지막 예정 10에폭은 mosaic 종료 |
| seed | 20260929 |
| 학습 시간 | {summary['training_seconds']/60:.1f}분 |
| 최대 PyTorch 할당 메모리 | {summary['peak_cuda_memory_gib']:.2f} GiB |
| 선택 epoch | {best_epoch}, 검증 fitness 기준 |
| 이번 실행의 10에폭 시점 val mAP50–95 | {epoch10:.2f}% |
| val mAP50 / mAP50–95 | {val50:.2f}% / {val95:.2f}% |
| 개발 test mAP50 / mAP50–95 | {test50:.2f}% / {test95:.2f}% |

![학습 곡선](learning_curves.png)

10에폭 시점 수치는 이번 50에폭 설정 실행의 중간 기록입니다. 이전의 별도 10에폭 파일럿과 비교한 수치가 아니며, 독립적인 하이퍼파라미터 대조 실험도 아닙니다.

## 클래스별 개발 평가

| 클래스 | test 박스 | test 연결 그룹 | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
{table}

![클래스별 평가](test_per_class_ap.png)

test 사진이 200장이어도 클래스당 독립 그룹이 적은 경우 불확실성이 큽니다. 실제 D455 촬영 성능과 다양한 조명·배경·보드 개체에서의 일반화는 별도 검증이 필요합니다. 이번 결과로 운영 임계값을 결정하거나 새 최종시험 성능이라고 주장하지 않습니다.

## 결과 파일

- [선택된 가중치](boards8-1000images-yolo11s-best.pt), SHA-256 `{summary['best_sha256']}`
- [학습 기록](evidence/training-summary.json) · [test 결과](evidence/test-evaluation.json)
- [분할 검증](evidence/dataset_verification.json) · [분할·그룹 근거](evidence/derivation.json)
- [클래스별 표](test_per_class.csv) · [분할별 라벨·그룹 수](split_class_counts.csv)
- [학습 코드](code/run_boards1000.py) · [분할 코드](code/prepare_boards1000.py)

```python
from ultralytics import YOLO
model = YOLO("boards8-1000images-yolo11s-best.pt")
model.predict(source="board_photo.jpg", imgsz=640, conf=0.25, save=True)
```

conf 0.25는 실행 예시이며 test로 최적화한 임계값이 아닙니다. 원본 사진과 ZIP은 기존 로컬 작업 폴더에 있습니다. 동봉 코드는 당시 로컬 경로를 사용하므로 다른 PC에서는 경로와 원자료를 복원해야 합니다.
'''
(OUT/'README.md').write_text(report,encoding='utf-8')
manifest={'scope':'all files except this manifest','files':[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='ARTIFACT_MANIFEST.json']}
(OUT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'output':str(OUT),'epochs':summary['epochs_completed'],'best_epoch':best_epoch,'optimizer_steps':summary['actual_optimizer_steps'],'test_map50':test50,'test_map50_95':test95},ensure_ascii=False,indent=2))
