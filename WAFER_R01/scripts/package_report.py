"""Archive local pilot models and scalar evidence, excluding source images/annotations."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, shutil, zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parents[1]
OUT=BASE/'outputs'; DEST=OUT/'WAFER_Surface_R01'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def copy(p,q): q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
def text(p,s): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(s.strip()+'\n',encoding='utf-8')


def main():
    if DEST.exists(): raise RuntimeError('Preserve existing wafer archive')
    train=read(ROOT/'reports/training_summary.json'); test=read(ROOT/'reports/test_metrics.json')
    data=read(ROOT/'reports/data_audit.json'); selection=read(ROOT/'reports/selection_frozen.json')
    audit=read(ROOT/'reports/independent_final_results_audit.json')
    if train['epochs_completed']!=10 or train['direct_optimizer_calls']<=0: raise RuntimeError('Ten actual epochs required')
    if not audit['status'].startswith('PASS'): raise RuntimeError('Final independent result audit required')
    if test['checkpoint_sha256']!=selection['sha256'] or sha(selection['checkpoint'])!=selection['sha256']: raise RuntimeError('Checkpoint SHA mismatch')
    DEST.mkdir(parents=True)
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
    figdir=DEST/'figures'; figdir.mkdir()
    inventory=[]
    def finish(fig,name,caption):
        fig.tight_layout(rect=(0,.06,1,.96)); fig.text(.02,.015,caption,fontsize=9,color='#475569')
        paths=[]
        for ext in ['png','svg']:
            p=figdir/f'{name}.{ext}'; fig.savefig(p,dpi=150,bbox_inches='tight'); paths.append({'file':p.name,'sha256':sha(p)})
        plt.close(fig); inventory.append({'name':name,'files':paths})
    fig,ax=plt.subplots(figsize=(13,5)); ax.axis('off')
    boxes=[('공개 현미경 자료\n2,132장 · bbox 3,693개',.17,.74),
           ('라벨 충돌 98장 격리\n증강본 제외 · 유사 영상 묶기',.5,.74),
           ('학습 1,472 / 검증 275\n시험 287장',.83,.74),
           ('YOLO11n · 10에폭\n1024 · batch 4 · FP32',.17,.28),
           ('검증 AP로 가중치 선택\n선택 SHA 고정',.5,.28),
           ('학습 미사용 이미지 시험\nAP · 오류 · 라벨/가중치 기록',.83,.28)]
    for label,x,y in boxes:
        ax.text(x,y,label,ha='center',va='center',fontsize=12,bbox={'boxstyle':'round,pad=.7','facecolor':'#e6f3f5','edgecolor':'#16758b'})
    for i,j in [(0,1),(1,2),(3,4),(4,5)]:
        _,x,y=boxes[i];_,xx,yy=boxes[j];ax.annotate('',(xx-.14,yy),(x+.14,y),arrowprops={'arrowstyle':'->','color':'#16758b','lw':2})
    ax.annotate('',(.17,.43),(.83,.59),arrowprops={'arrowstyle':'->','color':'#16758b','lw':2,'connectionstyle':'angle,angleA=-90,angleB=0,rad=12'})
    fig.suptitle('웨이퍼 표면 결함 검출 파일럿의 실제 실행 흐름',fontsize=18,fontweight='bold')
    finish(fig,'01_pipeline','현미경 촬영 공개자료 실험. 물리 웨이퍼·설계 독립성 및 D455 RGB 실측 성능은 미검증. instance mask 학습 없음.')
    names=data['names']; x=np.arange(len(names)); fig,ax=plt.subplots(figsize=(12,5)); bottom=np.zeros(len(names))
    for split,color in [('train','#147d92'),('val','#69b5a3'),('test','#e5af54')]:
        values=np.array([data['split'][split]['classes'][n] for n in names]); ax.bar(x,values,bottom=bottom,label=split,color=color)
        bottom+=values
    for i,v in enumerate(bottom): ax.text(i,v+12,str(int(v)),ha='center')
    ax.set_xticks(x,names,rotation=15);ax.set_ylabel('원본 이미지의 bbox 수');ax.legend();ax.set_ylim(0,bottom.max()*1.16)
    fig.suptitle('실제 사용 라벨 3,394개 · 신규 사람이 만든 라벨 0개',fontsize=18,fontweight='bold')
    finish(fig,'02_label_counts','원본 후보 bbox 3,693개 중 라벨 충돌 이미지 98장의 299개 제외. 6종, 별도 클래스 loss 가중 없음(cls_pw=0).')
    epochs=read(ROOT/'reports/epochs.json')
    with (ROOT/'runs/yolo11n_10ep/results.csv').open(encoding='utf-8-sig') as f:
        csvrows=list(csv.DictReader(f))
    csvrows=[{k.strip():v for k,v in r.items()} for r in csvrows]
    fig,axs=plt.subplots(1,2,figsize=(12,5)); ex=[e['epoch'] for e in epochs]
    axs[0].plot(ex,[e['validation']['metrics/mAP50-95(B)']*100 for e in epochs],marker='o',label='val mAP50–95')
    axs[0].plot(ex,[e['validation']['metrics/mAP50(B)']*100 for e in epochs],marker='s',label='val mAP50')
    axs[0].set_ylabel('%');axs[0].legend();axs[0].set_xlabel('실제 epoch')
    for k in ['train/box_loss','train/cls_loss','train/dfl_loss']:
        axs[1].plot([float(r['epoch']) for r in csvrows],[float(r[k]) for r in csvrows],label=k)
    axs[1].legend();axs[1].set_xlabel('실제 epoch');axs[1].set_ylabel('Loss')
    fig.suptitle('10에폭 학습의 실제 곡선',fontsize=18,fontweight='bold')
    finish(fig,'03_training_curves',f"직접 측정 optimizer {train['direct_optimizer_calls']:,}회 · 이미지 반복 노출 {train['image_exposures']:,}회. 검증 AP는 모델 선택에 사용한 값이다.")
    fig,ax=plt.subplots(figsize=(11,5)); cl=test['per_class']; yy=np.arange(len(cl))
    ax.barh(yy-.17,[100*r['ap50'] for r in cl],height=.32,label='AP50',color='#69b5a3')
    ax.barh(yy+.17,[100*r['ap50_95'] for r in cl],height=.32,label='AP50–95',color='#147d92')
    ax.set_yticks(yy,[r['name'] for r in cl]);ax.set_xlim(0,105);ax.set_xlabel('%');ax.legend()
    fig.suptitle(f"시험 287장 / 496개 bbox · mAP50–95 {100*test['bbox_map50_95']:.2f}%",fontsize=18,fontweight='bold')
    finish(fig,'04_test_class_ap','Ultralytics bbox AP, confidence floor 0.001 · NMS IoU 0.7 · 이미지 전체 최대 300개. 새 물리 웨이퍼 성능과 구분.')
    save(figdir/'inventory.json',inventory)
    for p in (ROOT/'scripts').glob('*.py'): copy(p,DEST/'scripts'/p.name)
    for p in (ROOT/'reports').rglob('*'):
        if p.is_file() and p.suffix in {'.json','.md','.sha256','.log'}: copy(p,DEST/'reports'/p.relative_to(ROOT/'reports'))
    copy(ROOT/'protocol.json',DEST/'protocol.json')
    for n in ['best.pt','last.pt']: copy(ROOT/'runs/yolo11n_10ep/weights'/n,DEST/'models'/n)
    for n in ['args.yaml','results.csv']: copy(ROOT/'runs/yolo11n_10ep'/n,DEST/'training'/n)
    copy(ROOT/'training.log',DEST/'training/training.log')
    for name in ['research.md','research.json','training_script_review.md']:
        copy(ROOT.parent/'wafer_research'/name,DEST/'research'/name)
    # Keep sample IDs, group membership and image hashes, but no source pixel or bbox coordinates.
    source_manifest=read(ROOT/'data/manifest.json')
    save(DEST/'image_split_inventory.json',[{k:r[k] for k in ['id','split','group_id','image_sha256','decoded_sha256']} for r in source_manifest['records']])
    best_epoch=max(epochs,key=lambda r:(r['validation']['metrics/mAP50-95(B)'],r['epoch']))['epoch']
    status={'revision':'WAFER_SURFACE_R01','status':'TRAINED_AND_PUBLIC_IMAGE_TESTED','utc':datetime.now(timezone.utc).isoformat(),
            'epochs':10,'optimizer_calls':train['direct_optimizer_calls'],'selected_epoch_reconstructed_from_val':best_epoch,
            'source_images_acquired':2132,'admitted_images':2034,'source_bbox':3693,'admitted_bbox':3394,
            'split':data['split'],'test_map50':test['bbox_map50'],'test_map50_95':test['bbox_map50_95'],
            'new_human_labels':0,'instance_masks_trained':False,'d455_verified':False,'physical_wafer_independence_verified':False,
            'source_data_redistributed':False,'license':'Dataset license unspecified; author-published research data',
            'checkpoint':{'path':'models/best.pt','sha256':sha(DEST/'models/best.pt')},'source_annotation_limit':'Remaining labels not exhaustively expert-reviewed'}
    save(DEST/'CURRENT_STATUS.json',status)
    table='\n'.join(f"| {s} | {r['images']} | {r['bbox']} |" for s,r in data['split'].items())
    cls_table='\n'.join(f"| {r['name']} | {100*r['ap50']:.2f}% | {100*r['ap50_95']:.2f}% |" for r in cl)
    operating=audit['fixed_confidence']['micro']
    text(DEST/'README.md',f'''# 웨이퍼 표면 결함 검출 R01 · 실제 학습 결과

2026-09-29. **YOLO11n 10에폭 학습과 학습 미사용 이미지 287장 평가 완료.** bbox mAP50 **{100*test['bbox_map50']:.2f}%**, mAP50–95 **{100*test['bbox_map50_95']:.2f}%**다. 현미경 공개자료의 첫 기준 모델이며 D455 실물 성능이나 생산 검사 합격을 뜻하지 않는다.

## 무엇을 학습했는가

[Wafer-Datas](https://github.com/ztao3243/Wafer-Datas)의 [원저자 공개 선언](https://fcds.cs.put.poznan.pl/FCDS/ArticleDetails.aspx?articleId=521)을 확인하고 commit `17dbc7e19f4e354f9b2b436af66e5abdaf2ea4b1`에서 파일을 받았다. [논문 §3.1](https://reference-global.com/pdf/10.2478/fcds-2024-0014)은 촬영을 electron microscopy와 부착 카메라로 설명한다. 일반 RGB 촬영 자료로 단정하지 않는다.

증강 접미사가 없는 원본 후보 2,132장과 bbox 3,693개를 확보했다. 같은 픽셀에 서로 다른 정답이 붙은 49쌍·98장을 격리해 2,034장·bbox 3,394개를 사용했다. 원본은 그대로 보존했다. source의 `Short_circuit`/`short_cricuit`를 `short_circuit`으로 명시적으로 통합했다. 신규 사람이 만든 라벨과 실제 instance mask는 각각0개다.

| 분할 | 이미지 | bbox |
|---|---:|---:|
{table}

완전 중복, 명시적인 파일명 계열, pHash≤4의 유사 외관 후보를 같은 split으로 묶었다. **물리 웨이퍼·lot ID가 없으므로 새로운 물리 웨이퍼나 새로운 설계의 성능은 검증하지 못했다.** 남은 라벨 전체를 전문가가 재검수한 것도 아니다.

사용한 2,034장은 모두 결함 라벨이 있는 사진이다. 정상 웨이퍼만으로 구성한 별도 음성 시험셋이 없으므로 정상 제품의 오경보율은 아직 검증하지 않았다.

## 모델과 설정

- 공식 COCO 초기 `yolo11n.pt` → 6종 bbox 검출, 2,591,010 parameters.
- 1024 입력, batch4, nbs8, AdamW lr0=0.001/lrf=0.01, FP32, seed42, 최대10에폭.
- mosaic/mixup/임의 crop/scale 없음. 좌우·상하 반전과 밝기 변화만 사용.
- box/cls/DFL 손실 계수 7.5/0.5/1.5, 추가 class multiplier 없음(cls_pw=0). 이미지별 균등 shuffle이며 class oversampling은 하지 않았다.
- 실제 optimizer 호출 **{train['direct_optimizer_calls']:,}회**, 이미지 반복 노출 **{train['image_exposures']:,}회**. 가중치 변경·EMA·실제 처리 수를 기록했다.
- 검증 AP50–95 기준 best 선택(기록에서 재구성한 epoch {best_epoch}), 가중치 SHA 고정 후 시험 실행. 시험 결과로 재선택하지 않았다.

## 시험 결과

| 결함 종류 | AP50 | AP50–95 |
|---|---:|---:|
{cls_table}

Ultralytics8.4.120 bbox AP, confidence floor0.001, NMS IoU0.7, 이미지 전체 최대300개 조건이다. PCB의 COCO maxDets 지표와 서로 다른 데이터·구현이므로 직접 우열 비교하지 않는다. 라이브러리 기본 P/R은 시험셋 F1 최적점이므로 실사용 운용점 결과로 사용하지 않았다. 별도 보조 분석이 있으면 고정 confidence0.25라는 조건을 명시한다.

시험 전에 정한 confidence 0.25 / IoU 0.5에서 **정답 {operating['gt']}개 중 {operating['tp']}개 검출**, FP {operating['fp']}개, FN {operating['fn']}개였다. Precision {operating['precision']*100:.2f}%, Recall {operating['recall']*100:.2f}%다. 이 수치는 저장 예측의 점수순 일대일 매칭으로 계산한 설명용 운용점이며 현장에 맞춰 최적화한 임곗값은 아니다.

## 그래프와 기록

![알고리즘](figures/01_pipeline.png)
![라벨 수](figures/02_label_counts.png)
![학습 곡선](figures/03_training_curves.png)
![종류별 시험 AP](figures/04_test_class_ap.png)

[훈련 기록](reports/training_summary.json) · [독립 데이터 감사](reports/independent_data_audit.json) · [결과 감사](reports/independent_final_results_audit.json) · [시험 지표](reports/test_metrics.json) · [전체 후보 조사](research/research.md).

## 사용하는 법과 다음 단계

`models/best.pt`는 이6종의 현미경 표면 결함 bbox 모델이다. Python 환경에서 `YOLO('models/best.pt').predict(source='image.jpg',imgsz=1024,conf=0.25)`로 시험할 수 있다. confidence0.25는 현장 보정된 값이 아니다. D455 입력·노출·초점·최소 결함 픽셀 수와 실물 정답으로 재검증해야 한다.

실제 segmentation은 [Roboflow Wafer Defect v2](https://universe.roboflow.com/wafer-irhuv/wafer-defect-rv1vx/dataset/2)의 4,532장·7종 polygon 자료가 후속 후보이나 정식 export 로그인이 필요하다. 이번 bbox를 mask 정답으로 바꾸지 않았다.

자료는 연구용으로 공개됐지만 별도 데이터 라이선스는 명시되지 않았다. 원본 사진·원본/변환 라벨·원본 bbox 좌표·학습/예측 사진 grid는 이 패키지에 넣지 않았다. 로컬 원본 위치는 `work/wafer_r01/raw`, 준비된 데이터는 `work/wafer_r01/data`다. 패키지에는 모델·코드·수량·해시·곡선과 실행 기록이 포함된다.
''')
    files=[{'path':p.relative_to(DEST).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(DEST.rglob('*')) if p.is_file()]
    save(DEST/'SHA256SUMS.json',files)
    zip_path=OUT/'WAFER_Surface_R01.zip'
    if zip_path.exists(): raise RuntimeError('Preserve existing zip')
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():z.write(p,Path(DEST.name)/p.relative_to(DEST))
    with zipfile.ZipFile(zip_path) as z:
        if z.testzip() is not None: raise RuntimeError('ZIP CRC failed')
    for r in files:
        if sha(DEST/r['path'])!=r['sha256']:raise RuntimeError('Packaged file changed')
    save(OUT/'WAFER_Surface_R01_검증기록.json',{'status':'FILES_HASHED_ZIP_CRC_VERIFIED','files':len(files),'zip':str(zip_path),'zip_bytes':zip_path.stat().st_size,'zip_sha256':sha(zip_path),'source_photos_included':False})
    print(json.dumps({'status':'PACKAGED','zip':str(zip_path),'files':len(files)}))


if __name__=='__main__':main()
