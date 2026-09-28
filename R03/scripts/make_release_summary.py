"""Build the inference registry and Korean report from actual frozen evaluations."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def percent(x):return '해당 정답 없음' if x is None else f'{x*100:.2f}%'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verified_aggregate():
    """Bind the release to frozen selection and the exact evaluated metric files."""
    folder=ROOT/'reports/evaluation_suite'
    agg=read(folder/'aggregate.json')
    if agg['status']!='COMPLETED_OFFLINE_NOT_D455_VALIDATED':raise ValueError('Evaluation incomplete')
    required={'parts_yolo11n','parts_yolo11s','board_yolo11n','board_yolo11n_seg'}
    if set(agg['models'])!=required:raise ValueError('Four-model release requires all four evaluated models')
    selection=read(folder/'selection.json');protocol=read(folder/'protocol.json')
    payload={'plan':protocol['plan'],'evidence':protocol['evidence']}
    protocol_sha=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    if not protocol_sha==protocol['protocol_sha256']==selection['protocol_sha256']==agg['protocol_sha256']:
        raise ValueError('Protocol evidence hash mismatch')
    if sha(folder/'selection.json')!=agg['selection_sha256']:raise ValueError('Selection file hash mismatch')
    for mid,model in agg['models'].items():
        if model['selection']!=selection['selections'][mid]:raise ValueError(f'Selection differs: {mid}')
        for metric in model['validation_candidates']+list(model['tests'].values()):
            # Packaged evidence is authoritative; original absolute audit paths
            # can point to the development checkout after relocation.
            local=folder/'evaluations'/Path(metric['metrics_path']).parent.name/'metrics.json'
            if sha(local)!=metric['metrics_sha256']:raise ValueError(f'Metrics hash differs: {local}')
            actual=read(local)
            for key in ('checkpoint_sha256','manifest_sha256','split','cohort','bbox','mask','latency'):
                if metric.get(key)!=actual.get(key):raise ValueError(f'Aggregate {key} differs: {local}')
            operating={k:v for k,v in actual['operating'].items() if k!='per_image'}
            if metric['operating']!=operating:raise ValueError(f'Operating metrics differ: {local}')
        chosen=model['selection']['candidate']
        checkpoint=ROOT/'runs'/mid/'fit/weights'/(chosen['checkpoint']+'.pt')
        if sha(checkpoint)!=model['selection']['checkpoint_sha256']:raise ValueError(f'Selected weights changed: {checkpoint}')
    return agg


def make_registry(agg):
    models=agg['models']
    def ranking(mid):
        selection=models[mid]['selection']
        chosen=next(v for v in models[mid]['validation_candidates'] if v['candidate_id']==selection['candidate']['candidate_id'])
        bbox=selection['validation_bbox']
        return (-bbox['ap50_95_max100'],-bbox['ap50_95_max300'],chosen['latency']['offline_predict_and_postprocess_ms_p50'],mid)
    default_parts=min(['parts_yolo11n','parts_yolo11s'],key=ranking)
    def entry(mid):
        s=models[mid]['selection'];candidate=s['candidate']
        checkpoint=Path('runs')/mid/'fit/weights'/(candidate['checkpoint']+'.pt')
        value={'model_id':mid,'checkpoint':checkpoint.as_posix(),'sha256':s['checkpoint_sha256'],
               'confidence':s['operating_confidence'],'imgsz':s['imgsz']}
        if mid.startswith('parts_'):
            value.update(pipeline='native-tiles' if candidate['tile_size'] else 'whole',tile_size=candidate['tile_size'],overlap=.2)
        return value
    return {'schema':'r03-inference-selection-v1','selection_source':'reports/evaluation_suite/selection.json',
        'default_parts_selection_rule':'Validation AP100, AP300, p50 latency; never Pi/general test scores',
        'board_detector':entry('board_yolo11n'),'board_segmenter':entry('board_yolo11n_seg'),'parts':entry(default_parts),
        'camera_verified':False,'target4_component_masks_available':False}


def main():
    agg=verified_aggregate()
    models=agg['models']
    # Select the default component model using VALIDATION only, even though test evidence is now available.
    registry=make_registry(agg)
    default_parts=registry['parts']['model_id']
    pi_result=models[default_parts]['tests']['pi_test']
    pi_counts=pi_result['operating']['per_class_counts_tp_fp_fn']
    save(ROOT/'selected_models.json',registry)
    model_records=[]
    for mid in models:
        architecture=read(ROOT/'runs'/mid/'architecture.json')
        environment=read(ROOT/'runs'/mid/'environment.json')
        summary=read(ROOT/'runs'/mid/'run_summary.json')
        model_records.append({'model_id':mid,'architecture':architecture,'training':environment,'run_summary':summary,
                              'selection':models[mid]['selection'],'tests':models[mid]['tests']})
    save(ROOT/'model_registry.json',{'schema':'r03-model-registry-v1','default_parts_model':default_parts,
         'legacy_micro_A_H_I_status':'QUARANTINED_CLASS_MAPPING_CONFLICT','models':model_records,
         'target4_component_mask_status':'NOT_TRAINED_NO_REVIEWED_NATIVE_MASK_GT','d455_status':'DEVICE_COUNT_ZERO_NO_FRAMES'})
    lines=['# Raspberry Pi 보드·부품 인식 R03 개발결과','',
      '기록일: 2026-09-28. 실제 공개자료로 네 모델을 학습하고 학습·검증에 쓰지 않은 시험셋을 평가했다. **D455 실사용 검증과 저항 포함 4종 부품 윤곽 분할은 아직 완료되지 않았다.**','',
      '클로드 검토 중 원본 해상도, 실제 optimizer 갱신 수, 보드별 분할, 원본 COCO 평가, 크기별 재현율, 검증셋 confidence 고정을 반영했다. 90/220에폭 확대와 8px·16px를 절대적인 검출 한계로 보는 해석은 적용하지 않았다. 자세한 판정은 [검토 반영 기록](claude_review_application.md)에 남겼다.','',
      f'**현재 판단: 세부 부품의 실사용 수준에는 미달한다.** 기본 {default_parts}는 학습·검증에서 제외한 Pi3B 2장/189개에서 bbox mAP50–95 {percent(pi_result["bbox"]["ap50_95_max100"])}다. 검증에서 고정한 confidence {registry["parts"]["confidence"]:.2f}, IoU≥0.5에서 {sum(c[0] for c in pi_counts)}/189개를 찾았고, IC는 {pi_counts[2][0]}/16개, 커넥터는 {pi_counts[3][0]}/12개였다. 보드 전체 검출·외곽 분할 점수가 높아도 부품 인식 성공을 뜻하지 않는다.','',
      '3에폭은 라벨·메모리·실제 가중치 갱신이 정상인지 확인하는 점검 단계로 사용한다. 실사용 충분성을 에폭 수로 보장할 수 없다. 이번 비교는 보드 5에폭, 부품 20에폭으로 끝냈으며, 다음 우선순위는 실제 D455 원본 30장의 식별 가능성 점검과 검수된 부품 mask 확보다.','',
      '**추가로 발견한 오류:** 기존 micro-PCB 설명의 A/H/I와 실제 Raspberry Pi 사진이 일치하지 않았다. G/H/M으로 바로잡아 새로 학습했다. 과거 A/H/I 모델의 Raspberry Pi AP를 정상 성능으로 비교하지 않는다.','',
      '## 실제 학습한 모델','',
      '| 모델 | 역할 | 파라미터 | 전체 run 에폭 | 전체 run optimizer 갱신 | 선택 checkpoint / 입력 | confidence |','|---|---|---:|---:|---:|---|---:|']
    labels={'parts_yolo11n':'4종 부품 bbox, 경량','parts_yolo11s':'4종 부품 bbox, 비교','board_yolo11n':'RPi 보드 bbox','board_yolo11n_seg':'RPi 보드 외곽 mask'}
    for r in model_records:
        s=r['selection']; c=s['candidate']; a=r['architecture']; t=r['run_summary']
        lines.append(f"| {r['model_id']} | {labels[r['model_id']]} | {a['parameters']:,} | {t['last_epoch']} | {int(t['actual_optimizer_steps']):,} | {c['candidate_id']} | {s['operating_confidence']:.2f} |")
    lines+=['','갱신 수는 전체 학습 run의 실제 누적 optimizer counter다. 앞선 에폭의 best.pt가 선택됐다고 해서 이 누적 횟수 전체가 해당 best 파일에 반영됐다는 뜻은 아니다. 체크포인트는 파일 SHA로 식별하고 에폭별 최소/최대 counter는 epoch_updates.json에 보존했다.',
       '',f'기본 부품 추론 모델은 검증셋 기준 **{default_parts}**로 선택했다. 기본 실행은 전체 원본 사진을 사용하며 ROI cascade는 별도 선택 기능이다. ROI cascade와 카메라 전체 처리의 mAP·지연은 아직 측정하지 않았다.','',
       '## 원본 정답 기반 시험 결과','',
       '아래 mAP는 IoU 0.50:0.95, COCO maxDets=100 기준이다. mAP50은 IoU 0.50에서의 AP다. 서로 다른 과제나 시험셋의 숫자를 하나의 모델 순위로 합치지 않는다.','',
       '| 모델 | 시험셋 | 사진 / 정답 | bbox mAP50 | bbox mAP50–95 | mask mAP50–95 | Precision / Recall |','|---|---|---:|---:|---:|---:|---:|']
    for mid in models:
        for name,t in models[mid]['tests'].items():
            if name not in ['general_test','pi_test']:continue
            dataset='Pi3B 앞·뒤 holdout' if name=='pi_test' else ('일반 PCB holdout' if mid.startswith('parts_') else '보드 source-group holdout')
            mask=percent(t['mask']['ap50_95_max100']) if t['mask'] else '과제 아님'
            lines.append(f"| {mid} | {dataset} | {t['images']} / {t['native_gt_instances']:,} | {percent(t['bbox']['ap50_max100'])} | {percent(t['bbox']['ap50_95_max100'])} | {mask} | {percent(t['operating']['precision'])} / {percent(t['operating']['recall'])} |")
    lines+=['','Precision/Recall은 모델별 검증에서 정한 confidence와 box IoU≥0.5 기준이다. mask의 operating recall이 아니다. 부품이 밀집한 사진을 위한 maxDets=300 AP도 JSON과 그래프에 별도로 기록했다.','',
       'Pi 시험은 이번 파인튜닝에서 제외한 1개 보드 그룹의 2장이다. 공개 이미지가 사전학습 자료에 포함됐는지는 확인할 수 없다. 원본 unknown 67개는 4종 정답으로 추측하지 않았다. 이 작은 시험만으로 여러 실물 보드의 일반화나 전체 부품 인식률을 확정할 수 없다.','',
       '### Pi3B 종류별 결과','',
       '| 모델 | 부품 | 정답 | AP50–95 | Recall | TP / FP / FN |','|---|---|---:|---:|---:|---|']
    for mid in ['parts_yolo11n','parts_yolo11s']:
        t=models[mid]['tests']['pi_test']
        for i,(name,ap) in enumerate(t['bbox']['per_class'].items()):
            tp,fp,fn=t['operating']['per_class_counts_tp_fp_fn'][i]
            lines.append(f'| {mid} | {name} | {tp+fn} | {percent(ap["ap50_95_max100"])} | {percent(tp/(tp+fn) if tp+fn else None)} | {tp} / {fp} / {fn} |')
    lines+=['','### 축소·블러 민감도','',
       '같은 Pi3B 189개 정답과 고정 confidence를 사용했다. 절반 크기로 축소 후 원래 크기로 보간한 영상, Gaussian σ=1 영상이다. **D455 촬영 결과가 아닌 합성 조건**이며 모델 선택에 사용하지 않았다.','',
       '| 모델 | 조건 | bbox AP50–95 | Recall |','|---|---|---:|---:|']
    for mid in ['parts_yolo11n','parts_yolo11s']:
        for name,label in [('pi_test','원본'),('pi_test_half_resolution','절반 해상도'),('pi_test_gaussian_sigma1','Gaussian blur')]:
            if name in models[mid]['tests']:
                t=models[mid]['tests'][name]
                lines.append(f'| {mid} | {label} | {percent(t["bbox"]["ap50_95_max100"])} | {percent(t["operating"]["recall"])} |')
    lines+=['','## 추가 공개 사진과 실제 촬영 준비','',
       '[학습에 쓰지 않은 공개 사진 24장 추론 예시](reports/qualitative_external24/README.md)를 별도로 저장했다. 정답과 비교하지 않은 예시이므로 성공률·mAP 수치는 붙이지 않았다. 여러 보드·조명·각도에서 남는 누락 사례를 확인할 수 있다.','',
       '[라벨링·수집 가이드](annotation/target4_labeling_guide.md)에는 30장 파일럿, 조건부 200장 예산, 클래스·경계 규칙과 고유 부품 수 집계 방법을 기록했다. [D455 픽셀 예산](annotation/D455_PIXEL_BUDGET.md)은 실제 intrinsics 측정 전의 예시 계산이다. 신규 D455 사진·target4 검수 mask는 현재 각각 0개다.','',
       '## 구조와 그래프','']
    for png in sorted((ROOT/'reports/figures').glob('*.png')):
        lines+=['!['+png.stem+'](reports/figures/'+png.name+')','']
    lines+=[(ROOT/'RELEASE_NOTES_DRAFT.md').read_text(encoding='utf-8').replace('# Raspberry Pi 촬영 대상: R03 사용 범위','## 데이터·설정·한계의 상세 기록',1),'',
      '## 실행과 파일','',
      '평가·선택·추론 CPU 검사 28개를 통과했다. [최종 평가 감사](reports/final_evaluation_audit.json)는 20개 평가 작업의 원본 정답·운영 TP/FP/FN을 독립 대조하고 Pi n/s bbox와 보드 mask AP를 공식 COCO로 재계산해 일치를 확인했다. [추론 일치 검증](reports/inference_equivalence.json)은 Pi 원본 한 장의 부품 16개 class/bbox/score가 기존 평가 기록과 정확히 같음을 확인한다. 이는 평가 방법과 구현의 검증이며 실사용 성능 합격은 아니다.','',
      '이미지 추론: Python 환경에서 다음 명령을 실행한다. [추론 설명](INFERENCE.md)에 폴더 입력·원본 좌표·보드 mask·선택 JSON 규약이 있다.','',
      '```powershell',"& 'C:/Users/hkjun/Documents/mcu-vision/.venv-yolo11/Scripts/python.exe' -B scripts/infer_rpi.py --input 'C:/images/board.jpg' --output 'C:/results/r03_run01' --device 0",'```','',
      '학습·평가는 같은 PC의 원본 데이터 캐시를 참조한다. raw 사진/ZIP은 이 패키지에 반복 복사하지 않았다. local_data_paths.json에 위치를 남겼다. 모델·추론 코드는 패키지 내부 경로를 쓰지만, 패키지만 다른 PC로 옮겨 학습 자료까지 복원되는 구성은 아니다.','',
      '각 모델의 best.pt/last.pt, 구조, 환경, 설정, 학습곡선, 실제 update, 선택 근거와 체크포인트 SHA를 포함했다. 재학습은 기존 결과를 덮지 않도록 config의 run_dir를 새로운 이름으로 바꾼 뒤 실행한다. 검증·시험 재실행도 새 output-root를 지정한다.','',
      '상세 지표: reports/evaluation_suite/aggregate.json. 원본 GT/예측: 같은 폴더 evaluations. 클래스/부품 수·출처·분할·loss는 model_registry.json과 data 및 component_assets 폴더의 manifest에서 추적한다.','',
      '다음 실제 촬영·라벨 예산과 경계 규칙은 annotation 폴더에 있다. 현재 공개자료 결과를 실사용 승인 또는 target4 mask 완료로 기록하지 않는다.']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'default_parts':default_parts,'report':str(ROOT/'README.md')},ensure_ascii=False))


if __name__=='__main__':main()
