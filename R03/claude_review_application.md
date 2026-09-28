# Claude 검토 의견 반영 기록

확인일: 2026-09-28. 검토 대상은 사용자가 제공한 `붙여넣은 텍스트.txt`의 「R01 계획 검토 — PCB 부품 검출·세그멘테이션」이다. 원문도 R01 전체를 읽지 못하고 요약을 검토했다고 밝힌다. 제안은 검토 자료로 취급했으며, 사용자의 20epoch 범위와 현재 실제 관측을 우선했다.

## 반영·수정·기존 해결 구분

| 의견 | 판정 | 실제 반영과 해석 |
|---|---|---|
| 원본 유효 픽셀·초점이 작은 부품의 핵심 제약 | 반영 | 원본 해상도·ROI·native tile geometry를 보존한다. 타일은 전체 화면 축소로 잃는 픽셀을 줄이지만 촬영 시 없었던 정보를 생성하지 않는다. D455 실측은 아직 아니다. |
| 8px 미만 검출 불가, 16px 미만 윤곽 불가에 가까운 설명 | 수정 | stride와 물체 최소 크기는 같은 개념이 아니다. 8/16px는 평가 구간·파일럿 판단용 경험칙이며 하드 제한으로 쓰지 않는다. 작은 정답을 버려 지표를 올리지 않는다. WACV 학습 타일의 8px 미만 양수 조각도 보존했다. |
| D455에서 저항 클래스가 성립하기 어렵다는 단정 | 수정 | 공개 데이터로 네 종류 검출을 개발하되 D455 적용 범위는 원본 픽셀·초점·조명 파일럿 후 결정한다. D455 실제 영상이 없으므로 가능/불가를 확정하지 않는다. |
| GSD=Z/fx, 초점·노출·조명 확인 | 다음 실측 항목으로 반영 | 보정 intrinsics, 촬영거리, 실물 자, blur/클리핑 측정과 눈으로 작은 부품을 확인해야 한다. 공개 JPEG 해상도를 D455 원본 해상도라고 부르지 않는다. 카메라/IR 설정을 실제로 바꿨다는 주장은 하지 않는다. |
| smd_v1 class ID 재사용 금지 | 반영 | 새로운 target4 순서 resistor/capacitor/ic/connector를 명시하고 기존 capacitor=0/resistor=1 등의 체계와 분리한다. WACV bbox와 PCBVision semantic3도 구분한다. |
| other_component를 반드시 추가 | 수정 | source type을 확인할 수 있는 범주만 사용한다. WACV의 비대상 라벨은 이번 target4 과제의 배경이며 unknown을 자동으로 other 또는 R/C로 추측하지 않는다. Pi3B unknown67개와 정답 불완전성 제한을 명시한다. 새로운 other 클래스는 전문가 정의·라벨 없이 만들지 않는다. |
| IC를 패키지 모양으로 정의 | 수정 | 부품 종류와 패키지 속성을 분리한다. 모양만으로 IC·트랜지스터·전압조정기를 재분류하지 않는다. 이번 공개 자료는 원본 전문가 type을 유지한다. |
| 몸체·리드 경계 규칙, 폴리곤→bbox 일관성 | 일부 적용/실물 라벨링 전 확정 필요 | WACV는 원본 사람 bbox이므로 실제 mask로 바꾸지 않는다. 향후 target4 polygon 가이드에서 리드/패드/솔더/그림자 규칙을 별도 버전으로 고정해야 한다. 단순 사각형을 정답 mask로 대체하지 않는다. |
| 사람 간 mask IoU는 모델 성능의 상한 | 수정 | 이중 라벨의 일치도는 경계 애매함과 라벨 불확실성의 지표다. 모델 AP 또는 IoU의 엄밀한 수학적 상한은 아니다. 이중 라벨 측정은 아직 수행하지 않았다. |
| 장수보다 보드 다양성과 고유 부품 수가 중요 | 반영 | WACV 원본47장/이름기준29그룹, train18·val5·일반test5·Pi test1그룹. 원본 annotation 수와 타일 반복 노출을 별도로 기록한다. 물리 시리얼 독립성은 미검증이다. |
| 90/220epoch로 확대 | 미적용 | 사용자의 짧은 평가 예산에 맞춰 모델당 최대20epoch를 코드로 제한한다. 에폭 확대 대신 실제 optimizer update·가중치 변경·검증 지표를 기록한다. 자동 연장하지 않는다. |
| 기본 nbs64 때문에 업데이트 부족 | R02에서 이미 수정, R03 유지 | 초기 AMP smoke3의 실제 update0과 baseline20 update6을 발견해 FP32+nbs8+warmup_bias_lr0으로 고쳤다. 수정된 R02 source3 20epoch는 실제104회 업데이트였다. R03도 실제 optimizer state의 step을 에폭별로 저장한다. 새 데이터 규모가 달라 이전 횟수를 그대로 예측치로 쓰지 않는다. |
| optimizer=auto가 lr0를 덮어씀 | 반영 | R03 부품 검출 설정은 AdamW를 명시하고 lr0=.001, nbs8, FP32, batch2, imgsz1024를 기록한다. box/cls/dfl gain과 클래스별 추가 가중치1.0도 구분한다. |
| 작은 물체를 지나치게 줄이는 증강 제한 | 반영 | R03 설정 scale.2, saturation.15, mosaic/mixup/copy-paste0. 회전/이동 등 실제 적용값은 config와 학습 args에 기록한다. 상하 반전은 무조건 안전하다고 가정하지 않고 이번 값은0이다. |
| YOLO11n-seg 및 YOLO26 비교 | 범위 조정 | 비교 후보를 사전에 정하고 검증셋으로 선택한다. 이번 WACV 정답은 bbox만 있으므로 해당 모델 task는 detect다. seg 데이터와 다른 과제 지표를 같은 순위표로 합치지 않는다. YOLO26 우수성을 홍보 문구만으로 가정하지 않는다. |
| 내장 mask val와 원본 mask COCO를 구분 | 반영 | native_evaluate.py는 원본 GT polygon을 원본 크기로 rasterize하고 retina_masks 출력의 H×W 일치를 검사한다. shape가 다르면 리사이즈로 숨기지 않고 중단한다. WACV는 bbox AP만 평가한다. |
| COCO 작은 물체 외 별도 크기 구간 | 반영 | 원본 GT bbox 짧은 변 <8 / 8–16 / 16–32 / ≥32px 재현율을 기록한다. 이것은 해당 공개 원본에서의 결과이며 D455 한계값이 아니다. |
| 검증셋에서 confidence 고정 후 test | 반영 | validation의 .05~.95, .05 간격 micro-F1(box IoU≥.5)로 operating point를 선택한다. test는 명시적 confidence 없이는 실행하지 못한다. mask AP와 operating box 지표를 구분한다. 모델/whole-vs-tile 선택도 val만 사용한다. |
| 보드 bootstrap 95% CI | 제한을 명시해 반영 | 구현은 고정 threshold recall의 그룹 bootstrap CI이다. AP의 CI라고 부르지 않는다. Pi는 보드그룹1개여서 재표집 구간이 일반화 불확실성을 설명할 수 없다. |
| 500프레임 촬영→후처리 지연 | 미실측 | 현재 evaluator는 warmup 제외, 이미지 decode/촬영 제외의 offline predict+postprocess p50/p95만 낸다. 이를 D455 실시간 FPS 또는 카메라 end-to-end 지연으로 바꾸어 말하지 않는다. |
| 같은 보드·설계·세션·파생 이미지 누수 통제 | 반영/일부 미검증 | WACV는 원본 이름으로 그룹 고정 후 train에만 타일을 만든다. RPI3B 두 면은 모두 최종 시험에 남겼다. 공개 사진의 실제 개체·촬영세션은 일부 알 수 없으므로 미검증 상태를 기록한다. pHash 자체도 독립성의 증명은 아니다. |
| 시험 라벨을 모델 제안으로 만들지 않음 | 반영 | 이번 WACV 시험 GT는 원본 사람이 제공한 bbox다. 예측으로 추가 GT를 만들지 않았다. 공개 semantic connected regions를 물리 인스턴스 정답으로 승격하지 않았다. |
| FPIC·PCBVision·PCB-SAID 재검토 | 수행 | FPIC는 원본 등록 요구/권리 미확인으로 미사용. PCBVision53은 확보된 CC BY4.0 source3 semantic 자료지만 저항·instance ID가 없다. PCB-SAID는 무인증 파일·라이선스 미확인. 추가로 WACV 원본47장과 Pi3B 부품 bbox를 실제 확보했다. |
| 공개 dataset은 train에만 사용 | 조정 | 원본 공개 데이터의 보드 holdout은 공개 벤치마크 개발 시험으로 쓸 수 있다. 다만 D455/새 실물 일반화를 증명하지 않는다. 최종 D455 주장을 위한 별도의 실물 수집 시험은 여전히 필요하다. |

## 코드·데이터 검증 근거

- `component_assets/component_records.json`: source annotation5,896개, class 매핑, 사진/XML SHA, raw VOC·변환 xyxy, 분할, unknown 등 제외 type 보존.
- `component_assets/preparation_verification.json`: YOLO445파일/9,145bbox 왕복 오차 최대4.11e-7px, 타일 생성 시 원본 target 누락0, 보드그룹 교차0.
- `scripts/test_native_evaluate.py`: CPU 검사10개 통과. 완벽한 native bbox/mask AP=1, 없는 클래스 AP=null, 빈 예측 AP=0, 밀집150객체에서 max100과 max300 차이, binary mask dtype/빈 shape, 클래스별1:1매칭, 음성 이미지 오검출률, 크기 경계 검증.
- `scripts/native_evaluate.py`: COCO category ID 순서와 클래스 이름 연결을 수정했고, loadRes가 원본 예측 evidence를 변경하지 않게 복사한다. `--cohort pi_test`로 Pi 시험을 일반 test와 구분한다.
- `../pcb_components/runs/source3_fp32_pilot20/run_summary.json`: 이전 source3 semantic 파일럿20epoch/실제104update. 이 결과를 target4 instance mask mAP로 제시하지 않는다.

현재의 확인은 데이터 변환·평가 수학·오프라인 코드 검증이다. D455 촬영 성공, 네 부품 종류의 실제 경계 분할, 실물 보드 일반화, 실제 장비 지연은 별도 증거가 필요하다.
