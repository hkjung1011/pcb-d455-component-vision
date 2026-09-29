# PCB · 보드 · 웨이퍼 비전 연구 기록

## 2026-09-30 · 보드 1,000장 50에폭 학습 완료

**YOLO11s 8클래스, train 650 / val 150 / test 200장, 실제 50에폭, 검증 선택 epoch 48.** 개발 test 1회 결과는 **mAP50 84.72% / mAP50–95 72.16%**다.

[설정·결과 전체](BOARDS1000_2026-09-30/README.md) · [복원·사용](BOARDS1000_2026-09-30/REPRODUCE.md) · [학습 설정](BOARDS1000_2026-09-30/CONFIGURATION.md) · [클래스별 결과](BOARDS1000_2026-09-30/RESULTS.md) · [현재 상태](BOARDS1000_2026-09-30/STATUS.json) · [가중치와 정확한 데이터 Release](https://github.com/hkjung1011/pcb-d455-component-vision/releases/tag/boards1000-2026-09-30)

전체 설정, 환경 버전, 실제 코드, 1,000장 분할/라벨/그룹/해시, 학습 로그, 곡선과 평가를 보존했다. Release에는 best/last/공식 초기 가중치와 정확한 1,000장·YOLO 라벨, 이전 10에폭 모델이 있다. 이전 파일럿은 [별도 이력](BOARDS1000_2026-09-30/history/pilot10/README.md)으로 남겼다.

test는 이전 실험 사진을 재분할한 개발 평가이며 새 최종시험이 아니다. Jetson AP50–95 30.44%가 주요 약점이다. Nucleo·포트는 이번 모델에 없고, 이 모델의 D455 실측 성능은 미검증이다. 기존 D455 실시간 모델은 변경하지 않았다. 아래 준비 기록의 `training_ready=false`는 원래 9클래스/포트 스냅샷이며 이번 8클래스 완료 상태와 구분한다.

## 2026-09-29 · 보드 종류·포트 데이터 준비

[준비 결과·매핑·재실행 안내](BOARDS_PORTS_PREP_2026-09-29/README.md) · [준비 상태](BOARDS_PORTS_PREP_2026-09-29/STATUS.json) · [업로드 검증](archive/verification_2026-09-29/boards_ports_preparation_upload_verification.json) · [검증 결과](BOARDS_PORTS_PREP_2026-09-29/reports/evidence/verification.json)

Roboflow ZIP 3개 1,509장을 확보하고, IoTKITs와 합쳐 보드 2,013장(train 1,701/val 206/test 106)을 구성했다. Nucleo 393장은 모두 train이므로 val/test 지원이 없고, 포트 549장은 누락 라벨 검수가 남아 있다. **이번 추가 작업은 학습·test 평가를 실행하지 않았으며 두 모델 모두 `training_ready=false`다.** 코드·매핑·출처·해시·검수 기록을 보존했다.

## 2026-09-29 · D455 실시간 보드·부품 시험 추가

[현재 카메라 모델·평가·실행 방법](D455_LIVE_2026-09-29/README.md) · [모델 카드](D455_LIVE_2026-09-29/MODEL_CARD.md) · [평가 JSON](D455_LIVE_2026-09-29/evaluation/summary.json)

기존 R04 IC 검출에 로컬 보드·포트 시험 모델을 함께 적용했다. 기록한 D455 프레임에서 **보드 2개·부품 후보 21개**, 화면 추론 약 **4.56 fps**를 관찰했다. 추가 모델의 실제 가중치와 촬영 증거·학습 기록·검증 코드를 이 폴더에 보존했다.

**실제 카메라 정확도/AP는 미측정**이다. 추가 모델은 같은 보드 두 장의 단일 원본 장면으로 만든 합성 train/val을 사용했다. 개발셋 mAP50–95 98.04%를 실제 정확도로 해석하지 않는다. 저항·커패시터 누락과 다른 배치에서의 일반화 검증은 남아 있다. 기존 R03/R04·웨이퍼 모델과 평가 기록은 보존한다.

[현재 결과부터 보기](CURRENT_SUMMARY.md)

2026-09-29 기록 보완: [PCB 업로드 검증](archive/verification_2026-09-29/pcb_r04_upload_verification.json) · [웨이퍼 업로드 검증](archive/verification_2026-09-29/wafer_r01_upload_verification.json) · [웨이퍼 감사 정정 범위 확인](archive/verification_2026-09-29/wafer_post_test_amendment_scope_check.json).

업로드 검증 JSON은 각 Release를 만든 시점의 커밋과 파일 해시를 담은 기록이다. 후속 문서 커밋으로 main이 이동해도 해당 Release의 근거로 보존한다. 웨이퍼 감사의 0면적 검출 허용 정정은 입력 검사에만 적용했고, IoU·매칭·AP 함수가 그대로임을 추가 대조했다. 모델·분할·시험 지표와 기존 Release ZIP은 변경하지 않았다.


## 2026-09-29 · 웨이퍼 표면 결함 R01 추가

현미경 공개자료의 **YOLO11n 10에폭** bbox 파일럿을 별도로 학습했다. 실제 optimizer 호출 **1,932회**, 학습 미사용 이미지 287장·bbox 496개에서 **mAP50 73.73% / mAP50–95 36.42%**다.

| 분할 | 원본 이미지 | bbox |
|---|---:|---:|
| train | 1472 | 2418 |
| val | 275 | 480 |
| test | 287 | 496 |

원본 후보 2,132장·bbox 3,693개 중 픽셀이 같고 라벨이 충돌하는 98장을 격리해 2,034장·bbox 3,394개를 사용했다. 여섯 결함 종류의 기존 공개 라벨을 사용했고 신규 수작업 라벨·mask는 0개다. **D455 실측 성능, 물리 웨이퍼/lot 독립성, instance segmentation은 검증하지 않았다.** PCB 점수와 직접 우열 비교하지 않는다.

- [웨이퍼 결과·설정·알고리즘·그래프](WAFER_R01/README.md) · [실제 최신 상태](CURRENT_STATUS.json)
- [시험 지표](WAFER_R01/reports/test_metrics.json) · [독립 결과 감사와 고정 confidence0.25 보조 분석](WAFER_R01/reports/independent_final_results_audit.md)
- [데이터 독립 감사](WAFER_R01/reports/independent_data_audit.json) · [모델이 포함된 비공개 Release](https://github.com/hkjung1011/pcb-d455-component-vision/releases/tag/wafer-r01-2026-09-29)
- [보존된 PCB R04](R04/README.md) · [PCB R04 상태](R04/CURRENT_STATUS.json) · [PCB R03](R03/README.md)

Git의 `WAFER_R01/`에는 가중치가 생략돼 있다. 가중치는 Release의 `WAFER_Surface_R01.zip`에서 복원한다. 원본 사진·VOC XML·변환 라벨·예측 사진은 업로드하지 않는다. 데이터의 명시적 라이선스가 없어 원본 재배포는 하지 않았다.

---

## 보존된 PCB R04 완료 당시 기록

# Raspberry Pi PCB · D455 인식 개발 기록

**2026-09-29 · 최신 R04 · 비공개 연구 기록**

Raspberry Pi는 촬영·인식 대상이다. Claude 검토를 원본·코드와 대조하고, RTX5060 Laptop8GB에서 기준/개선 YOLO11s를 같은 예산으로 학습했다. 각 모델 최대20에폭, 실제 optimizer1,165회이며 마지막epoch는일부만진행했다. R03/R02와 과거 Release는 보존했다.

## 최신 실제 지표

| 모델 / 개발 holdout | bbox mAP50–95@100 | bbox mAP50–95@300 | 고정 confidence P / R |
|---|---:|---:|---:|
| baseline / general_test | 33.10% | 36.02% | 82.21% / 51.41% |
| baseline / pi_test | 35.35% | 35.33% | 81.30% / 52.91% |
| improved / general_test | 39.56% | 42.77% | 83.78% / 62.46% |
| improved / pi_test | 42.04% | 42.08% | 76.04% / 77.25% |

@100/@300은 이미지 전체 최대 1,000개 검출을 보존한 뒤 적용한 COCO 영상·클래스별 검출 한도다. 추천은 **improved, epoch16, dual, confidence0.65**로 검증셋에서만 결정했다. 40개 checkpoint·추론조합을 검증하고 선택 SHA를 고정한 뒤 일반11장/5그룹, Pi3B2장/1그룹을 평가했다. 이 그룹은 학습·검증에서 제외했지만 R03에서 이미 오류를 살펴본 **개발 holdout**이며 새 최종시험은 아니다.

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
- [R03 공개자료4개모델](R03/README.md) · [R03 Release](https://github.com/hkjung1011/pcb-d455-component-vision/releases/tag/r03-2026-09-28) · [R02 보조semantic기록](history/PCB_D455_R02_개발결과.md)

모델20개 후보와 초기 가중치는 [R04 비공개 Release](https://github.com/hkjung1011/pcb-d455-component-vision/releases/tag/r04-2026-09-29)의 `PCB_D455_R04.zip`에 있다. Git은 가중치를 제외한 탐색용 기록이며 clone만으로 추론 가중치가 복원되지 않는다. 원본 데이터 사진·타일이미지·다운로드ZIP·가상환경·인증 설정은 업로드하지 않았다. metadata의 절대경로는 당시 실행 증거이며 새PC에서는 자료와 경로 설정이 필요하다.

다음은 실제D455 30장광학파일럿과 별도200장수집·실물/세션분할·검수된4종bbox/mask다. 이 수량은 계획이며 확보된 실적이 아니다. 출처와 라이선스는원자료/R03기록을따르며 비공개보관이추가재배포권한을부여하지않는다.

