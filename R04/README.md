# PCB D455 · R04 검토 반영과 비교 실험

**공개자료에서 기준/개선 YOLO11s 두 모델을 실제 학습·평가했다. 검증셋 기준 선택은 `improved`, 16번째 epoch, `dual`, confidence 0.65다.** Raspberry Pi 개발 holdout에서 AP50–95@300은 42.08%, 고정 confidence recall은 77.25%다. 실제 D455와 저항 포함 4종 부품 mask는 아직 검증하지 못했다.

합성 축소 사진에서는 선택 모델의 인식률이 크게 떨어졌다. nominal_3p2_pxmm: AP50–95@300 15.40%, 고정 confidence recall 3.17% / nominal_2p1_pxmm: AP50–95@300 14.30%, 고정 confidence recall 2.12%. 이 조건은 보드 ROI를 확보하고 확대한다는 가정의 진단이며 D455 실측이나 성능 상한·하한이 아니다. 공개 고해상도 사진의 점수만으로 실사용 가능하다고 판단하지 않는다.

## 성능과 평가 범위

| 모델 / 시험 | 사진 / GT | bbox mAP50–95@100 | bbox mAP50–95@300 | 고정 confidence P / R |
|---|---:|---:|---:|---:|
| baseline / general_test | 11 / 1348 | 33.10% | 36.02% | 82.21% / 51.41% |
| baseline / pi_test | 2 / 189 | 35.35% | 35.33% | 81.30% / 52.91% |
| improved / general_test | 11 / 1348 | 39.56% | 42.77% | 83.78% / 62.46% |
| improved / pi_test | 2 / 189 | 42.04% | 42.08% | 76.04% / 77.25% |

@100/@300은 **이미지 전체 최대1,000개로 먼저 제한한 저장 예측에서 COCO의 영상·클래스별 최대100/300개를 평가한 조건**이다. 제한하지 않은 예측 대비 AP 차이는 측정하지 않았다. 원본 좌표와 원본 target4 정답을 그대로 사용했다. 운영 P/R은 box IoU≥0.5와 검증셋에서 고정한 전역 confidence 기준이다. AP는 낮은 confidence부터의 점수 곡선이므로 운영 P/R과 목적이 다르다.

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
| train | 25 | 17 | 1201 / 1584 / 204 / 318 |
| val | 9 | 6 | 410 / 454 / 73 / 115 |
| test | 11 | 5 | 443 / 636 / 93 / 176 |
| pi_test | 2 | 1 | 57 / 104 / 16 / 12 |

새 사람이 만든 라벨은0개, 새 mask는0개다. 기존 공개 bbox5,896개를 재사용했고 그중 학습원본3,307개를 여러 crop으로 노출했다. 영상 annotation, crop 중복 노출, 실제 draw 노출을 구분해 저장했다.

클래스 loss multiplier는 각각1.0, loss gain은 box7.5/cls0.5/dfl1.5다. 17개 학습 보드그룹은 기대 샘플 확률이 각각1/17이며 그룹 내 IC/커넥터 포함 view에1.5배 우선순위를 줬다. 그룹별 weight 합을1로 정규화하고 epoch당448회 복원 추출했다. 이는 클래스 gradient 비율을1.5배로 보장하지 않는다. 실제 draw와 클래스 노출은 `runs/*/sampled_exposures.json`, 모든 view의 weight는 `data/*/views.json`에 있다.

| 모델 | 학습 epoch | 실제 optimizer 호출 | 실제 view draw | 선택 epoch / 호출 | 추론 / confidence |
|---|---|---:|---:|---:|---|
| baseline | 20번째 진입 / 마지막 일부 | 1,165 | 8,872 | 20 / 1165 | dual / 0.75 |
| improved | 20번째 진입 / 마지막 일부 | 1,165 | 8,872 | 16 / 952 | dual / 0.65 |

각 모델 최대20에폭·1,165회 직접 optimizer 호출에서 멈췄다. 20번째 epoch는90/112배치까지만 처리했다. AdamW parameter별 step과 직접 호출 수를 별도로 기록했다. FP32, 1024px, batch4/nbs8, AdamW lr0.001/lrf0.01, warmup2, seed42, RTX5060 Laptop8GB를 사용했다. 3에폭은 동작 점검에 유용하지만 실사용 성능을 입증하는 기준이 아니다.

## Raspberry Pi 클래스별 인식

| 모델 | 종류 | 찾은 GT / 전체 | Recall |
|---|---|---:|---:|
| baseline | resistor | 37/57 | 64.91% |
| baseline | capacitor | 54/104 | 51.92% |
| baseline | ic | 9/16 | 56.25% |
| baseline | connector | 0/12 | 0.00% |
| improved | resistor | 46/57 | 80.70% |
| improved | capacitor | 88/104 | 84.62% |
| improved | ic | 9/16 | 56.25% |
| improved | connector | 3/12 | 25.00% |

원본 connector 정의에는 미실장 RUN 패드 annotation2개가 포함된다. 점수를 높이려고 기존 GT에서 삭제하지 않았다. 새 D455 실장부품 정의는 별도 annotation 버전과 검수로 관리한다. unknown 객체가 있는 데이터이므로 오검출의 의미와 전체 부품 coverage에 한계가 있다.

## 알고리즘과 실제 그래프

[그래프 목록](reports/figures/README.md) · [Claude 검토 반영표](reports/CLAUDE_REVIEW_RESPONSE.md) · [원본 데이터 독립 감사](reports/data_independent_audit.md) · [최종 결과 감사](reports/final_results_audit.md)

[모델별 구조와 역할](reports/MODEL_FAMILY.md) · [어떤 부품을 찾고 놓쳤는지](reports/error_analysis.md) · [실행 코드의 고정 예측 재현 확인](reports/inference_equivalence.json)

[추가 분석 규칙·독립 감사·그룹별 안정성·FP 분류](reports/analysis_addendum_v1_1/README.md)에 검증 시점과 재현 자료를 기록했다. 보조 진단은 선택된 모델·confidence나 1차 지표를 다시 고르는 데 사용하지 않았다.

[합성 해상도 민감도 시험](reports/resolution_stress/README.md)은 별도 사전 고정 조건의 진단이다. 사진별 배율 근사값으로 축소했으며 실제 D455 광학·초점·노이즈를 재현한 결과나 성능 하한이 아니다.

[모델·예산 CSV](models.csv) · [전체 지표 CSV](test_metrics.csv) · [클래스 지표 CSV](class_metrics.csv) · [검증 선택/평가 원본](reports/evaluation_suite/aggregate.json)

[원본 라벨·반복 노출·클래스 가중치 장부](label_weight_ledger.csv) · [그룹별 확률과 실제 학습 횟수](group_sampling_ledger.csv)

## 실사용 준비와 재현

[오프라인 실행법](INFERENCE.md)의 고정 모델·추론 파이프라인으로 새 이미지를 처리할 수 있다. 실제 D455 연결이나 실시간 검증을 수행한 것은 아니다. 합성 축소를 D455 성능 하한이라고 부르지 않는다. Depth Min-Z와 RGB 초점/부품 판독성을 구분하며, 실제 영상의 intrinsics·거리·초점·조명·원본 부품 픽셀을 측정해야 한다.

다음은 실제D455 30장 파일럿, 조건부200장 수집·원본4종 bbox/instance mask 검수, 실물/세션별 독립분할이다. 수량은 계획이며 아직 확보한 데이터가 아니다. 한 대만 있으면 새로운 실물 일반화가 아닌 미관측 촬영세션 평가로 명시한다. 상세 조건과 예산은 검토 반영표에 있다.

이 패키지는 체크포인트20개, 공식 초기 가중치, 코드·라벨·weight·평가·그래프를 담는다. 원본 WACV 사진·타일이미지·다운로드 ZIP·가상환경·인증정보는 포함하지 않는다. 학습 재현에는 원 출처 자료와 기록된 환경이 필요하다. metadata의 절대경로는 실행 당시 증거이며 다른 PC에서 자동 유효하지 않다. 오프라인 추론의 선택 registry는 패키지 상대경로를 쓴다.

R03/R02 기록과 GitHub 이전 Release는 유지한다. 공개자료·Ultralytics 코드/가중치의 출처와 라이선스는 R03 기록 및 원 프로젝트를 따른다. 비공개 보관이 새로운 재배포 권한을 만들지는 않는다.
