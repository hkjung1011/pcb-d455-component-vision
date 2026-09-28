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
