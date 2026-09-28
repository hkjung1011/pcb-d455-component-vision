# Claude의 FP 중첩·낮은 점수 후보 주장 재검증

2026-09-28. R03 s 모델의 저장 예측을 CPU에서 다시 계산했다. GPU·새 추론·학습·threshold 변경·원본 정답 수정은 하지 않았다. 입력 SHA는 실행 전후 일치했다.

## 고정 운용점과 분모

원본 metrics의 confidence **0.35000000000000003**을 그대로 사용했다. 점수 내림차순으로 동일 클래스의 아직 매칭하지 않은 GT에 IoU≥0.5를 적용했다. 클래스별 TP/FP/FN은 기존 결과와 일치했다. Pi FP **36개**, 일반 FP **252개**라는 분모는 확인됐다.

## 제외 객체와의 중첩

원문은 FP가 제외 객체 “위에 있다”고 했으나 중첩 지표·문턱값·text 제외 여부·개별 대응 목록을 제공하지 않았다. 다음은 원본 excluded_source_objects에 대해 정의를 명시하고 계산한 값이다. 하나의 FP가 여러 제외 객체에 겹쳐도 FP는 한 번만 센다.

| 규칙 | Pi FP 36개 중 | 일반 FP 252개 중 |
|---|---:|---:|
| 최대 bbox IoU ≥ 0.5 | 21 | 98 |
| 교차 면적 / 예측 bbox 면적 ≥ 0.5 | 24 | 116 |
| 예측 중심이 제외 bbox 안에 있음 | 26 | 129 |
| 중심 포함, text/component text 제외 | 24 | 116 |

**24/36은 일부 정의에서 재현되지만, 119/252는 위 정의에서 재현되지 않았다.** 여러 정의를 조합해 119에 맞추지 않는다. 원래 코드나 개별 대응 목록이 있어야 원문의 기준을 확정할 수 있다. IoU≥0.5를 공통 기준으로 보고하려면 **21/36·98/252**를 사용한다.

제외 객체에는 unknown, 확인된 비대상, 패드와 문자 영역 등이 섞여 있다. FP와 겹친다는 사실은 원본 라벨 오류나 해당 비대상을 학습에서 전부 무시해야 한다는 증거가 아니다. unknown 검토와 확실한 비대상에 대한 정상적인 오검출 억제를 분리한다.

## Pi에서 놓친 IC·커넥터의 낮은 점수 후보

운용점에서 FN인 GT별로 **동일 클래스이며 IoU≥0.5인 저장 예측**을 모두 찾고 가장 높은 점수를 기록했다. 저장 파일은 raw floor0.001과 NMS를 거친 결과다. 모델의 모든 raw 후보를 본 것은 아니다.

| 클래스 | 운용점 FN | 같은 클래스·IoU≥0.5 후보 있음 | 후보 최고 점수 범위 | floor에서 1:1 매칭 추가 회수 |
|---|---:|---:|---|---:|
| ic | 13 | 9 | 0.002995 ~ 0.327841 | 9 |
| connector | 12 | 8 | 0.017761 ~ 0.253553 | 8 |

**동일한 명시 조건에서는 커넥터 8/12, IC 9/13으로 계산된다.** 원문의 7/12와 8/13보다 각각 1개 많다. 각 GT의 최고 후보는 서로 다른 예측이며, 하나의 후보를 중복 사용한 수치가 아니다. 0.001까지 내려 점수순 1:1 매칭을 이어도 각각 8개·9개가 회수된다. 이 후보의 점수는 모두 현재 threshold보다 낮다.

SoC `RPI3B_Top ic U1`은 score **0.002994568**, IoU **0.582502496**으로 “약 0.003”이라는 개별 주장은 맞다. Top RUN 미실장 패드 annotation도 source connector 정의대로 포함했다. 정답을 삭제하지 않았다.

| 영상 / 원본 이름 | class | 최고 후보 score | IoU |
|---|---|---:|---:|
| RPI3B_Bottom / ic U18 | ic | 0.015091228 | 0.535137 |
| RPI3B_Bottom / ic U15 | ic | 0.327840596 | 0.782586 |
| RPI3B_Bottom / connector J9 | connector | 0.253552854 | 0.824406 |
| RPI3B_Bottom / ic U19 | ic | 0.038908221 | 0.684827 |
| RPI3B_Bottom / connector unknown | connector | 없음 | — |
| RPI3B_Top / connector J1 | connector | 없음 | — |
| RPI3B_Top / ic U4 | ic | 없음 | — |
| RPI3B_Top / ic U9 | ic | 0.097537994 | 0.653715 |
| RPI3B_Top / connector J6 | connector | 없음 | — |
| RPI3B_Top / ic U16 | ic | 0.046004634 | 0.896609 |
| RPI3B_Top / ic U8 | ic | 없음 | — |
| RPI3B_Top / ic U3 | ic | 없음 | — |
| RPI3B_Top / connector J4 | connector | 0.017761031 | 0.585035 |
| RPI3B_Top / ic U13 | ic | 0.274771452 | 0.752218 |
| RPI3B_Top / connector RUN | connector | 0.229119316 | 0.725778 |
| RPI3B_Top / connector J12 | connector | 0.144461557 | 0.672742 |
| RPI3B_Top / connector J11 | connector | 0.143429950 | 0.712803 |
| RPI3B_Top / connector J10 | connector | 0.018173154 | 0.680189 |
| RPI3B_Top / ic U11 | ic | 0.138601825 | 0.746319 |
| RPI3B_Top / ic U10 | ic | 0.048265208 | 0.645276 |
| RPI3B_Top / connector J7 | connector | 0.116058178 | 0.628289 |
| RPI3B_Top / connector J3 | connector | 0.027755817 | 0.625201 |
| RPI3B_Top / ic U17 | ic | 없음 | — |
| RPI3B_Top / connector J8 | connector | 없음 | — |
| RPI3B_Top / ic U1 | ic | 0.002994568 | 0.582502 |

낮은 점수가 누락에 기여한다는 진단은 지지되지만, threshold를 낮추면 새로운 FP도 늘어난다. 후보 수를 threshold 조정 후 정확도나 재현율 보장으로 읽지 않는다. threshold는 val에서만 정하고 시험 자료로 조정하지 않는다. 높은 점수의 J9 microSD→IC 혼동도 있어 점수 부족 하나로 원인을 단정하지 않는다.

각 FP의 저장 prediction index, 예측 좌표, 제외 객체 source ID와 중첩 값, 모든 IC/커넥터 FN의 대응 후보는 `claude_fp_claim_check.json`에 기록했다.
