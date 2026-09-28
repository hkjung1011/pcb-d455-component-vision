# R04 분석 부록 v1.1 — 초안 (DRAFT · 미동결)

> **일부 validation 확인 후, R04 holdout 재평가 전에 고정한 보조 분석 계획.**
> 결과를 보기 전의 사전 등록이 아니다. 기존 모델 선택 규칙과 1차 지표는 바꾸지 않는다.

| 항목 | 내용 |
|---|---|
| 상태 | 초안. 사용자 확인 전이며 해시로 동결하지 않았다. |
| 작성 | 2026-09-29, 검토자 Claude |
| 기준 기록 | GitHub `hkjung1011/pcb-d455-component-vision` 커밋 `17f72fdd9143bca6645f9b342414b0c5a2c1ed6d`의 `R04_PAUSED/`와 로컬 `work/r04`(가중치·manifest·저장 예측). 두 곳의 공통 파일은 바이트 단위로 같다(부록 R01). |
| 이번 작성에서 한 일 | 문서·규칙 작성과 CPU 읽기 전용 재현 스크립트 실행. 추가 학습, GPU 평가, 기존 파일 수정은 하지 않았다. |
| 재현 자료 | [REVIEW_APPENDIX.md](REVIEW_APPENDIX.md), `repro/` |

## 0. 대상 v1 식별자 — 변경하지 않음

| 대상 | 파일 | SHA-256 |
|---|---|---|
| 학습 프로토콜 | `work/r04/protocol.json` | `c712d8f3e24ce8fa8291d4b2e7e2593924ccdb154ba4361f50958e63555a4def` |
| 평가 suite fingerprint | `reports/evaluation_suite/protocol.json`의 `protocol_sha256` 값 | `a45e4d372f507af24518000b37e7bd39423eafdf290b46fe8072332855c325e1` (파일 SHA는 `e52f35a7…`, 정정 E5) |
| evaluator | `scripts/evaluate_r04.py` | `b62dce3fa1e3b8a0fe0b4526809bde5bf63f914e0e53ac8398f2f29b1c35b3e5` |
| orchestrator | `scripts/run_r04_evaluation.py` | `82e5a95fd5c2c14e827761c923786be15e7ac790f5df622678ae96f2dbd57f65` |
| 전체 감사 | `scripts/audit_r04_results.py` | `90e4155d40e3ce8f420290704e567291c53534c5c6924dd96d894ca19c9b7799` |
| 해상도 시험 프로토콜 | `stress_protocol.json` | `7afc529c3ddb6131ba5ee2db9281f8ff8a851407f165799cbdd5b6840d9fbb1d` |
| native manifest | `data/native_manifest.json` | `a91d4956da6adc8b3eebcd87da6e8e428362e1c14cef9e8f9d5a4583817d8c19` |
| 후보 체크포인트 목록 | `runs/{baseline,improved}/candidates.json` | `37326c24…7510` / `d9d154ea…90d4` (체크포인트 20개 SHA 일치, 부록 R01) |

## 1. 이 부록의 시점과 해석 범위

### 1.1 작성 시점에 이미 본 자료와 아직 보지 않은 자료

| 이미 본 자료 | 아직 보지 않은 자료 |
|---|---|
| R03의 일반·Pi holdout 결과와 오류 분석. R04 설계(val 재구성, ignore, 문맥 crop)에 반영됨 | R04 val 나머지 18개(improved epoch 4–20) |
| R04 학습 기록 전체(두 팔) | R04 holdout 4개(두 팔 × 일반/Pi)의 예측과 성능 |
| R04 val 22/40 결과(baseline 20개, improved epoch 2의 2개)와 저장 예측. 검토 중 재계산에 사용 | 합성 해상도 시험 결과 |
| 데이터 집계: holdout GT의 개수·크기 구간·그룹 비중·제외 객체 수. 예측이나 성능은 아님 | |

따라서 이 부록은 **결과를 보기 전의 사전 등록이 아니다.** 전역 상한 표기, 그룹 편중 진단, FP 분류 같은 규칙은 이미 본 val 22개와 R03 결과에서 착안했다. R04 holdout은 R03에서 이미 본 **개발용 holdout**이며, 처음 보는 최종 시험이 아니다.

### 1.2 v1에서 바꾸지 않는 것

1. **모델 선택 규칙**: val AP50–95 max300 내림차순 → max100 내림차순 → offline p50 오름차순 → candidate id.
2. **operating confidence**: val micro-F1, 0.05–0.95를 0.05 간격으로, 동률이면 높은 값.
3. **holdout 순서**: 두 팔 val 40개 완료 → 선택 해시 동결 → 선택 팔마다 일반/Pi 1회씩.
4. **1차 지표**: AP50–95 max300(주), max100(호환), 동결 confidence에서의 P/R/F1.
5. **파일**: evaluator, orchestrator, 학습 프로토콜, 후보 체크포인트, stress 프로토콜은 수정하지 않는다.
6. **재선택 금지**: 이 부록의 결과로 checkpoint·pipeline·confidence·`recommended_arm`을 다시 고르거나 holdout을 다시 실행하지 않는다.
7. **1차 결과 보존**: 보조 분석이 1차 결과와 다른 방향을 보여도 1차 결과를 고치지 않고 차이만 기록한다.

### 1.3 보고 방식

- 보조 분석 결과는 1차 결과와 다른 파일·표에 두고 "보조(해석용)"라고 표시한다.
- 확정본 `ANALYSIS_ADDENDUM_v1.1.md`의 SHA-256을 `ANALYSIS_ADDENDUM_v1.1.sha256`에 기록한다.
- 사전 감사(§6.2)에서 동결 시각이 holdout job들의 `started_utc`보다 앞선다는 것을 확인한다.
- 이후 수정이 필요하면 v1.2를 새로 만들고, v1.1을 덮어쓰지 않는다.

### 1.4 결과 문장 규칙

| 쓸 수 있는 문장 | 쓰지 않는 문장 |
|---|---|
| 두 팔의 val·개발 holdout 수치를 같은 표에 나란히 제시 | improved가 우수하다 또는 열등하다 (seed 1개, 유의성 미검정) |
| "val 기준 추천 팔(v1 규칙, 단일 seed)" | "최종 mAP", "처음 보는 시험 성능" |
| 그룹·클래스·크기별 분자/분모 | ignore·문맥 crop·중심 crop 각각의 효과 |
| 해상도 시험을 "ROI 확대를 가정한 합성 해상도 민감도"로 기술 | D455 실사용 성능, 그 상한·하한 |
| | 8px 미만 부품 성능(val에 0개), target4 mask 성능(학습 안 함) |

## 2. 수량 용어와 기록 보완

### 2.1 원본 객체·저장 라벨·실제 노출의 구분 (train split)

| 구분 | 정의 | baseline | improved |
|---|---|---:|---:|
| 원본 목표 객체 | train 원본 사진의 target4 bbox | 3,307 | 3,307 |
| 저장 crop 양성 라벨 | 학습 view 라벨 파일의 행 수. crop마다 같은 부품이 반복됨 | 6,575 | 24,807 |
| ↳ 그중 <50% 조각 양성 | 원본 bbox 면적 대비 보이는 비율 < 0.5 | 837 | 0 |
| 저장 ignore 영역 | sidecar의 영역 수 | 0 | 4,121 (조각 2,722 · unknown 1,399) |
| 저장 view (추출된 view) | `views.json`의 view 수 (그중 1회 이상 추출된 수) | 418 (418) | 968 (966) |
| 실제 추출 | 복원 추출 draw = 2,218 batch × 4 | 8,872 | 8,872 |
| 실제 양성 라벨 노출 | Σ(view 추출 수 × 그 view의 라벨 수) | 268,170 | 324,762 |
| 실제 ignore 영역 노출 | Σ(view 추출 수 × 그 view의 ignore 영역 수) | 0 | 46,469 (조각 21,180 · unknown 25,289) |
| 한 번 이상 본 고유 원본 객체 | 추출된 view에 양성 라벨로 한 번 이상 나온 고유 instance | 3,307 | 3,307 |
| 완전한 형태로 한 번 이상 본 고유 객체 | 그중 보이는 비율 ≥ 1−1e-12인 경우 (`prepare_r04.py:129` 기준) | 3,215 | 3,307 |

| 클래스 | 원본 | 저장 라벨 B / I | 실제 노출 B / I | 노출 비 I/B |
|---|---:|---:|---:|---:|
| resistor | 1,201 | 2,186 / 9,523 | 80,031 / 107,356 | 1.34 |
| capacitor | 1,584 | 3,054 / 11,092 | 131,249 / 156,437 | 1.19 |
| ic | 204 | 507 / 1,721 | 31,951 / 34,352 | 1.08 |
| connector | 318 | 828 / 2,471 | 24,939 / 26,617 | 1.07 |
| 합계 | 3,307 | 6,575 / 24,807 | 268,170 / 324,762 | 1.21 |

- B는 baseline, I는 improved다.
- 반복 노출은 새 정답이나 고유 부품 수가 아니다. gradient 기여 비율도 아니다.
- 실제 노출과 고유 객체 수는 view 기록과 추출 수로 다시 계산한 값이다. 기록된 `class_exposures`, `unique_positive_source_instances_seen(_complete)`와 일치한다(부록 R06).

### 2.2 sampling weight와 loss weight

| 구분 | 값 | 의미 | 아닌 것 |
|---|---|---|---|
| sampling weight | 17개 보드 그룹마다 합 1. 그룹 안에서 IC 또는 connector 목표를 가진 view는 1.5, 나머지는 1.0을 주고 그룹 합으로 나눔. 에폭당 448회 복원 추출 | 그룹 선택 확률을 같게 함. 실제 그룹별 추출 수는 두 팔이 완전히 같다(그룹당 490–560회) | 클래스·고유 객체·gradient 균형 |
| 클래스 loss weight | 실효 1.0 | Ultralytics 8.4.120은 `cls_pw=0`이면 `model.class_weights`를 만들지 않는다(`start.json`의 `class_weights: null`). 모든 클래스 BCE가 같은 가중 | `training_summary.json`의 `[1,1,1,1]`은 측정값이 아니라 기록 문자열이다(정정 E2) |
| loss gain | box 7.5 · cls 0.5 · dfl 1.5 | 두 팔 공통 | |
| ignore | improved만 해당 | ignore 영역 안의 할당되지 않은 anchor에서 음성 classification BCE를 0으로 함. 양성·할당 anchor와 box/DFL은 그대로 | 픽셀 덮기, GT 삭제 |

### 2.3 노출 집중 — UTL-103

| 항목 | baseline | improved |
|---|---:|---:|
| UTL-103 그룹의 view 수 | 1 | 1 |
| 그 view의 추출 수 / 라벨 수 | 510 / 127 | 510 / 127 |
| 그 view의 양성 노출 | 64,770 | 64,770 |
| 전체 양성 노출 중 비중 | **24.2%** | **19.9%** |
| 상위 3그룹 비중 | 53.3% (UTL-103 · DigitalDiscovery · MicroZed) | 48.7% |
| view별 추출 수 (최대 / 중앙값 / 최소) | 510 / 14 / 2 | 510 / 6 / 1 |
| 추출 기준 실효 view 수 (역심슨 지수) | 105 / 418 | 139 / 968 |

그룹 균형 추출은 그룹 선택 확률만 맞춘다. 그래서 view가 한 장뿐인 조밀한 그룹의 라벨이 노출의 큰 몫을 차지한다. 두 팔이 같은 그룹 추출 순서를 공유하므로 팔 비교를 한쪽으로 기울이는 요인은 아니다. 다만 절대 성능과 일반화 해석의 한계로 기록한다. R04에서는 수정하지 않는다(후속 후보 §8).

### 2.4 view 종류별 추출과 양성 노출

| view 종류 | baseline 추출 / 양성 노출 | improved 추출 / 양성 노출 |
|---|---:|---:|
| native1024 (원본 배율) | 8,872 / 268,170 | 4,249 / 145,594 |
| context2048 (0.5–1배로 축소) | 0 / 0 | 1,130 / 57,566 |
| object_centered | 0 / 0 | 3,493 / 121,602 |
| 합계 | 8,872 / 268,170 | 8,872 / 324,762 |

- 같은 1,165회 갱신에서 improved는 양성 라벨 노출이 **21% 많고**, native 배율 view 추출은 **52% 적다**.
- 따라서 두 팔 비교는 다음 요소가 합쳐진 **정책 묶음** 비교다.
  - view 구성
  - <50% 조각 처리: baseline은 양성, improved는 ignore
  - unknown 처리: baseline은 배경, improved는 ignore
  - 위의 노출량 차이
- 계산량은 batch 4, 입력 1024, 1,165회로 두 팔이 같다. 라벨 노출량과 배율 구성까지 같게 맞춘 것은 아니다.

### 2.5 얇은 ignore 영역 — anchor 중심이 없음

| 구분 | 영역 수 | anchor 중심이 없는 영역 | 노출 기준 |
|---|---:|---:|---:|
| 조각(<50%) ignore | 2,722 | 309 (11.4%) | 2,608 / 21,180 (12.3%) |
| unknown ignore | 1,399 | 11 (0.8%) | 98 / 25,289 (0.4%) |
| (참고) baseline 조각 양성 | 837 | 110 | — |

- **정의**: 학습 LetterBox(1024, 가운데 정렬, 확대 허용; 라벨은 비율 r로 늘리고 반올림한 left/top만큼 이동) 뒤, stride 8/16/32 anchor 중심 ((i+0.5)×stride)이 박스 안(경계 포함)에 하나도 없는 영역이다. 이런 영역에서 ignore는 어떤 anchor의 손실도 바꾸지 않는다. 독립 검사에서도 2px ignore 박스는 억제한 anchor가 0개였다(부록 R05).
- **해석 제한**: 이 비율은 기하 집계일 뿐이다. 성능 손실 비율, ignore 효과의 크기, AP 변화로 해석하지 않는다. TAL도 중심이 GT 안에 있는 anchor만 후보로 쓰므로, 그 주변 anchor는 원래 해당 조각을 담당하지 않을 가능성이 크다. 실제 영향은 측정하지 않았다.

## 3. FP 분류 규칙 (보조 진단)

### 3.1 목적과 금지 사항

- **목적**: v1이 FP로 센 검출이 무엇과 겹치는지 서술적으로 분류해, 팔 간 precision 차이를 해석하는 데 쓴다.
- **대상 FP**: v1 `operate()`와 같은 규칙(점수 내림차순, 같은 클래스, IoU ≥ 0.5, 1:1 greedy)으로 **v1이 이미 FP로 센 검출만**이다.
  - val 후보는 그 후보 자신의 val 선택 confidence를 쓴다.
  - holdout은 동결 confidence를 쓴다.
- **바꾸지 않는 것**
  - TP·FN과 시험 GT는 바꾸지 않는다.
  - 제외 객체와 겹친다는 이유로 정상 TP를 제거하지 않는다.
  - GT를 추가·삭제·수정하지 않는다.
  - 1차 지표(AP, P/R/F1)를 다시 계산하지 않는다.
- **판단하지 않는 것**: "source unknown과 겹친 FP"는 source가 unknown으로 남긴 영역 위의 검출이라는 사실만 뜻한다. 그 검출이 실제 target4 부품인지, 라벨 오류인지는 판단하지 않는다.
- **측정하지 않은 것**: unknown·비대상 FP를 뺀 가상 precision·AP 같은 보정 지표는 이 부록에서 계산·보고하지 않는다. **따라서 이런 FP가 precision/AP에 미친 영향은 아직 측정되지 않았다.** 보정 지표가 필요하면 사람이 검수한 새 정답 버전과 새 평가 계약으로 따로 정의한다.

### 3.2 입력

- **예측·GT**: 각 평가 폴더의 `samples.json`. native 좌표 xyxy, class, score, truth가 들어 있다.
- **제외 객체**: `work/r04/data/native_manifest.json`의 `excluded_source_objects`. 아래 사항은 부록 R08에서 확인했다.
  - `original_records.json`과 동일하다.
  - 원본 VOC XML 47장의 객체 18,201개 = 목표 5,896개 + 제외 12,305개다. 빠진 객체가 없다.
  - source_type은 28종이다. `bbox_xyxy`는 목표 GT와 같은 변환(xmin−1, ymin−1, xmax, ymax)의 native 픽셀이며, 좌표 범위 이탈이나 면적 0인 상자는 없다.
  - 저장된 val truth는 원본 목표 상자와 같고, 22개 job이 같은 `native_ground_truth.json`을 쓴다. 따라서 제외 상자·GT·예측이 같은 좌표계다.
  - cohort별 제외 객체 수(그중 unknown): train 6,753(145) · val 2,050(25) · 일반 2,995(44) · Pi 507(67).

### 3.3 기하 정의 (native 픽셀, 연속 좌표, xyxy)

- area(b) = max(0, x2−x1) × max(0, y2−y1)
- IoU(p, q) = area(p∩q) / area(p∪q)
- IoA_pred(p, e) = area(p∩e) / area(p) — 예측 상자 중 제외 박스 안에 든 비율
- IoA_excl(p, e) = area(p∩e) / area(e) — 기록용, 분류에는 쓰지 않음

### 3.4 기준값

| 용도 | 정의 | 기준 |
|---|---|---|
| 같은 클래스 목표 관련 | 같은 클래스 target GT와의 최대 IoU | ≥ 0.10 |
| 다른 클래스 목표 관련 | 다른 클래스 target GT와의 최대 IoU | ≥ 0.50 |
| 제외 객체 겹침 — **주 기준(엄격)** | 제외 객체와의 최대 IoU | ≥ 0.50 |
| 제외 객체 겹침 — **민감도(느슨)** | 제외 객체와의 최대 IoA_pred | ≥ 0.50 |

엄격 기준과 느슨 기준은 별도 표로 보고하고 합산하지 않는다.

### 3.5 분류와 우선순위

FP 하나는 한 범주에만 들어간다. 위에서부터 처음 만족하는 범주에 넣는다.

| 순서 | 범주 | 조건 | 세부 기록 |
|---:|---|---|---|
| 1 | 기타 FP — 같은 클래스 목표 관련 | 같은 클래스 GT와 IoU ≥ 0.10 | IoU ≥ 0.5이고 그 GT가 이미 더 높은 점수의 예측과 매칭됐으면 **중복**, 그 밖은 **위치 오차** |
| 2 | 기타 FP — 클래스 혼동 | 다른 클래스 GT와 IoU ≥ 0.50 | (예측 클래스 → GT 클래스) 쌍 |
| 3 | **source unknown** | IoU가 가장 큰 제외 객체의 source_type이 unknown이고 그 IoU ≥ 0.50 | |
| 4 | **확인된 비대상** | IoU가 가장 큰 제외 객체가 unknown이 아니고 그 IoU ≥ 0.50 | source_type별 개수 |
| 5 | 기타 FP — 배경 | 위 조건 없음 | |

- **동점 규칙**
  - 제외 객체끼리 IoU가 같으면 unknown을 우선하고, 그다음 `source_object_index`가 작은 쪽으로 정한다.
  - GT끼리 동점이면 GT 인덱스가 작은 쪽으로 정한다.
- **우선순위 근거**: 시험 정답(target GT)과의 관계를 먼저 판정한다. 목표 부품의 위치·클래스 오류가 unknown·비대상 겹침으로 잘못 설명되지 않게 하는 보수적 순서다.
- **느슨 기준표**
  - 3·4번에서 IoU 대신 IoA_pred ≥ 0.50을 쓴다.
  - 제외 객체는 IoA_pred 최대 → IoU 최대 → unknown 우선 → index 순으로 고른다.
  - 1·2·5번은 엄격 기준과 같다.
- **중첩 공개**: 범주와 별개로 FP마다 모든 조건의 충족 여부를 기록한다. 우선순위 때문에 다른 범주로 간 FP 수를 교차표로 공개한다.

### 3.6 확인된 비대상의 보고 방식

**주 보고**는 source_type별 개수(원문 이름 그대로)다.

**참고 소계**는 아래 고정 목록으로 묶는다. 결과를 보기 전에 선언하며, 새 라벨이나 새 클래스가 아니다.

| 소계 | source_type |
|---|---|
| 외형 유사 수동소자 | resistor network, resistor jumper, capacitor jumper, ferrite bead, inductor, fuse, emi filter, potentiometer |
| 핀·패드 | pins, pads (pins는 향후 D455 계약에서 핀헤더를 connector로 둘 수 있어 따로 표시) |
| 표기 | text, component text |
| 나머지 | 위 목록에 없고 unknown도 아닌 source_type (현재 15종, 부록 R08) |

목록의 이름은 모두 데이터의 source_type과 글자 그대로 일치한다(부록 R08).

### 3.7 산출물

- **대상**
  - val: 두 팔의 v1 선택 후보와 고정 비교 후보(§4.4의 epoch 20 native/dual, 두 팔)
  - holdout: 두 팔 × 일반/Pi
- **표**
  - 범주 × 예측 클래스의 개수와 전체 FP 대비 비율(엄격·느슨 각각)
  - source_type별 개수
  - 조건 교차표
  - cohort별 제외 객체 수(분모 정보)
- **저장**: 새 폴더 `work/r04_review/analysis_addendum_v1_1/outputs/fp_taxonomy_*.json`. 1차 metrics와 분리한다.

## 4. 선택 안정성 분석 (보조 진단)

### 4.1 원칙

- v1 선택 결과(동결 파일)를 기준으로 삼는다. 이 절의 결과로 최종 모델이나 confidence를 다시 고르지 않는다.
- 모든 계산은 저장된 val·holdout 출력을 CPU에서 재집계한다. GPU 재추론은 하지 않는다.
- AP는 v1과 같은 정의(COCO bbox, IoU .50:.95, maxDets 300 주·100 호환, area all)로 이미지 부분집합에 대해 다시 계산한다.
  - 먼저 같은 코드로 전체 집합을 계산해 v1 값과 일치하는지 확인한다.
  - 검토에서 만든 독립 구현은 22개 job 모두 차이 ≤ 2.2e-16이었다(부록 R04).

### 4.2 두 종류의 집계

| 이름 | 정의 | 용도 |
|---|---|---|
| 전체 객체 합산 (pooled, v1 1차) | 모든 이미지의 예측·GT를 합쳐 한 번에 계산 | v1 1차 지표 그대로 |
| 그룹 평균 (macro) | 그룹별로 계산한 값의 단순 평균. 그룹 가중 없음 | 한 그룹 편중 확인용 보조 |

- 그룹별 표에는 이미지 수와 클래스별 객체 수를 함께 적는다.
  - val 그룹 객체 수: Arty 479(45.5%) · Virtex6 208 · Zybo 143 · ACM-109 104 · XCM-307A 71 · UTL-016 47
  - 일반 holdout: 5그룹, ML450이 61.3%
  - Pi: 1그룹
- 해당 클래스가 없는 그룹은 COCO 규칙대로 그 클래스 AP를 계산하지 않는다. 객체가 적은 그룹·클래스의 값은 불안정하다고 표기한다.

### 4.3 한 그룹 제외(leave-one-group-out) 순위 안정성 — val 전용

- **계산**: val 그룹 6개 각각에 대해, 그 그룹의 이미지를 뺀 나머지로 팔마다 20개 후보의 AP300·AP100을 다시 계산한다.
- **순위 키**: v1과 같다. offline p50은 원래 val job 값을 그대로 쓰고 재측정하지 않는다.
- **보고 (팔별)**
  - 그룹 제외 시 1위 후보와 v1 승자의 일치 여부
  - v1 승자의 순위
  - 전체 순위와 제외 후 순위의 Kendall τ
  - 전체 val의 1·2위 AP300 차이
  - `recommended_arm`이 그룹 제외 시 바뀌는지
- **해석**: 승자가 바뀌면 "선택이 val 그룹 구성에 민감함"으로만 기록한다. v1 선택·confidence는 그대로 둔다.

### 4.4 같은 epoch·같은 pipeline의 두 팔 비교

- **쌍**: epoch 2, 4, …, 20 × native/dual = 20쌍. 두 팔의 해당 후보는 직접 optimizer 호출 수가 같다(168, 280, …, 1,064, 1,165).
- **epoch 20은 부분 에폭**이다. 20번째 에폭의 90/112 배치까지만 실행됐고, 두 팔 모두 1,165회 갱신에서 멈췄다. 표에는 "e20(부분, 90/112 배치)"로 적는다.
- **보고**
  - pooled AP300·AP100 차이(improved − baseline)
  - 그룹 평균 AP300 차이
  - 그룹별 차이의 부호 수(6개 중 몇 그룹에서 improved가 높은지)
  - 클래스별 AP300 차이
- **operating 지표**: 후보마다 confidence가 달라 P/R/F1은 직접 비교하지 않는다. 쌍 비교는 threshold가 없는 AP로 한다.
- **한계**: seed 1개, val 9장·6그룹이다. 유의성이나 일반화는 주장하지 않는다.

### 4.5 holdout의 그룹·클래스 표 (holdout 후 계산, 선택에 사용하지 않음)

- **일반 holdout (5그룹)**
  - 그룹별 AP300·AP100, 동결 confidence의 recall·precision, 그룹 평균과 pooled(v1)를 함께 보고한다.
  - v1의 bootstrap recall 구간은 그대로 옮기되, ML450이 객체의 61.3%라 구간이 불안정할 수 있다고 적는다.
- **Pi (1그룹)**
  - 그룹 간 구간은 없다.
  - 클래스별 TP/FP/FN을 분자/분모로 적는다(분모: IC 16, connector 12, resistor 57, capacitor 104). 비율만 단독으로 쓰지 않는다.
- **크기 구간**: v1 metrics의 클래스 × 짧은 변 recall을 그대로 옮긴다. val에는 8px 미만 객체가 0개(8–16px 98개)라, 이 구간에 대해서는 선택 근거가 없다고 명시한다.

## 5. 평가 상한과 합성 해상도 시험의 표기

### 5.1 AP300의 정확한 조건

v1의 "AP50–95 max300"은 다음 순서로 만들어진 값이다.

1. **crop마다**: `model.predict(conf=0.001, iou=0.6, max_det=1000)` — crop당 최대 1,000개 (`evaluate_r04.py:414`).
2. **이미지마다**: 모든 crop·배율의 예측을 원본 좌표로 옮긴 뒤 클래스별 NMS(IoU 0.5)를 하고, 점수 상위 **이미지 전체 최대 1,000개**(클래스 합산)를 남긴다 (`:99–108`).
3. **COCO**: 이미지·클래스마다 최대 300개(주) / 100개(호환) (`:181`).

→ **모든 표기는 "AP50–95 (이미지 전체 최대 1,000개 제한 후 COCO 클래스별 최대 300개)"로 한다.** AP100도 같은 1,000개 제한 이후의 값이다.

### 5.2 완료된 val 22개에서 관측한 사실 (부록 R04·R07)

- 198회의 이미지 평가 중 **87회**에서 이미지 전체 1,000개 제한이 작동했다.
  - epoch 20 dual: 9장 중 3장, epoch 20 native: 2장
  - 제한에 걸린 이미지에서 남은 최저 점수: 0.0011–0.0107 (epoch 18–20)
- 이미지·클래스 칸 36개 중 저장 예측이 100개를 넘는 칸은 14–30개, 300개를 넘는 칸은 4–10개다.
- val에는 GT가 100개를 넘는 이미지·클래스 칸이 2개 있다(최대 217개). 그 칸에서 AP100은 구조적으로 재현율 상한이 있다.
- **상한 때문에 AP가 실제로 얼마나 달라지는지는 현재 자료로 확인하지 못했다.** 제한 전에 제거된 예측은 저장되지 않았다.

### 5.3 이번 결정

- 평가 v2(상한 변경)는 추가하지 않는다. v1 결과를 그대로 쓰고, 모든 표에 §5.1의 표기를 붙인다.
- 상한 민감도가 필요해지면 후속 후보(별도 버전·별도 폴더, v1과 혼합 금지)로 둔다.

### 5.4 합성 해상도 시험: "ROI 확대를 가정한 합성 해상도 민감도"

- **절차**
  - 이미 본 Pi holdout 2장(1그룹)을 명목 3.2·2.1 px/mm로 LANCZOS 축소한다.
  - 동결된 평가 pipeline이 입력을 1024로 letterbox **확대**한다.
  - 동결된 checkpoint·pipeline·confidence를 그대로 쓴다.
  - 2조건 × 2팔 = 4작업이다.
- **근사**: 원본 배율 22.57(Top) / 14.23(Bottom) px/mm는 헤더 핀 간격으로 추정한 값이다.
- **재현하지 않는 것**
  - 실제 D455 1280×800 전체 프레임에서 보드가 일부만 차지하고 확대가 일어나지 않는 조건
  - D455 광학(초점·MTF·노이즈·노출·반사)
  - 보드 검출·ROI 추출 단계
- **해석하지 않는 것**: 실제 D455 성능, 그 상한·하한, 카메라 적합·부적합 판정. 모델·pipeline·confidence 선택에도 쓰지 않는다.
- stress v2(전체 프레임 배치 조건)는 이번에 추가하지 않는다.

## 6. 감사 실행 순서

### 6.1 확인 결과: 기존 `audit_r04_results.py`는 holdout 완료를 요구한다

- **holdout 완료가 필수 조건이다.**
  - `aggregate.json`의 status가 `COMPLETED_DEVELOPMENT_HOLDOUT_NOT_D455_VALIDATED`여야 통과한다 (`:395`).
  - 팔마다 일반·Pi holdout 결과를 검사하고 (`:417–422`), test job이 정확히 4개여야 한다 (`:428–431`).
- **실패해도 파일을 남긴다.** 실패 시에도 `reports/final_results_audit.json/.md`를 FAIL 상태로 쓴다 (`:501–508`).
  - holdout 전에 실행하면 동결 폴더에 오해 소지가 있는 "최종 감사 실패" 파일이 생긴다. **holdout 전에는 실행하지 않는다.**
- **멈출 지점이 없다.** orchestrator(`run_r04_evaluation.py:359–393`)는 val → 선택 동결 → holdout을 한 번의 실행으로 이어서 한다. 동결과 holdout 사이에 멈추는 옵션이 없다.

### 6.2 제안: 동결 코드를 수정하지 않는 읽기 전용 사전 감사

코드는 부록 확정 후 별도로 작성하고 검토한다.

**① `stop_before_holdout.py` (새 파일, 가칭)**
- `run_r04_evaluation` 모듈을 import해 `execute()`를 그대로 호출한다. 다만 `split='test'` 작업이 job 기록을 만들기 **전에** 정해진 예외로 멈추도록 `evaluate_job`을 감싼다.
- 결과
  - 남은 val 18개를 실행한다.
  - `selections_frozen.json`과 `aggregate.json`(status `SELECTIONS_FROZEN_DEVELOPMENT_HOLDOUT_PENDING`)을 저장한다.
  - holdout job 기록 0개인 상태로 멈춘다.
- orchestrator 파일이 바뀌지 않으므로 evidence fingerprint도 같다. 이후 원래 orchestrator를 그대로 실행하면 완료된 val과 동결 선택을 해시로 재사용하고 holdout만 진행한다.
- 멈춘 뒤 test job 파일과 test 출력 폴더가 없는지 확인하고 종료한다.

**② `pre_holdout_audit.py` (새 파일, 가칭)**
- 기존 `Audit` 클래스를 import해 아래 항목만 실행한다.
- 결과는 `work/r04_review/pre_holdout_audit/`에 저장한다. 기존 `reports/final_results_audit.*`에는 쓰지 않는다.

| 사전 감사 항목 | 내용 |
|---|---|
| `source_truth()` | 원본 GT·제외 객체·holdout cohort·그룹 누수 (기존 로직 그대로) |
| `training(protocol)` | 1,165회, 배치, EMA, 노출 재계산, 체크포인트 20개 CPU 로드 |
| val 40개 `evaluation_job(..., validate_grid=True)` | 해시 연결, GT 변환, 독립 COCO·매칭, confidence grid |
| 선택 재계산 | 팔별 v1 순위 키로 승자를 구해 `selections_frozen.json`과 일치하는지, `recommended_arm` 일치, 모든 val job 종료 시각 ≤ `frozen_utc` |
| holdout 부재 확인 | `jobs/*__general_test.json`, `*__pi_test.json`과 해당 evaluation 폴더가 없음 |
| 부록 동결 기록 | 확정본 SHA-256과 시각이 holdout 시작 전인지 기록 |
| 기록 방식 주석 | 기존 감사의 `class_weights == [1,1,1,1]` 검사는 기록 문자열 확인이다(정정 E2). 실효 가중치는 Ultralytics 소스와 `start.json`으로 따로 확인한다. |

**③ holdout 완료 후** 기존 `audit_r04_results.py`를 수정 없이 실행해 전체 감사를 한다.

**④ 보조 분석(§3·§4) 스크립트**도 새 폴더에 둔다.
- val 부분은 holdout 전에 실행하고, 결과와 스크립트 SHA를 기록한다.
- holdout 부분은 holdout 후 같은 스크립트로 실행한다. 스크립트를 수정하면 새 버전으로 기록한다.

## 7. 정정 내역 — 원래 실행 기록은 덮어쓰지 않음

| ID | 위치 | 원문 | 정정 | 근거 | 영향 |
|---|---|---|---|---|---|
| E1 | `R04_PAUSED/RESUME.md:39` | "사전 동결한 별도 합성 해상도4조건 작업" | "2조건(명목 3.2·2.1 px/mm) × 2팔 = 4작업" | `stress_protocol.json`의 conditions 2개, jobs 4 | 표기 |
| E2 | `runs/*/training_summary.json`의 `class_weights` | `[1.0, 1.0, 1.0, 1.0]` | 측정값이 아니라 `train_r04.py:261`의 고정 문자열이다. 실효 값은 같은 1.0이다(`cls_pw=0`이면 Ultralytics가 class_weights를 만들지 않음, `start.json` null) | `train_r04.py:150, 261`, Ultralytics `detect/train.py`의 `set_class_weights` | 해석. 기존 감사 `:222`의 해당 검사는 문자열 확인 |
| E3 | `README.md:19, 23–24` (R03 표) | "고정 threshold의 box Recall" 열에 n·s 값(43.39%·49.21%)을 나란히 표기 | n은 confidence 0.60, s는 0.35에서의 값이다. 운용점이 달라 직접 비교하지 않는다 | `R04_PAUSED/reports/CLAUDE_REVIEW_RESPONSE.md:25` | R03 표기 |
| E4 | `runs/*/epochs.json`의 `full_image_loader_validation` | "Diagnostic only" | 전체 이미지를 1024로 축소한 진단 지표다. improved는 2048→1024 축소 view로 학습해 이 입력 조건에 유리하므로 팔 비교에 쓰지 않는다 | `prepare_r04.py:111, 138–139` | 해석 |
| E5 | `snapshot.json`의 `evaluation_protocol_sha256` | `a45e4d37…` | 파일 SHA가 아니라 평가 evidence의 fingerprint다. 파일 `reports/evaluation_suite/protocol.json`의 SHA는 `e52f35a7…`다 | `run_r04_evaluation.py`의 `fingerprint()` | 명칭 |

정정은 이 부록과 이후 결과 문서에만 적는다. 위 원본 파일은 수정하지 않는다.

## 8. 후속 후보 — 이번 실행에는 포함하지 않음

R04 평가와 감사를 마친 뒤 결과와 비용을 보고 결정한다.

| 후보 | 목적 | 대략 비용 (RTX 5060 8GB, 모델당 ≤20에폭) | 먼저 검토할 조건 |
|---|---|---|---|
| seed 반복 (두 팔 × seed 2개 추가) | 팔 차이가 seed 잡음보다 큰지 확인 | 학습 약 15분 × 4, e20 native/dual val 16작업(작업당 약 15초) | 팔 간 AP 차이가 작거나 한 그룹 제외 분석에서 선택이 불안정할 때 |
| 요인 분리 (baseline+ignore, baseline+문맥/중심 view) | 개별 요소의 효과 | 팔당 약 15분 + 평가. seed 반복 없이는 해석이 제한됨 | seed 반복 결과를 본 뒤 |
| sampler 변경 (view 반복 상한 또는 고유 객체 기준 가중) | UTL-103 같은 노출 집중 완화 | 추가 계산 없음. 다음 학습 버전에 사전 등록 | 다음 학습을 계획할 때 |
| 평가 상한 민감도 (evaluator v2) | 1,000개 제한의 영향 측정 | 40+4 재추론, 별도 폴더. v1과 혼합 금지 | 선택이 AP 소수점 차이로 갈렸을 때 |
| stress v2 (1280×800 전체 프레임 배치) | ROI 확대가 없는 조건 근사 | CPU 준비 + 4작업 | D455 실측 전에 참고가 필요할 때 |

## 9. D455 실측·라벨링 — 별도 단계

**현재 상태 (유지)**: D455 실물 영상 0장, target4 사람 검수 mask 0개, **target4 instance mask 학습 안 함**. 이 단계는 R04 holdout을 사용하지 않으며, R04 평가와 무관하게 D0부터 시작할 수 있다.

아래 수량의 참고값은 공개 WACV Pi 3B 두 면의 정답(target4 189개: R 57 · C 104 · IC 16 · connector 12, unknown 67)이다. 보유 보드의 실제 부품 수는 다를 수 있다.

### D0. 카메라 특성화 — 보드 1대로 충분, 목표 라벨 없음

| 측정 | 방법 | 기록 |
|---|---|---|
| 내부 파라미터 | SDK에서 사용하는 RGB profile(예: 1280×800)의 fx·fy·ppx·ppy·왜곡 | SDK/펌웨어 버전 포함 |
| 보드 평면 배율 | 보드와 같은 평면에 자·체커보드. RGB 초점 범위 안의 거리 3–4개 | 거리별 실측 px/mm(가로·세로), 보드의 화면 점유 크기 |
| 초점·해상도 | 슬랜티드 엣지 또는 해상도 차트 | 거리별 판독 가능한 최소 크기. depth Min-Z(사양 약 52 cm)와 RGB 초점은 따로 기록 |
| 부품 픽셀 크기 | 실제 보드의 대표 R/C·IC·커넥터 | 거리별 짧은 변 px. 8/16 px은 진단 구간이며 합격선이 아님 |
| 노출·조명 | 노출·gain·WB 자동/고정, 확산광, 필요 시 편광 | 실제 값과 반사 위치 |
| 처리 지연 | 동결된 R04 pipeline을 실제 실행 장치에서 | 캡처→결과 p50/p95. 정확도와 별개 |

- 예시 규모: 거리 4 × 2면 × 조명 2 = 약 16장.
- 결과로 촬영 거리와 R/C 라벨 가능 여부를 정한다. 명목 계산(fx≈640 가정)으로는 200 mm에서 3.2 px/mm, 300 mm에서 2.1 px/mm인데, 이는 실측으로 대체한다.

### D1. 파일럿 — 30장 계획 유지, 보드 1대 기준

| 항목 | 최소 계획 |
|---|---|
| 구성 | 2면 × D0에서 정한 거리 3개 × 조명·각도 5조건 = 30장. 연속 프레임은 독립 표본으로 세지 않음 |
| bbox | IC·connector는 30장 모두 빠짐없이(참고값 면당 약 14개 → 약 420개). R/C는 미리 정한 6장에서만 빠짐없이(약 480개), 판독 불가는 `unresolved` |
| mask | 같은 6장의 IC·connector만(참고값 약 84개). R/C mask는 판독성 확인 전까지 보류 |
| 이중 검수 | bbox 6장(20%), mask 2장. 한 명이면 시간차 재검토로 표기 |
| 기록 | 객체당 작성·검수 시간, 불일치, unresolved 수. 파일럿은 최종 시험으로 쓰지 않음 |

### D2. 분할 — 보유 보드 수에 맞춤

| 보유 보드 | 분할 | 결과 명칭 |
|---|---|---|
| 1대 | 촬영 세션 단위로 나눔(날짜·재배치·조명이 다른 세션). test 세션은 라벨 규칙 확정 전에 떼어 둠 | "같은 실물의 미관측 세션" 평가. 설계 일반화 주장 없음 |
| 같은 설계 2–3대 | 1대를 test로 통째로 분리(test-A). 3대면 val도 별도 실물 | "같은 설계의 새 실물" |
| 다른 설계 포함 2–3대 | 한 설계를 통째로 test로(test-B). 그 설계의 인터넷 사진도 학습에서 제외 | "새 설계(1종)"로 한정 |
| 4대 이상 | 기존 200장 예시(train 3 · val 1 · test-A 1 · test-B 1) | test-A/B 각각 보고 |

- 고유 부품 다양성은 보드 수가 상한이다. Pi 한 대의 고유 커넥터는 10개 안팎이라, 고유 IC·커넥터 각 100개 목표는 사진 수로 채울 수 없다. 실제 고유 수를 그대로 보고한다.
- 사진을 늘리기보다, 적은 장수를 빠짐없이 라벨하고 독립 실물·세션을 늘리는 것을 우선한다.
- instance mask 학습은 D1에서 IC·connector mask 작성 가능성을 확인한 뒤 별도 manifest·분할·평가 계약으로 시작한다.

## 10. 정리

### 10.1 남은 평가 전에 반드시 확정할 항목

1. 이 부록의 확정본과 SHA-256, 동결 시각. holdout 시작 전이어야 한다.
2. FP 분류의 기준값·우선순위·동점 규칙·참고 소계 목록(§3).
3. 선택 안정성 분석의 정의: 그룹별 지표, 그룹 평균과 pooled의 구분, 한 그룹 제외, 같은 epoch·pipeline 쌍 비교, e20 부분 에폭 표기(§4).
4. AP300 표기 문구, 합성 해상도 시험 명칭, 평가 v2·stress v2를 추가하지 않는다는 결정(§5).
5. 감사 순서, 그리고 `stop_before_holdout.py`·`pre_holdout_audit.py`의 코드 검토. 동결 파일 무수정을 해시로 확인한다(§6).
6. 정정 내역 E1–E5(§7).
7. 보조 분석 스크립트를 holdout 전에 작성·검토하고, val 부분을 실행해 스크립트 SHA를 고정한다.

### 10.2 기존 결과를 그대로 재사용할 수 있는 범위

| 구분 | 재사용 여부 |
|---|---|
| 학습 2팔 기록, 체크포인트 20개 (SHA 일치) | 그대로 재사용 |
| 완료된 val 22개 (해시·CSV 일치, 독립 재계산 일치) | 그대로 재사용 |
| 데이터·split·manifest, 프로토콜 v1, evaluator·orchestrator·감사 v1, stress 프로토콜 v1 | 그대로 재사용 |
| 남은 val 18개, 선택 동결, holdout 4개, 해상도 시험 4개, 전체 감사 | v1 규칙 그대로 새로 계산 |
| FP 분류, 그룹별·한 그룹 제외·쌍 비교 | 저장 예측으로 CPU 계산만. GPU 재추론 없음 |
| 1차 지표·선택의 재계산이나 변경 | 하지 않음 (보조 결과로 바꾸지 않음) |

### 10.3 초안 확정 후 실행 순서

1. 부록 확정과 SHA-256 기록. 정정 내역 포함.
2. 새 스크립트 작성·검토 (CPU, 새 폴더): `stop_before_holdout.py`, `pre_holdout_audit.py`, 보조 분석(FP 분류, 선택 안정성). 동결 파일 SHA 불변을 확인한다.
3. `stop_before_holdout.py`로 남은 val 18개를 실행하고 선택 동결에서 멈춘다 (GPU 약 5분, holdout 미실행).
4. 사전 감사와 보조 분석 val 부분 실행 (CPU). 결과 보관, 선택 변경 없음.
5. 기존 orchestrator를 그대로 실행한다. val과 선택 재사용을 확인한 뒤 holdout 4개를 실행한다 (GPU).
6. 기존 `audit_r04_results.py` 전체 감사 (CPU), 보조 분석 holdout 부분 (CPU).
7. `evaluate_resolution_stress.py --run` 4작업 (GPU). "ROI 확대를 가정한 합성 해상도 민감도"로 기록한다.
8. 오류 분석, 그림, 패키지, GitHub 기록 (기존 계획대로, 원본 사진 제외).
9. 후속 후보(seed·요인 분리·sampler·v2)를 결과와 비용을 보고 결정한다.
10. D455: D0 → D1 → D2 (별도 단계. D0는 1–9와 병행 가능).
