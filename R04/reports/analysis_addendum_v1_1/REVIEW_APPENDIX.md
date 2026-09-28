# 검토 부록 — R04 독립 검토 재현 자료

작성: 2026-09-29, Claude. [분석 부록 v1.1 초안](ANALYSIS_ADDENDUM_v1.1_DRAFT.md)의 수치를 재현하는 자료다.

- 스크립트는 입력을 **읽기만** 하고 `repro/outputs/*.json` 한 개씩을 쓴다.
- 학습, 추론, 카메라 사용, 입력 파일 수정은 하지 않는다. GPU도 쓰지 않는다(`CUDA_VISIBLE_DEVICES=''`).
- 검토 당시 scratchpad에서 만든 두 스크립트(독립 AP, ignore 기하 검사)와 대화 중 즉석에서 실행한 집계들을 이 폴더의 독립 스크립트로 옮겨 정리했다.

## 1. 실행 환경

| 항목 | 값 |
|---|---|
| Python | `C:\Users\hkjun\Documents\mcu-vision\.venv-yolo11\Scripts\python.exe` (학습·평가와 같은 환경. numpy·PyYAML·torch·Ultralytics 사용) |
| Ultralytics | 8.4.120. R02·R05·R06의 규칙 근거 파일: `utils/loss.py` `dbda3046…d349`, `engine/trainer.py` `21591c43…128c`, `data/augment.py` `2394a58f…85b9` |
| GitHub 기록 | `work/github_private_archive` (clone). HEAD = `17f72fdd9143bca6645f9b342414b0c5a2c1ed6d`, 변경 없음(R01에서 매번 확인) |
| 로컬 실행 폴더 | `work/r04` (GitHub에 없는 가중치·manifest·저장 예측) |
| 경로 기준 | `repro/common.py`가 스크립트 위치에서 `work` 폴더를 찾는다(`parents[2]`). 다른 PC에서는 `common.py`의 `VENV_SITE`와 위치만 맞추면 된다. |

## 2. 재현 명령

PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\hkjun\Documents\Codex\2026-09-27\new-chat\work\r04_review\analysis_addendum_v1_1_draft\repro\run_all.ps1"
```

개별 실행(예시):

```powershell
cd "C:\Users\hkjun\Documents\Codex\2026-09-27\new-chat\work\r04_review\analysis_addendum_v1_1_draft\repro"
$env:CUDA_VISIBLE_DEVICES=''; & 'C:\Users\hkjun\Documents\mcu-vision\.venv-yolo11\Scripts\python.exe' -B r04_independent_ap_matching.py
```

출력 JSON의 구성과 확인 방법:
- `definitions`, `results`, `inputs`(읽은 파일 전체의 절대경로·SHA-256·크기)를 담는다.
- `generated_utc`가 들어 있어 파일 SHA는 실행마다 달라진다. 비교는 `results` 필드로 한다.
- 입력 파일의 SHA가 아래 §3과 같은지 먼저 확인한다.

## 3. 주요 입력과 SHA-256

| 입력 | SHA-256 |
|---|---|
| `R04_PAUSED/protocol.json` = `work/r04/protocol.json` | `c712d8f3e24ce8fa8291d4b2e7e2593924ccdb154ba4361f50958e63555a4def` |
| `R04_PAUSED/snapshot.json` | `7644417918d35396ff7f58bb69278997df904c9663e55d3b043552f682316c91` |
| `R04_PAUSED/partial_validation.csv` | `148adf8b9d88e52c73a91379790dd3a489ad40a3cb64472845e2b45ed8b650b8` |
| `R04_PAUSED/runs/baseline/epochs.json` | `39d12c00ad31abebb05591d43dec7f323854eee0b8cd1c8e17fc994876856032` |
| `R04_PAUSED/runs/improved/epochs.json` | `9b11cdd78a3a0b574106c1b484cd8ebf9481c85fd25af68c3c15dd2b2b4c9493` |
| `R04_PAUSED/runs/baseline/sampled_exposures.json` | `fb214f20f2d7517b6e8fae58b558e1323e3e708c449527eb219682f8873f49b6` |
| `R04_PAUSED/runs/improved/sampled_exposures.json` | `d9b2706218ff76988c7edbeabd88b96e7ea88cbd343575b19bd8f839870e6e9a` |
| `R04_PAUSED/runs/baseline/candidates.json` | `37326c24b33ce48d51b131a633da1111d99636552f2601d2a2fc8d94f9497510` |
| `R04_PAUSED/runs/improved/candidates.json` | `d9d154ea88ad3c9154bdcda7c72466e5e23725168b277869057e4571b51f90d4` |
| `R04_PAUSED/scripts/ignore_adapter.py` | `a8d5bd2baa4b43f86ce0ffd9507eeec69be1a29d431a1ba12b1b38534ca95280` |
| `R04_PAUSED/scripts/evaluate_r04.py` | `b62dce3fa1e3b8a0fe0b4526809bde5bf63f914e0e53ac8398f2f29b1c35b3e5` |
| `work/r04/data/original_records.json` | `8f09eec359fcae412e539a2ecdc2aee1a10102aaa489edd792b17b46c551aa13` |
| `work/r04/data/split.json` | `0640f66e0d058af706e9a3693b7a56d4ed475d1b4058e5e6c24374413c566429` |
| `work/r04/data/native_manifest.json` | `a91d4956da6adc8b3eebcd87da6e8e428362e1c14cef9e8f9d5a4583817d8c19` |
| `work/r04/data/baseline/views.json` | `fa507ef74db349875cd2f24c9e793dc6276a8d3b63427f0a0b8972ee2375df23` |
| `work/r04/data/improved/views.json` | `df7b5bdb67323af200a5809253120e048027b08851f47be66ba9c863e0237986` |
| `work/r04/data/baseline/ignore_regions.json` | `449046938ca0d375c8b93328838199f79452a8b08d2f84fef871fcf07ebdf1ea` |
| `work/r04/data/improved/ignore_regions.json` | `5a2fb1e4799f9c9595a3d6674f0e26e3fd7050e3061ad364351a739d4223878c` |
| `work/r04/reports/evaluation_suite/evaluations/*/…` | 22개 job의 저장 파일. 각 SHA는 R01·R04·R07·R08 출력의 `inputs`에 있음 |
| `…/evaluations/*/native_ground_truth.json` | 22개 job 모두 `8657ed2394bbbe666f230cf8cc87b0fb95f1b57b300eddaa1b998626c9999580` (R08) |
| 원본 VOC XML 47개 (`work/r03/component_assets/wacv_original/pcb_wacv_2019/*/*.xml`) | 각 SHA가 `original_records.json`의 `annotation_sha256`과 일치 (R08) |
| `work/r04/runs/*/candidates/epoch_*.pt` | 20개. `candidates.json`의 SHA와 대조(R01) |

## 4. 스크립트와 계산 정의

| 스크립트 | SHA-256 | 확인 내용과 정의 |
|---|---|---|
| `common.py` | `8a527ce025121ff9ae963e7a02e56e7952a2eeee28cf8b4651918c4f612a4157` | 경로·SHA·출력 공통 함수 |
| `r01_hash_bindings.py` | `afc6814dce3525b6547477c604d0924c257f69e5e452dafa3d1f3e4edc40c196` | 커밋·변경 없음 확인. GitHub와 로컬 공통 파일 23개의 바이트 동일성. 프로토콜에 기록된 입력 해시. val 22개의 metrics가 snapshot SHA·CSV AP·candidates SHA와 맞는지, evaluator·manifest가 한 종류인지, P/R/F1 산술. 체크포인트 20개 SHA. 두 팔 `args.yaml` 차이 |
| `r02_optimizer_schedule.py` | `7e5b0031d53bfabb47380ac5935c48c18415698a8e4530b1ace410f05ac559d0` | Ultralytics 8.4.120 규칙 `nw=round(min(2,19)×112)=224`; `ni<nw`이면 accumulate=interp(1→2); `ni−last≥accumulate`일 때 갱신; 1,165회에서 해당 batch 뒤 정지. 에폭별 기록과 비교. 그룹별 추출 수의 팔 간 동일성 |
| `r03_split_reproduction.py` | `b554ac9991253bc3aef28eaa4de2fc2c73201e2eda45cdef234861abc34459fc` | `prepare_r04.freeze_split` 재구현. R03 train+val 23그룹에서 6그룹 선택, 제약·가중 목적함수. split별 크기 구간, 최대 그룹 비중, GT 밀도, 제외 객체 유형 |
| `r04_independent_ap_matching.py` | `d1e65f94458b14a2cd72503d7420e741d9e71ff73d304840e454817edf6b83c7` | pycocotools·평가기·감사 코드를 import하지 않는 NumPy COCO bbox AP. IoU .50:.05:.95, 101 recall, 이미지·클래스별 maxDets 절단 후 누적, precision envelope, GT 있는 클래스 평균. 운용점 greedy 매칭(같은 클래스, IoU≥0.5, 1:1, 동점은 큰 GT 인덱스) |
| `r05_ignore_geometry_check.py` | `3d3f9732fc781f20bbb169668f83d11c6da5a4697e867bfc841b67a24c8f675e` | 64×128(비정사각) 입력, x/y 비대칭 ignore 박스 4경우. 억제 anchor 집합과 NumPy 기준(중심이 ignore 안, 양성 박스 밖, TAL foreground 아님)의 일치. 억제 anchor의 class gradient 0. ignore가 없을 때 stock loss와 동일. 2px ignore 박스 |
| `r06_exposure_and_anchor_coverage.py` | `bcbfc6ec69cef750b92c2b61d967690cac6dca8749b803e4a47d66069f12b977` | 원본 객체·저장 라벨·실제 노출(Σ 추출 수×라벨 수)·view 종류·그룹 비중·실효 view 수. 추출된 view에 양성으로 나온 고유 원본 객체와 그중 완전 포함(보이는 비율 ≥ 1−1e-12, `prepare_r04.py:129`)을 기록값과 대조. sampling weight 그룹 합. 학습 LetterBox 후 anchor 중심이 없는 ignore 영역·조각 양성. **기하 집계이며 성능 손실률이 아님** |
| `r07_eval_caps.py` | `a3fb54191d33487d00d1b349d5603be1aa623c221c5540928acbf4d004d3280d` | 이미지 전체 1,000개 제한에 걸린 이미지 수, crop 포화, 남은 최저 점수, 300개 미만이 남은 클래스 칸. **상한으로 인한 AP 변화는 계산할 수 없음(제거 전 예측 미저장)** |
| `r08_excluded_objects.py` | `ac961aaa4b49150117c281d99614ce8b58cff4e5e64a893f15dc5a21510c3308` | FP 분류(분석 부록 §3)에 쓸 제외 객체 입력 검증. manifest와 원본 기록의 동일성. 원본 XML 객체 수 = 목표 + 제외, `bbox_voc_raw`와 XML 상자 일치(`source_object_index`는 0부터). `bbox_xyxy` 변환 규칙, 상자 범위·면적. 저장 val truth와 원본 목표 상자의 일치. 참고 소계 목록의 이름 존재 여부. **예측은 분류하지 않음** |
| `run_all.ps1` | `7e8e9408d89883c5a10bb2d79100fa9d87f81f1203667f6673e7515bebb8bb86` | 8개 순차 실행 |

## 5. 결과 요약 (2026-09-29 실행)

| 검사 | 결과 |
|---|---|
| R01 커밋·파일 | HEAD가 검토 커밋, 변경 없음. 공통 파일 23개 모두 동일. 프로토콜 입력 해시 3개(초기 가중치·data summary·native manifest) 일치 |
| R01 val 22개 | snapshot SHA·CSV AP·candidates SHA·역할 표기·P/R/F1 모두 일치(문제 0건). evaluator `b62dce3f…` 1종, manifest `a91d4956…` 1종 |
| R01 체크포인트·인자 | 20개 SHA 일치. 두 팔 `args.yaml` 차이는 `data`·`name`·`save_dir`뿐 |
| R02 갱신 일정 | 두 팔 모두 warmup 224. 유도한 일정이 기록과 같음(에폭 1: 112회, 이후 56회, 마지막 45회 → 1,165회, 2,218 batch, 20에폭 90/112 batch). EMA = 호출 수. 그룹별 추출 수 두 팔 동일 |
| R03 분할 | 100,947개 조합 중 29,385개 적격. 선택 6그룹(ACM-109·Arty·UTL-016·Virtex6·XCM-307A·Zybo)과 목적값 일치. pool에 R03 test/Pi 없음 |
| R04 독립 AP·매칭 | 22개 job에서 AP100·AP300·AP50 최대 차이 2.2e-16. 운용점 클래스별 TP/FP/FN 모두 일치. 이미지·클래스 칸 36개 중 예측이 100개 초과인 칸 14–30개, 300개 초과인 칸 4–10개 |
| R05 ignore 기하 | 4경우 모두 NumPy 기준과 일치, 억제 anchor gradient 0, foreground 보존. ignore 없는 경우 stock과 같음. 2px 박스의 억제 anchor 0개 |
| R06 노출 | 분석 부록 §2의 표 참조. 기록된 `class_exposures`와 고유 객체 수(baseline 3,307·완전 3,215 / improved 3,307·3,307)가 재계산과 일치. 그룹별 sampling weight 합 1(17그룹) |
| R07 상한 | 198회 이미지 평가 중 87회가 1,000개 제한에 걸림. 표 §5.3 참조 |
| R08 제외 객체 | 47장, XML 객체 18,201 = 목표 5,896 + 제외 12,305. 문제 0건. source_type 28종. val truth = 원본 목표 상자. 22개 job의 GT 파일 1종. 표 §5.4 참조 |

### 5.1 split 집계 (R03 출력)

| split | 사진 | 그룹 | 객체 | <8 / 8–16 / 16–32 / ≥32 px | 최대 그룹 비중 | GT>100 칸 (최대) | unknown |
|---|---:|---:|---:|---|---|---|---:|
| train | 25 | 17 | 3,307 | 0 / 515 / 1,189 / 1,603 | ML365 29.8% | 6 (371) | 145 |
| val | 9 | 6 | 1,052 | 0 / 98 / 430 / 524 | Arty 45.5% | 2 (217) | 25 |
| 일반 holdout | 11 | 5 | 1,348 | 0 / 446 / 405 / 497 | ML450 61.3% | 4 (159) | 44 |
| Pi holdout | 2 | 1 | 189 | 2 / 77 / 72 / 38 | RPI3B 100% | 0 (52) | 67 |

holdout 행은 GT 개수 집계일 뿐이며, 예측이나 성능은 포함하지 않는다.

### 5.2 완료된 val 22개 (R01·R04·R07 출력)

AP는 "이미지 전체 최대 1,000개 제한 후 COCO 클래스별 최대 300/100개" 조건의 값이다. P/R/F1과 TP/FP/FN은 후보마다 다른 val 선택 confidence에서의 값이므로 후보끼리 직접 비교하지 않는다.

| 후보 | epoch (직접 호출) | AP300 | AP100 | conf | TP/FP/FN | 재계산 최대 차 | 1,000 제한 이미지 |
|---|---|---:|---:|---:|---|---:|---:|
| baseline e02 dual | 2 (168) | 14.90 | 12.06 | 0.60 | 573/1114/479 | 1.4e-17 | 5/9 |
| baseline e02 native | 2 (168) | 12.04 | 9.10 | 0.50 | 591/1093/461 | 5.6e-17 | 5/9 |
| baseline e04 dual | 4 (280) | 31.62 | 28.36 | 0.60 | 679/460/373 | 5.6e-17 | 5/9 |
| baseline e04 native | 4 (280) | 27.80 | 24.88 | 0.55 | 567/358/485 | 5.6e-17 | 5/9 |
| baseline e06 dual | 6 (392) | 31.55 | 29.07 | 0.60 | 708/391/344 | 0 | 5/9 |
| baseline e06 native | 6 (392) | 27.72 | 25.43 | 0.50 | 671/435/381 | 5.6e-17 | 5/9 |
| baseline e08 dual | 8 (504) | 34.62 | 31.73 | 0.65 | 673/213/379 | 0 | 4/9 |
| baseline e08 native | 8 (504) | 28.86 | 25.99 | 0.50 | 648/248/404 | 2.2e-16 | 4/9 |
| baseline e10 dual | 10 (616) | 36.11 | 33.21 | 0.75 | 693/182/359 | 0 | 4/9 |
| baseline e10 native | 10 (616) | 30.34 | 27.47 | 0.50 | 728/302/324 | 1.1e-16 | 4/9 |
| baseline e12 dual | 12 (728) | 34.32 | 30.54 | 0.45 | 715/336/337 | 1.1e-16 | 4/9 |
| baseline e12 native | 12 (728) | 30.20 | 26.29 | 0.30 | 708/332/344 | 0 | 3/9 |
| baseline e14 dual | 14 (840) | 34.42 | 31.16 | 0.70 | 746/228/306 | 5.6e-17 | 4/9 |
| baseline e14 native | 14 (840) | 30.85 | 27.67 | 0.45 | 764/322/288 | 1.1e-16 | 3/9 |
| baseline e16 dual | 16 (952) | 37.92 | 34.42 | 0.70 | 737/223/315 | 0 | 4/9 |
| baseline e16 native | 16 (952) | 33.60 | 30.28 | 0.50 | 734/271/318 | 5.6e-17 | 3/9 |
| baseline e18 dual | 18 (1,064) | 38.50 | 34.96 | 0.70 | 753/252/299 | 1.1e-16 | 3/9 |
| baseline e18 native | 18 (1,064) | 34.81 | 31.46 | 0.55 | 739/260/313 | 0 | 2/9 |
| baseline e20 dual | 20, **부분** 90/112 (1,165) | 39.25 | 35.54 | 0.75 | 728/197/324 | 5.6e-17 | 3/9 |
| baseline e20 native | 20, **부분** 90/112 (1,165) | 35.39 | 32.02 | 0.45 | 763/276/289 | 5.6e-17 | 2/9 |
| improved e02 dual | 2 (168) | 21.17 | 18.84 | 0.50 | 596/492/456 | 5.6e-17 | 5/9 |
| improved e02 native | 2 (168) | 17.29 | 14.98 | 0.45 | 530/418/522 | 5.6e-17 | 5/9 |

이 표는 재현 확인용이다. improved는 epoch 2만 완료됐으므로 **팔 비교나 선택에 쓰지 않는다.**

### 5.3 1,000개 제한에 걸린 epoch 18–20 이미지 (R07 출력)

| 후보 | 이미지 | 남은 최저 점수 | 300개 미만이 남은 클래스 (개수) |
|---|---|---:|---|
| baseline e18 dual | Arty_Top · Virtex6 · Zybo | 0.0017 · 0.0107 · 0.0028 | R 222/C 265/IC 101 · R 226/IC 144/conn 290 · R 195/IC 125 |
| baseline e18 native | Virtex6 · Zybo | 0.0045 · 0.0012 | R 201/IC 140 · R 153/IC 132 |
| baseline e20 dual | Arty_Top · Virtex6 · Zybo | 0.0011 · 0.0087 · 0.0025 | R 211/C 258/IC 112 · R 208/IC 171/conn 285 · R 180/IC 133 |
| baseline e20 native | Virtex6 · Zybo | 0.0043 · 0.0011 | R 189/IC 168 · R 160/C 291/IC 151 |

"300개 미만"은 제한 때문에 잘렸을 가능성이 있는 클래스라는 뜻일 뿐이다. 실제로 잘린 예측의 수와 그로 인한 AP 변화는 저장 자료로 알 수 없다.

### 5.4 제외 객체 (R08 출력)

| cohort | 제외 객체 | 그중 unknown |
|---|---:|---:|
| train | 6,753 | 145 |
| val | 2,050 | 25 |
| 일반 holdout | 2,995 | 44 |
| Pi holdout | 507 | 67 |
| 합계 | 12,305 | 281 |

분석 부록 §3.6의 참고 소계를 전체 제외 객체에 적용하면 다음과 같다. 분류 대상인 FP의 수가 아니라, 소계 목록이 데이터와 맞는지 보는 값이다.

| 소계 | 전체 개수 | 구성 |
|---|---:|---|
| 외형 유사 수동소자 | 246 | resistor network 74 · inductor 69 · emi filter 51 · ferrite bead 30 · fuse 7 · potentiometer 7 · resistor jumper 6 · capacitor jumper 2 |
| 핀·패드 | 649 | pads 332 · pins 317 |
| 표기 | 10,185 | text 9,190 · component text 995 |
| 나머지 (unknown 제외 15종) | 944 | test point 292 · led 214 · button 85 · jumper 85 · transistor 84 · diode 69 · switch 58 · clock 37 · display 5 · zener diode 5 · heatsink 4 · diode zener array 3 · battery 1 · buzzer 1 · transformer 1 |
| unknown (§3.5 3번 범주) | 281 | |

## 6. 이 자료로 확인할 수 없는 것

- 평가기의 1,000개 제한 이전 예측(미저장). 따라서 상한의 AP 영향은 알 수 없다.
- R04 holdout 성능과 나머지 val 18개(아직 실행 전).
- 원본 사진을 보고 확인해야 하는 라벨 정의: pins·pads·DNP 패드, 외형 유사 비대상의 실제 모습.
- D455 실측 전반: 배율·초점·반사·지연.
- 여러 seed 사이의 변동(seed 1개).
