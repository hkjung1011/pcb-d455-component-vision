# 보드 1,000장 · Jetson 학습 자료 보완 · 2026-09-30

**검증 승격 기준을 통과하지 못해 이전 미세조정 선택 모델을 유지했습니다.** 이번 후보의 test 실행은 **0회**입니다. 유지한 이전 모델의 성능은 이전 미세조정 기록을 따릅니다.

기존 Jetson train의 반복 TX2 40장을 검수한 Nano 40장으로 바꿔 **TX2 27장 + Nano 40장**을 학습에 넣었습니다. 총 1,000장과 650 / 150 / 200 분할을 유지했고 val/test 350장의 이미지·라벨·index record는 바이트까지 그대로입니다. 다른 클래스 train 583장도 유지했습니다.

실제 추가 학습은 **11에폭**(최대 30, 후보 best 1)이며 initializer는 원래 50에폭 best입니다. 동일 val mAP50–95는 원래 baseline **80.41%**, 직전 선택 모델 **81.46%**, 새 후보 **81.12%**입니다. 사전에 고정한 global·클래스·Jetson·직전 모델 대비 조건을 모두 확인해 선택했습니다.

Jetson val AP50–95는 원래 **61.88%**, 새 후보 **61.23%**입니다. mosaic 종료는 최대 30에폭 일정의 21에폭부터 예정됐지만 실제 11에폭에서 종료되어 해당 단계에 도달하지 않았습니다.

이전 원래 모델 → 이전 미세조정 모델의 역사적 Jetson test AP50–95는 **30.44% → 19.94%**입니다. 이번 후보의 점수가 아니며 승격 조건은 val만 사용했습니다.

| 내용 | 파일 |
|---|---|
| 결과·학습곡선·클래스별 비교 | [RESULTS.md](RESULTS.md) |
| 설정·선택 규칙 | [train.json](configs/train.json) · [승격 정책](configs/promotion_policy.json) · [선택 원자료](evidence/selection.json) |
| 결과와 선택의 독립 감사 | [outcome_audit.json](evidence/outcome_audit.json) |
| 데이터 교체·검수 근거 | [보완 검증](evidence/dataset/data_correction_verification.json) · [검수](evidence/dataset/visual_review.json) |
| 현재 후보 상태 | [STATUS.json](STATUS.json) |
| 가중치·정확한 1,000장·완전 복원 ZIP | [Release](https://github.com/hkjung1011/pcb-d455-component-vision/releases/tag/boards1000-balanced-jetson-2026-09-30) |
| 다른 PC 복원·검증·새 학습 | [REPRODUCE.md](REPRODUCE.md) |
| Git 묶음 해시 | [GIT_MANIFEST.json](GIT_MANIFEST.json) |

전체 Release에는 exact1,000장/라벨, candidate/last/selected/original-baseline 가중치, source annotations, 검수·제외 근거·GT contact sheets, 실제 코드·설정·환경·라이선스가 있습니다. `local_package/`는 원래 로컬 결과·manifest를 수정 없이 보존했습니다. `portable/`는 새 PC용 상대경로 data.yaml과 별도 `SNAPSHOT_MANIFEST.json`을 갖습니다. YAML을 바꾸면서 portable 전체 fingerprint가 달라지므로 원래 학습 fingerprint `3120af5994f7494f608b31b59b48833eebee8423408f5a0190f516f6e708ba6f`와 구분합니다. 이미지·라벨 바이트는 동일합니다.

선택은 val만 사용하며 test는 반복 사용한 개발 평가입니다. 다음 데이터 실험의 방향에 앞선 개발 test 오류를 활용했으므로 새 최종시험으로 해석할 수 없습니다. 데이터와 학습 설정을 함께 바꿔 순수 데이터 효과/에폭 효과를 분리하지 않았습니다. Nano 출처 그룹·pHash·육안 라벨 검사는 실제 실물/촬영 세션 독립성의 증명이 아닙니다. Nucleo·포트·D455 실측은 미검증이고 D455 기본 프로그램에 배포하지 않았습니다.

[원래 50에폭](https://github.com/hkjung1011/pcb-d455-component-vision/tree/main/BOARDS1000_2026-09-30)과 [중간 미세조정](https://github.com/hkjung1011/pcb-d455-component-vision/tree/main/BOARDS1000_FINETUNE_2026-09-30) 기록·Release를 모두 보존했습니다.
