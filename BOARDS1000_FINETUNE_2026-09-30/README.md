# 보드 1,000장 추가 미세조정 · 중간 실험 · 2026-09-30

**검증 승격 조건을 통과해 후보 모델을 선택했습니다.** 기존 50에폭 모델에서 낮은 학습률과 mosaic 종료 조건으로 새 optimizer/schedule을 시작했습니다. 최대 20에폭 중 실제 13에폭을 진행하고 fine-tune epoch 5을 후보로 선택했습니다.

동일 프로토콜의 검증 mAP50–95는 **80.41% → 81.46%**입니다. 후보 개발 test는 mAP50 **84.05%**, mAP50–95 **72.43%**이며 선택 고정 후 1회 평가했습니다.

재사용 개발 test의 Jetson AP50–95는 **30.44% → 19.94%**로 내려갔습니다. 이 결과는 부족한 하위 종류/장면 다양성을 보완할 다음 데이터 실험의 근거로 기록했습니다. 이미 고정한 모델 승격을 test 점수로 뒤집지는 않았습니다.

이 기록은 검증 기준으로 선택한 중간 후보입니다. 모든 클래스에서 개선됐다는 뜻은 아니며 D455 기본 실행 모델로 배포하지 않았습니다. Nano 40장으로 반복된 TX2 40장을 바꾸는 데이터 보완은 별도의 다음 실험이고, 이 기록 작성 시점에는 그 결과가 확정되지 않았습니다.

| 확인할 내용 | 파일 |
|---|---|
| 결과·학습 곡선·클래스별 비교 | [RESULTS.md](RESULTS.md) |
| 실제 설정 | [train.json](configs/train.json) · [validation.json](configs/validation.json) |
| 선택 근거 | [selection.json](evidence/selection.json) · [동결한 승격 규칙](configs/promotion_policy.json) |
| 다른 PC에서 자료 복원 | [REPRODUCE.md](REPRODUCE.md) |
| 상태와 해시 | [STATUS.json](STATUS.json) · [Git 파일 목록](GIT_MANIFEST.json) |
| 후보·선택 가중치와 원래 결과 전체 | [새 Release](https://github.com/hkjung1011/pcb-d455-component-vision/releases/tag/boards1000-finetune-2026-09-30) |
| 정확한 1,000장·라벨·초기 가중치 | [기존 데이터 Release](https://github.com/hkjung1011/pcb-d455-component-vision/releases/tag/boards1000-2026-09-30) |

전체 1,000장의 train 650 / val 150 / test 200 분할, 이미지와 라벨은 이전 실험과 같습니다. 이번 ZIP은 데이터 사진을 중복 동봉하지 않으며 이전 `BOARDS1000_FULL_SNAPSHOT.zip`을 함께 복원해야 합니다. test는 여러 실험에서 재사용한 개발 평가이며 새로운 최종시험이 아닙니다. 모델 선택은 val 규칙으로만 결정했습니다.

Jetson은 train의 TX2 출처 그룹이 3개이고 Nano train 그룹은 0개입니다. 설정 변경으로 부족한 실물·장면 다양성이 보충되지는 않습니다. Nucleo·포트와 D455 실측 정확도는 이번 실험 범위에 포함되지 않습니다. 기존 D455 실행 프로그램에 모델을 자동 적용하지 않았습니다.

`evidence/original_local_artifact_manifest.json`은 원래 로컬 결과 묶음의 해시 기록입니다. Git용 문서의 가중치 링크를 Release로 바꿨으므로 Git의 현재 기준은 `GIT_MANIFEST.json`입니다. ZIP의 `local_package/`에는 원래 로컬 결과와 manifest를 그대로 보존했고, 전체 ZIP 내용은 `ZIP_MANIFEST.json`으로 검증합니다.
