# 보드 검출 1,000장 · 50에폭 · 2026-09-30

**YOLO11s 8클래스 모델을 총 1,000장(train 650 / val 150 / test 200)으로 50에폭 학습했습니다. 검증 기준 epoch 48을 선택하고 test를 1회 평가한 결과 mAP50 84.72%, mAP50–95 72.16%입니다.**

| 목적 | 문서·파일 |
|---|---|
| 결과, 학습 곡선, 클래스별 AP | [RESULTS.md](RESULTS.md) |
| 전체 설정과 알고리즘 | [CONFIGURATION.md](CONFIGURATION.md) · [실제 전체 args](configs/effective_args.yaml) |
| 다른 PC에서 복원·검증·추론·새 학습 | [REPRODUCE.md](REPRODUCE.md) |
| 데이터 출처·클래스·분할·한계 | [DATASET.md](DATASET.md) · [출처와 라이선스](SOURCES.md) |
| 모델 용도·성능·한계 | [MODEL_CARD.md](MODEL_CARD.md) |
| 현재 상태·해시 | [STATUS.json](STATUS.json) · [전체 묶음 해시 목록](BUNDLE_MANIFEST.json) |
| 이전 10에폭 파일럿 | [history/pilot10](history/pilot10/README.md) |
| 가중치·정확한 데이터·라벨·초기/마지막 모델 | [Release 다운로드](https://github.com/hkjung1011/pcb-d455-component-vision/releases/tag/boards1000-2026-09-30) |

Git에는 문서, 실제 코드 스냅샷, 설정, 환경 버전, 분할·그룹·파일 해시, 학습 로그와 평가 표/그림을 저장했습니다. Release의 `BOARDS1000_FULL_SNAPSHOT.zip`에는 위 자료에 **학습에 사용한 1,000장과 YOLO 라벨, best/last/초기 가중치, 이전 10에폭 가중치**를 더했습니다. [현재 best만 다운로드](https://github.com/hkjung1011/pcb-d455-component-vision/releases/download/boards1000-2026-09-30/boards8-1000images-yolo11s-best.pt)할 수도 있습니다. 이 저장소는 비공개이며 다운로드에는 저장소 접근 권한이 필요합니다.

![학습 곡선](learning_curves.png)

test는 이전 test 106장·val 86장·train 8장을 재분할한 개발 평가입니다. 이번 모델의 학습·선택에는 test를 사용하지 않았지만 새로운 최종시험으로 해석할 수 없습니다. Nucleo와 포트는 이번 모델에 포함하지 않았습니다. Jetson AP50–95 30.44%가 주요 약점입니다. 실제 D455 성능과 실물/촬영 세션 독립성은 미검증입니다.

이전 [9클래스·포트 준비 상태](https://github.com/hkjung1011/pcb-d455-component-vision/tree/main/BOARDS_PORTS_PREP_2026-09-29)는 당시 기록으로 보존했습니다. Nucleo val/test 부족과 포트 누락 라벨 검수는 남아 있습니다. 50에폭 모델은 기존 D455 실시간 프로그램에 자동 적용하지 않았습니다.
