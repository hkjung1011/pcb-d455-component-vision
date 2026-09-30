# 진행 이력

## 2026-09-30 · 보드 1,000장 미세조정 중간 결과

**검증 승격 조건을 통과해 후보 모델을 선택했습니다.** 동일 검증 mAP50–95 **80.41% → 81.46%**, 실제 추가 학습 **13에폭**(최대 20, 후보 best 5). 후보 개발 test는 mAP50 **84.05%**, mAP50–95 **72.43%**이며 선택 고정 후 1회 평가했습니다.

[설정·결과](BOARDS1000_FINETUNE_2026-09-30/README.md) · [선택 근거](BOARDS1000_FINETUNE_2026-09-30/evidence/selection.json) · [복원 안내](BOARDS1000_FINETUNE_2026-09-30/REPRODUCE.md) · [후보·선택 가중치 Release](https://github.com/hkjung1011/pcb-d455-component-vision/releases/tag/boards1000-finetune-2026-09-30)

재사용 개발 test의 Jetson AP50–95는 **30.44% → 19.94%**로 내려갔습니다. 이 결과는 부족한 하위 종류/장면 다양성을 보완할 다음 데이터 실험의 근거로 기록했습니다. 이미 고정한 모델 승격을 test 점수로 뒤집지는 않았습니다.

이 모델은 검증 기준으로 선택한 중간 후보이며 D455 기본 실행 모델로 배포하지 않았습니다. Nano 40장으로 반복 TX2 40장을 교체하는 데이터 보완은 별도의 다음 실험이고, 결과는 아직 확정되지 않았습니다.

데이터 1,000장과 650 / 150 / 200 분할은 [기존 기록](BOARDS1000_2026-09-30/README.md)과 같습니다. 원래 50에폭 모델과 기존 Release를 보존했습니다. 이 실험은 fresh optimizer/schedule의 미세조정이며 에폭 수만 비교하는 대조 실험이 아닙니다. test는 재사용 개발 평가이고 선택은 val에만 근거합니다. Nucleo·포트와 D455 실측은 미검증입니다.

## 2026-09-30 · 보드 1,000장 50에폭 학습 완료

**YOLO11s 8클래스, train 650 / val 150 / test 200장, 실제 50에폭, 검증 선택 epoch 48.** 개발 test 1회 결과는 **mAP50 84.72% / mAP50–95 72.16%**다.

[설정·결과 전체](BOARDS1000_2026-09-30/README.md) · [복원·사용](BOARDS1000_2026-09-30/REPRODUCE.md) · [학습 설정](BOARDS1000_2026-09-30/CONFIGURATION.md) · [클래스별 결과](BOARDS1000_2026-09-30/RESULTS.md) · [현재 상태](BOARDS1000_2026-09-30/STATUS.json) · [가중치와 정확한 데이터 Release](https://github.com/hkjung1011/pcb-d455-component-vision/releases/tag/boards1000-2026-09-30)

전체 설정, 환경 버전, 실제 코드, 1,000장 분할/라벨/그룹/해시, 학습 로그, 곡선과 평가를 보존했다. Release에는 best/last/공식 초기 가중치와 정확한 1,000장·YOLO 라벨, 이전 10에폭 모델이 있다. 이전 파일럿은 [별도 이력](BOARDS1000_2026-09-30/history/pilot10/README.md)으로 남겼다.

test는 이전 실험 사진을 재분할한 개발 평가이며 새 최종시험이 아니다. Jetson AP50–95 30.44%가 주요 약점이다. Nucleo·포트는 이번 모델에 없고, 이 모델의 D455 실측 성능은 미검증이다. 기존 D455 실시간 모델은 변경하지 않았다. 아래 준비 기록의 `training_ready=false`는 원래 9클래스/포트 스냅샷이며 이번 8클래스 완료 상태와 구분한다.

## 2026-09-29 · 보드 종류·포트 데이터 준비

[준비 결과·매핑·재실행 안내](BOARDS_PORTS_PREP_2026-09-29/README.md) · [준비 상태](BOARDS_PORTS_PREP_2026-09-29/STATUS.json) · [검증 결과](BOARDS_PORTS_PREP_2026-09-29/reports/evidence/verification.json)

Roboflow ZIP 3개 1,509장을 확보하고, IoTKITs와 합쳐 보드 2,013장(train 1,701/val 206/test 106)을 구성했다. Nucleo 393장은 모두 train이므로 val/test 지원이 없고, 포트 549장은 누락 라벨 검수가 남아 있다. **이번 추가 작업은 학습·test 평가를 실행하지 않았으며 두 모델 모두 `training_ready=false`다.** 코드·매핑·출처·해시·검수 기록을 보존했다.

## R04 · 2026-09-29

두 번째 Claude 검토의 타일 잘림·GT 정의·실제 optimizer count·validation 분포·카메라 픽셀 가정을 원본과 대조했다. 동일 조건의 기준/개선 YOLO11s 각각20에폭이내/1,165회 직접갱신을 수행하고40개val후보의고정선택 후 개발holdout을 평가했다. 원본시험GT와R03Release는보존했다. D455실물/target4mask는미완료이며30+200장 계획을 기록했다.

## R03 · 2026-09-28

Raspberry Pi 촬영 대상 확정. 공개 보드·부품 자료 확장, 클래스 매핑 G/H/M 정정, 보드 검출/보드 mask/부품 n/s 네 모델 학습. 검증셋으로 모델·confidence를 고정한 뒤 원본 COCO 평가, 별도 Pi3B 시험, 모의 열화, 정성24장 추론과 결과 감사를 완료했다. D455 및 target4 instance mask는 미완료다.

## R02 · 2026-09-28

PCB-Vision source3 semantic 보조 실험을 수행했다. 실제20에폭·104 optimizer 갱신, 원본 test8장의 foreground mIoU60.59%다. 저항 포함4종 instance mAP와는 다른 과제다. 초기 AMP/학습 누적 설정 문제와 수정 기록을 보존한다.

## R01 · 계획 이력

D455 기반 부품 검출·분할과 라벨링·평가 계획을 작성했다. 계획과 실적을 구분하고, R02/R03의 실제 설정·결과가 최신 상태다. Claude 검토의 적용/수정/미적용 항목은 R03 반영표에 남겼다.
