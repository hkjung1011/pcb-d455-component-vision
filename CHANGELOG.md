# 진행 이력

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
