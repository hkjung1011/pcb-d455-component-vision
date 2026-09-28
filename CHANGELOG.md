# 진행 이력

## R03 · 2026-09-28

Raspberry Pi 촬영 대상 확정. 공개 보드·부품 자료 확장, 클래스 매핑 G/H/M 정정, 보드 검출/보드 mask/부품 n/s 네 모델 학습. 검증셋으로 모델·confidence를 고정한 뒤 원본 COCO 평가, 별도 Pi3B 시험, 모의 열화, 정성24장 추론과 결과 감사를 완료했다. D455 및 target4 instance mask는 미완료다.

## R02 · 2026-09-28

PCB-Vision source3 semantic 보조 실험을 수행했다. 실제20에폭·104 optimizer 갱신, 원본 test8장의 foreground mIoU60.59%다. 저항 포함4종 instance mAP와는 다른 과제다. 초기 AMP/학습 누적 설정 문제와 수정 기록을 보존한다.

## R01 · 계획 이력

D455 기반 부품 검출·분할과 라벨링·평가 계획을 작성했다. 계획과 실적을 구분하고, R02/R03의 실제 설정·결과가 최신 상태다. Claude 검토의 적용/수정/미적용 항목은 R03 반영표에 남겼다.
