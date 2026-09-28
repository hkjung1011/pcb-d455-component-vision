# Raspberry Pi 촬영 대상: R03 사용 범위

이 작업의 Raspberry Pi는 **촬영·인식 대상**이다. 학습·추론 장비는 RTX 5060 Laptop 8GB PC이며 Raspberry Pi 배포 속도를 측정하지 않았다.

## 서로 다른 네 모델

| ID | 입력과 구조 | 학습 정답 | 출력 |
|---|---|---|---|
| board_yolo11n | RGB → YOLO11n → Detect | 수정 micro-PCB + IoTKITs | Raspberry Pi 보드 bbox 1종 |
| board_yolo11n_seg | RGB → YOLO11n → Segment + mask prototypes | IoTKITs 원본 polygon | Raspberry Pi 보드 외곽 mask 1종 |
| parts_yolo11n | RGB native tile → YOLO11n → Detect | WACV 원본 부품 bbox | 저항·커패시터·IC·커넥터 bbox |
| parts_yolo11s | RGB native tile → YOLO11s → Detect | 위와 같은 분할·정답 | 위와 같은 4종 bbox |

R02의 IC·전해콘덴서·커넥터 semantic 보조 실험은 별도로 보존한다. R03 보드 mask를 부품별 mask로 해석하면 안 된다. 저항을 포함하는 target4 instance mask는 검수한 polygon 정답을 확보한 뒤 학습한다.

## 실제 라벨과 학습 설정

보드 검출: 5,970장(학습 4,179 / 검증 896 / 시험 895). 다른 개발 보드는 음성 사례다. 원본 polygon이 있는 보드 분할: 3,094장(2,148 / 456 / 490), Raspberry Pi polygon 865 / 174 / 166개. 분할에 따라 음성 사진 수가 많으므로 사진 장수와 양성 객체 수를 혼동하지 않는다.

부품: 원본 47장, 이름 기준 보드 설계 29그룹, 4종 bbox 5,896개. 학습 27장/18그룹/3,484개, 검증 7장/5그룹/875개, 일반 시험 11장/5그룹/1,348개, Pi3B 시험 2장/1그룹/189개다. 학습에만 1024px·20% overlap 타일 425개를 만들었다. 부품 정답은 타일에서 6,733회 보이지만 새로운 원본 라벨이 6,733개 생긴 것은 아니다.

| 원본 부품 라벨 | 저항 | 커패시터 | IC | 커넥터 | 합계 |
|---|---:|---:|---:|---:|---:|
| 학습 | 1,390 | 1,570 | 213 | 311 | 3,484 |
| 검증 | 221 | 468 | 64 | 122 | 875 |
| 일반 시험 | 443 | 636 | 93 | 176 | 1,348 |
| Pi3B 시험 | 57 | 104 | 16 | 12 | 189 |
| 전체 | 2,111 | 2,778 | 386 | 621 | 5,896 |

이는 원본 사람이 제공한 bbox를 사용한 실제 수량이다. 현재 새 D455 사람 mask는 0개이며, 학습 타일의 중복 노출과 사람이 새로 그린 라벨 수를 합치지 않는다.

각 클래스의 추가 손실 가중치는 1.0이다. source sampling factor도 1.0이며 별도 클래스 oversampling은 없다. 다만 타일은 균일하게 뽑으므로 큰 원본과 겹친 부품은 더 많이 노출된다. loss gain은 box=7.5, cls=0.5, dfl=1.5, cls_pw=0이다. 설치된 Ultralytics의 segmentation loss gain은 box와 같은 7.5를 사용한다. 이는 모델 파라미터 수 또는 저장된 .pt 파일의 값과 다른 항목이다.

보드 모델은 5에폭, 부품 모델은 20에폭으로 제한했다. AdamW lr0=0.001, lrf=0.01, weight_decay=0.0005, nbs=8, FP32, seed=42다. 세부 설정과 실제 로더 인수는 configs와 각 실행의 args.yaml에 남긴다. 반복 횟수만으로 학습 성공을 판정하지 않고 optimizer state의 실제 step과 파라미터 변경을 확인한다.

## 평가 절차

1. 원본 보드·사진 family를 먼저 나누고 train에만 타일을 만든다. 보드 공개자료는 SHA256·source family·pHash 거리≤4 연결요소를 같은 split에 둔다. 촬영 실물의 시리얼 독립성까지 증명하는 것은 아니다.
2. 부품 모델별 best/last × whole/tile1024를 검증셋에서 비교한다. 원본 GT COCO bbox AP50–95@max100, 동률이면 max300, 그다음 지연 순으로 고른다.
3. 선택된 검증 결과에서 box IoU≥0.5 micro-F1 기준 confidence를 고정한다. 선택 JSON과 체크포인트 SHA를 저장한 뒤 일반 test와 Pi3B test를 실행한다.
4. AP50·AP50–95, 클래스별 AP, 고정 confidence의 precision/recall, 크기별 recall, 음성 사진 오검출률을 기록한다. mask AP는 보드 분할에만 계산한다. 그룹 bootstrap은 recall CI이며 AP CI가 아니다. Pi는 그룹이 하나라 일반화 CI를 만들지 않는다.
5. 절반 해상도와 Gaussian blur 실험은 카메라 열화에 대한 합성 민감도 검사다. 공개 JPEG를 열화시킨 결과를 D455 실측으로 부르지 않는다.

## 먼저 확인해야 하는 기존 데이터 오류

micro-PCB의 README A/H/I 설명과 실제 사진이 맞지 않았다. 13개 코드에서 서로 다른 조건 3장씩, 총 39장을 검토한 결과 Raspberry Pi는 G/H/M이었다. R03은 G/H/M 각 625장을 양성으로 다시 묶고 분할을 새로 만들었다. 기존 저장소와 과거 결과는 그대로 보존한다. A/H/I 기반 과거 Raspberry Pi 점수는 유효한 성능 기준으로 사용할 수 없다.

## D455 적용과 남은 경계

RGB 원본 픽셀 수·초점·조명·촬영 거리부터 확인한다. native tile은 전체 사진을 한 번에 축소하며 잃는 정보를 줄이지만, 촬영 때 없었던 작은 부품 정보는 만들어 내지 못한다. depth는 추후 보드 평면·거리 보조 정보로 사용할 수 있으며 현재 네 모델의 입력은 RGB다.

2026-09-28 SDK 조회에서는 D455 0대가 검색되어 실제 프레임을 취득하지 않았다. 이번 결과만으로 실물 Raspberry Pi 또는 D455 실사용 수준을 통과했다고 표시하지 않는다. 촬영 파이프라인, ROI cascade 전체 성능, 500프레임 지연, target4 부품 mask는 추가 검증 대상이다.

Pi3B 원본에는 target4 이외 unknown 67개가 있다. 이 자료의 189개 정답에 대한 AP는 전체 BOM 또는 모든 부품에 대한 완전한 정답률이 아니다. COCO 사전학습 자료와 공개 시험 사진의 잠재적 중복 여부도 확인되지 않았다.

## 출처

- [IoTKITs v1, AnhTuấn Đỗ Nguyễn](https://data.mendeley.com/datasets/x5thzmkxhy/1): CC BY 4.0. 내려받은 3,108장과 논문 설명 수량은 구분했다. source geometry를 보존하고 전역 분할을 재구성했다.
- [micro-PCB, frettapper](https://www.kaggle.com/datasets/frettapper/micropcb-images): 출처 기록 CC BY 4.0. 기존 로컬 에셋을 사용하고 위 클래스 매핑 오류를 정정했다.
- [Kuo et al. PCB Component Detection, WACV 2019](https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection): 저자 연결 ZIP에서 실제 47장·XML을 확보했다. 공개 다운로드 가능하나 원본 archive에 명시적 라이선스는 없다. 원본 이미지 묶음은 별도로 재포장하지 않는다.
- [Ultralytics 학습 설정](https://docs.ultralytics.com/modes/train/), [추론 설정](https://docs.ultralytics.com/modes/predict/): 실행은 설치된 8.4.120 코드·args를 기준으로 기록한다. 모델 및 라이브러리의 AGPL-3.0/별도 상용 라이선스 조건을 데이터 라이선스와 구분한다.
- Wikimedia Commons 사진은 각 이미지별 출처·작성자·라이선스를 commons_records.json에 기록한다. 사진 24장에 assistant가 작성한 보드 bbox 28개는 검토용 초안이며 정량 평가 정답으로 사용하지 않는다. 모델 예측을 표시한 24장 갤러리는 별도로 제공한다.
