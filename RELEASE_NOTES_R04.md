# R04 · Claude 검토 반영과 동일예산 비교

기준/개선 YOLO11s 각20에폭이내·실제optimizer1,165회. 새groupedvalidation으로40후보선택을동결하고학습미사용일반/Pi개발holdout평가를완료했다.

| 모델 / 개발 holdout | bbox mAP50–95@100 | bbox mAP50–95@300 | 고정 confidence P / R |
|---|---:|---:|---:|
| baseline / general_test | 33.10% | 36.02% | 82.21% / 51.41% |
| baseline / pi_test | 35.35% | 35.33% | 81.30% / 52.91% |
| improved / general_test | 39.56% | 42.77% | 83.78% / 62.46% |
| improved / pi_test | 42.04% | 42.08% | 76.04% / 77.25% |

D455실물/target4instance mask는미완료. R03에서이미살펴본holdout이며미관측최종시험이아니다. 신규사진/사람라벨/mask모두0. 원본사진/타일이미지/가상환경은포함하지않는다.

`PCB_D455_R04.zip`에는모델후보20개·초기가중치·코드·label/weight기록·평가·그래프·검토표가포함된다. R03Release는유지한다.

SHA256 `aa7b28a83fe52515ed7bfdb85a3bdb94e1ab841daf6bd9cf37f1c1466a5f06de`
