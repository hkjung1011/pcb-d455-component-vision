# R04 보조 분석 — holdout

저장 예측 CPU 분석. 모델·confidence·GT·1차 결과는 변경하지 않았다.

| 실험/집합 | pooled AP300 | 그룹 평균 AP300 |
|---|---:|---:|
| baseline__general_test | 36.021% | 33.483% |
| baseline__pi_test | 35.331% | 35.331% |
| improved__general_test | 42.770% | 39.591% |
| improved__pi_test | 42.077% | 42.077% |

AP는 이미지 전체 최대1000개 제한 후 COCO 클래스별 최대300개다. 단일 seed와 작은 그룹 수에서 통계적 우월성은 주장하지 않는다.
FP 분류·source_type·교차표·클래스/크기 분모는 대응 JSON에 있다. unknown 위 FP를 정답이나 라벨 오류로 단정하지 않으며 보정 AP는 계산하지 않았다.
