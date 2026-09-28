# R04 저장 결과 독립 감사

상태: **PASS_SAVED_RESULTS_AUDIT_NOT_D455_VALIDATED**

GPU 추론·재학습 없이 checkpoint를 CPU로 읽고, 저장 예측의 COCO와 별도 구현 매칭을 재계산했다. 실제 D455나 새로운 최종 시험을 검증한 결과는 아니다.

통과한 검사: 10982개. 실패: 0개.

| arm | 도달 epoch | 마지막 partial | 직접 optimizer 호출 | 파라미터 step 최소/최대 | 실제 draw | 선택 |
|---|---:|---|---:|---|---:|---|
| baseline | 20 | True | 1165 | 1165/1165 | 8872 | epoch20 / dual |
| improved | 20 | True | 1165 | 1165/1165 | 8872 | epoch16 / dual |

검사 범위: 동일 초기 가중치와 실제 초기 파라미터, 직접 호출·EMA·파라미터 step 분리, 반복 라벨 노출, 후보 SHA·표준 모델, 전체 VAL 후보 순위와 confidence grid, 선택 동결 시각, source GT·unknown 정의, 저장 예측 AP/TP/FP/FN와 클래스×크기 합계.

일반/Pi 자료는 R03 진단에 사용한 개선용 holdout이다. unknown·미실장 footprint 해석을 임의 수정하지 않았다. 새 mask나 수작업 라벨은 추가하지 않았다.

상세 수치·파일 SHA·개별 검사 결과는 `final_results_audit.json`에 기록했다.
