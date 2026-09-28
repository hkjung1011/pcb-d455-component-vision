# 시험 예측 육안 확인 · 2026-09-29

학습과 시험이 끝난 후 `evaluation/test/val_batch0_labels.jpg`와 `val_batch0_pred.jpg`를 나란히 확인했다. 이 관찰로 가중치·confidence·학습 입력을 다시 선택하지 않았다.

- 이 네 장 예시에서는 하나의 긴 스크래치 주변에 여러 scratch 박스가 중복해서 생겼다. 일부 작은 edge_bite와 open은 빠졌다.
- stains_enbedded와 일부 edge_bite의 위치는 찾았지만, 네 장 예시만으로 전체 클래스 성능을 판단하지 않는다.
- 전체 시험셋의 scratch AP50–95는 10.78%였다. 전체 mAP만 제시하면 이 약점을 가리므로 클래스별 AP와 고정 confidence의 FP/FN을 같이 기록한다.
- 이것은 원본 정답의 오류를 확정하는 검수가 아니다. 라벨 오류를 이유로 FP를 임의로 빼지 않았다.

사진 자체는 로컬에 보존하고 배포 패키지와 GitHub에는 넣지 않았다. D455 이미지나 실사용 장면의 확인은 수행하지 않았다.
