# 웨이퍼 표면 결함 R01 · 공개 현미경 이미지 파일럿

YOLO11n 10에폭·실제 optimizer 1,932회. bbox6종, 이미지2,034장/라벨3,394개. 검증셋의 AP로 best 가중치를 선택하고 SHA를 고정한 뒤 학습 미사용287장/496개를 평가했다.

**mAP50 73.73% · mAP50–95 36.42%** (Ultralytics8.4.120, confidence floor0.001, NMS IoU0.7, 이미지 전체 max_det300).

고정 confidence0.25의 별도 TP/FP/FN 분석은 test 전에 기준과 코드를 기록했으며 현장 보정값이 아니다. COCO 보조 AP는 다른 matching·보간·JSON 반올림을 사용하므로 원래 AP와 구분한다.

ZIP에는 best/last 가중치, 코드, 분할/라벨 수량, 해시, 평가·감사와 그래프가 있다. 원본 사진·VOC XML·변환 라벨·예측 사진은 포함하지 않는다. D455 실사용, 새 물리 웨이퍼/lot 일반화 및 mask 성능은 미검증이다. 기존 R04/R03 태그와 파일은 보존했다.

SHA256 `92650df814ce9410a51119c53db3fd0fc42b1501b7c9d1954e13ced43b3183c1`
