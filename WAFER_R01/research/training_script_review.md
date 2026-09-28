# 웨이퍼 파일럿 코드 사전 확인 · 2026-09-29

대상은 `work/wafer_r01/scripts/prepare_data.py`, `train_pilot.py`와 실제 설치된 Ultralytics 8.4.120 / PyTorch 2.12.1+cu130이다. 이번 확인은 정적 분석·AST 파싱·import 확인이며 GPU 학습을 실행하지 않았다.

확인 결과:

- 두 스크립트 AST 파싱 및 DetectionTrainer/unwrap_model import 통과.
- torch optimizer의 `register_step_post_hook` 존재. installed `engine/trainer.py`의 on_pretrain_routine_end는 optimizer/EMA와 dataloader 구성 뒤에 실행되므로 실제 optimizer 호출을 세기 위한 등록 시점이 적합하다.
- installed trainer는 각 에폭 validation 뒤 `_handle_nan_recovery`를 부르므로 fitness의 finite 검사는 값이 준비된 시점에 수행된다.
- stock final_eval은 val에서 선택한 best를 다시 검증하고 on_fit_epoch_end를 추가 호출한다. 스크립트의 `in_final_eval` guard는 이 이벤트를 11번째 학습 에폭으로 세지 않는다.
- on_train_end 시점에 optimizer 인스턴스와 상태는 유지된다. 실제 호출 수, parameter step, EMA 갱신, 파라미터 변화 해시를 기록하는 경로가 존재한다.
- 설치된 `utils/metrics.py`의 detection fitness 가중치는 `[0,0,0,1]`이다. validation mAP50–95 기준으로 best를 고른다는 기록과 일치한다.
- 가중치 해시를 고정해 selection_frozen을 기록한 후 test를 실행한다. test P/R의 F1 최적 임계값을 고정 운영점으로 오인해 보고하지 않는 처리도 확인했다.
- 설치된 기본 cls_pw는 0.0이다. 설정에 명시하면 외부 기본값 변경을 추적하기 쉽다. conf .001 / iou .7 / max_det300도 검증 조건으로 명시하는 것이 좋다.

자료 검사에서 발견한 중요한 문제:

- 완전 같은 사진 49그룹에서 라벨 불일치가 확인됐다. 단순히 중복 사진을 같은 split에 넣는 것만으로는 서로 충돌하는 GT가 해결되지 않는다.
- root 구현자가 98장 전부를 훈련 전에 격리하고, pHash≤4 후보를 같은 appearance group으로 묶도록 전처리를 수정했다. 이 문서는 그 수정에 대한 GPU 결과 확인을 대신하지 않는다.
- pHash 및 파일명 계열은 물리 웨이퍼 ID가 아니다. 보지 않은 물리 웨이퍼·로트 성능은 NOT VERIFIED로 남긴다.
- 나머지 라벨의 누락 가능성과 현미경 촬영/D455 차이는 그대로 남는다. 형식·좌표 검사 통과를 전체 GT 의미 검수 통과로 표현하지 않는다.

추가 학습을 늘리는 제안은 하지 않는다. 우선 고정된 10에폭 파일럿의 실행 결과와 독립 holdout 결과를 확인한다.
