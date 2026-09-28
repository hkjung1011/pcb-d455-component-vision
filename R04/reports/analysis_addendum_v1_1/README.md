# R04 분석 부록 v1.1 실행 기록

사용자 진행 지시에 따라 Claude 초안을 확정하고, 기존 R04 학습·평가 프로토콜은 유지했다.

- `ANALYSIS_ADDENDUM_v1.1.md`, `freeze_record.json`: holdout 전 분석 규칙·해시·시각. 일부 val/R03 holdout을 본 뒤 작성한 계획이다.
- `outputs/pre_holdout_audit.json`: 학습·val40·선택 동결을 독립 CPU 재계산한 holdout 전 감사.
- `outputs/supplementary_val.*`: 그룹별/그룹 제외 순위·20쌍 AP·FP 분류. 주 선택을 변경하지 않았다.
- `outputs/supplementary_holdout.*`: 개발 holdout 그룹·클래스/크기와 FP 진단.
- `repro/`, `REVIEW_APPENDIX.md`: 사용자가 전달한 Claude 재현 자료를 원형 보존했다.

이 폴더의 Python 파일은 실행 당시 소스의 바이트 사본이다. 재현은 원래 workspace의 `work/r04_addendum/` 배치와 `work/r04`/`work/r03` 로컬 데이터·가중치를 전제로 한다. 이미 완료된 holdout을 다시 실행하거나 결과를 덮어쓰는 용도가 아니다. 원본 사진은 포함하지 않는다.
