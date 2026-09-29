# 다음 작업: 데이터 준비 계속 (학습 미승인)

작업 폴더: `C:\Users\hkjun\Documents\Codex\2026-09-27\new-chat\work\boards_ports`
현재 결과 보고서: `C:\Users\hkjun\Documents\Codex\2026-09-29\iotkits-2-stm32-jpg-iotkits-stm32\outputs\README.md`
재현 명령 스크립트: `C:\Users\hkjun\Documents\Codex\2026-09-29\iotkits-2-stm32-jpg-iotkits-stm32\outputs\REBUILD_DATA_ONLY.ps1`

- Roboflow ZIP 3개 1,509장 가져오기 완료. 원본 ZIP과 staging을 유지한다.
- 보드 2,013장: train 1,701 / val 206 / test 106. 구조 검증 통과, training_ready=false.
- Nucleo 393장은 모두 train에 있으며 val/test 독립 자료가 필요하다.
- 포트 549장의 매핑 레코드가 있으나 완전성 미확인으로 ports_v1은 생성하지 않았다.
- Nucleo 공개 원본 107장은 https://app.roboflow.com/hk-jung/stm32-3eajy/1 에 증강 없는 버전을 생성했다. Chrome ERR_BLOCKED_BY_CLIENT로 ZIP 다운로드는 완료되지 않았다.

1. 위 107장 ZIP을 확보하면 raw에 저장하고 import_roboflow.py를 실행한다. 클래스 의미와 촬영 세션을 직접 확인한다.
2. Nucleo 독립 평가 자료를 확보하거나 pHash 후보 판별 근거를 검증한다. 임계값만 낮춰 준비 상태를 통과시키지 않는다.
3. 포트 누락 라벨(특히 dpiwf의 HDMI류)을 보완하고 모든 대상 라벨을 검수한 이미지에만 verified_complete를 부여한다.
4. assemble.py, verify_datasets.py 실행 후 실제 검증 JSON과 클래스별 그룹 지원 수를 확인한다.
5. 이번 승인은 데이터 준비까지다. 학습과 test 평가는 실행하지 않는다.

세부 의미와 검수 범위는 결과 보고서 및 PORTS_MAPPING_REVIEW.md, GROUP_SUPPORT_REVIEW.md에 기록되어 있다.
