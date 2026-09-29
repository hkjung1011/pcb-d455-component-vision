# 재실행 입력과 경로

이 문서는 실제 사용한 파이프라인을 복원하는 절차입니다. 원본 이미지가 빠진 GitHub 스냅샷을 즉시 실행 가능한 완성 데이터셋으로 표시하지 않습니다.

## 당시 환경

- Python 3.11.9, 데이터 의존성은 [requirements-data.txt](requirements-data.txt).
- 당시 작업 폴더: `C:\Users\hkjun\Documents\Codex\2026-09-27\new-chat\work\boards_ports`.
- 당시 실행기: `C:\Users\hkjun\Documents\mcu-vision\.venv-yolo11\Scripts\python.exe`.
- 학습 의존성 버전은 [runtime.json](runtime.json)에 기록만 했습니다. 데이터 준비에는 torch/ultralytics가 필요하지 않습니다.

## 필요한 원자료

1. IoTKITs 원본 입력 JSON 3,108개 레코드와 각 `image`가 가리키는 사진. 입력 JSON은 `provenance/iotkits_input_records.json.gz`에 보관했습니다. 압축 해제한 JSON SHA-256은 `411b9683b49f4a46f00bc0b4021a83646e69c77e25da9bfd5f4392ced9034680`입니다.
2. [다운로드 목록](reports/download_inventory.json)에 기록된 Roboflow ZIP 3개를 `boards_ports/raw/`에 둡니다. 파일 이름과 SHA-256을 먼저 맞춥니다.
3. Commons 평가용 이미지 44장을 별도로 복원합니다. 파일명/해시는 [검증 기록](reports/evidence/verification.json)의 `holdout_inventory`에 있습니다. 없으면 조립이 중단되는 것이 정상입니다.

원래 디렉터리 배치는 아래와 같습니다. `pipeline/` 내용을 복원 작업 폴더의 `boards_ports/`로 복사합니다.

```text
work/
  boards_ports/       # common.py, class_map.json, 나머지 .py
    raw/             # 확보한 ZIP 3개
    staging/         # 아래 명령이 생성
    datasets/        # 아래 명령이 생성
  r03/assets/
    board_records.json
    commons_holdout/images/  # 44장
```

새 PC에서는 JSON의 `image` 경로를 실제 파일 경로로 바꿔야 합니다. 이때 레코드 ID, source, 원본 파일 SHA, group은 보존하고 이미지 해시를 대조합니다. `common.py`의 `IOTKITS_RECORDS`, `HOLDOUT_IMAGES` 기본값이 위 배치를 사용합니다. 이러한 경로 변경과 재실행 뒤에는 새 검증 결과를 만들며 과거 검증 JSON을 복사해 준비 완료로 취급하지 않습니다.

## 데이터 준비 명령

복원한 `boards_ports` 폴더에서 아래 순서로 실행합니다.

```powershell
python build_iotkits.py
python import_roboflow.py
python assemble.py
python verify_datasets.py
```

기존 staging을 사용하는 현재 PC에서는 [REBUILD_DATA_ONLY.ps1](reports/REBUILD_DATA_ONLY.ps1)로 뒤의 3단계만 다시 실행할 수 있습니다. 해당 스크립트는 신규 clone 초기화용이 아닙니다.

원자료가 동일한 이번 스냅샷의 기대값은 보드 2,013장(train 1,701/val 206/test 106), Nucleo train 393/val 0/test 0, 포트 미생성입니다. `structure_pass=true`여도 `training_ready=false`입니다. 이 결과가 준비 미완료를 정확히 드러내는 것이며 통과시키기 위해 데이터를 임의로 나누면 안 됩니다.

포트 검수가 끝난 이미지에만 `annotation_completeness.ports=verified_complete`를 부여합니다. 신규 Nucleo ZIP을 추가하거나 매핑·holdout을 변경하면 조립과 검증을 다시 수행합니다. 이 절차에는 학습과 test 평가 명령을 넣지 않았습니다.
