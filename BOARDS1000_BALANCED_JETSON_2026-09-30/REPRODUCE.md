# 완전 복원과 새 학습

비공개 저장소에 접근할 수 있는 계정에서 다운로드합니다. 이번 ZIP 하나에 정확한 새 1,000장과 모든 필요 가중치가 있습니다.

```powershell
gh release download boards1000-balanced-jetson-2026-09-30 --repo hkjung1011/pcb-d455-component-vision --pattern BOARDS1000_BALANCED_JETSON_FULL_SNAPSHOT.zip --pattern SHA256SUMS.txt
Get-FileHash .\BOARDS1000_BALANCED_JETSON_FULL_SNAPSHOT.zip -Algorithm SHA256
Get-Content .\SHA256SUMS.txt
Expand-Archive -LiteralPath .\BOARDS1000_BALANCED_JETSON_FULL_SNAPSHOT.zip -DestinationPath .\restored
Set-Location .\restored\BOARDS1000_BALANCED_JETSON_2026-09-30\portable
python .\scripts\verify_snapshot.py
```

`verify_snapshot.py`는 표준 라이브러리만 사용해 모든 portable 파일 해시, exact650/150/200 이미지·라벨·그룹, 350장 val/test의 동결 해시를 확인합니다. 이미지·라벨의 원래 바이트와 portable data.yaml의 차이를 분리해 기록했고 기존 Commons/pHash 검사를 새로 실행하는 도우미는 아닙니다. `local_package/BOARDS1000_BALANCED_JETSON_2026-09-30`에는 수정하지 않은 원래 패키지와 `ARTIFACT_MANIFEST.json`이 있습니다. `ZIP_MANIFEST.json`은 ZIP 전체, `portable/SNAPSHOT_MANIFEST.json`은 경로를 바꾼 복원 묶음, Git의 `GIT_MANIFEST.json`은 가중치/원본 이미지를 생략한 탐색용 기록의 해시입니다.

환경은 [전체 런타임](environment/runtime.json)과 [패키지 목록](environment/requirements-lock.txt)에 있습니다. Python 3.11 환경에서 예를 들면:

```powershell
py -3.11 -m venv C:\BoardEnvs\balanced11
C:\BoardEnvs\balanced11\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r .\environment\requirements-lock.txt
C:\BoardEnvs\balanced11\Scripts\python.exe .\scripts\reproduce_training.py --action check
C:\BoardEnvs\balanced11\Scripts\python.exe .\scripts\reproduce_training.py --action train --output C:\BoardExperiments --name balanced_reproduction_01
```

새 출력은 보존 스냅샷 바깥에 만들고, initializer는 `weights/original-baseline-best.pt`, 설정은 `configs/train.json`을 사용합니다. 도우미는 새 train/val만 실행하고 test를 자동 재실행하지 않습니다. 원래 선택·test 결과를 덮어쓰지 않습니다. 실제 runner의 관측 callback·승격 평가 전체를 자동 복제하는 도우미는 아니며 설치 자체는 이번 보관 작업에서 새 환경으로 시험하지 않았습니다. OS/드라이버/라이브러리 차이로 가중치나 점수가 비트 단위로 같다고 보장하지 않습니다.

새 환경·실험 결과는 동결된 스냅샷 밖에 둡니다. Python `__pycache__`와 Ultralytics `.cache`는 재실행 중 생성되는 캐시라 manifest 파일 집합 검사에서 제외하지만, 기록된 파일 해시는 모두 그대로 검사합니다. mosaic 종료는 최대 30에폭 일정의 21에폭부터 예정됐지만 실제 11에폭에서 종료되어 해당 단계에 도달하지 않았습니다.

```python
from ultralytics import YOLO
model = YOLO('weights/selected-best.pt')
model.predict(source='board_photo.jpg', imgsz=640, conf=0.25, save=True)
```

선택 모델 SHA: `fa0b8fb5fc11de748e9d8051b93fc57dde5c6cc34db6167045e3fee9e4912752`. conf0.25는 예시이며 test로 최적화한 운영 임계값이 아닙니다. 유지/승격 판단은 [selection.json](evidence/selection.json)의 val 정책을 따릅니다. D455 기본 모델로 배포하지 않았습니다.
