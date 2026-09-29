# 복원과 사용

아래 PowerShell 명령은 [Release](https://github.com/hkjung1011/pcb-d455-component-vision/releases/tag/boards1000-2026-09-30) ZIP을 작업 폴더에 받은 뒤 실행합니다. 비공개 저장소 접근 권한이 있는 GitHub 계정을 사용합니다.

```powershell
gh release download boards1000-2026-09-30 --repo hkjung1011/pcb-d455-component-vision --pattern BOARDS1000_FULL_SNAPSHOT.zip --pattern SHA256SUMS.txt
Get-FileHash .\BOARDS1000_FULL_SNAPSHOT.zip -Algorithm SHA256
Get-Content .\SHA256SUMS.txt
Expand-Archive -LiteralPath .\BOARDS1000_FULL_SNAPSHOT.zip -DestinationPath .\restored
Set-Location .\restored\BOARDS1000_2026-09-30
python .\scripts\verify_bundle.py
```

verify_bundle은 Python 표준 라이브러리만 사용해 파일별 SHA-256, 1,000장 이미지/라벨, 클래스·박스, 그룹 겹침, 기존 데이터 fingerprint를 확인합니다. 추론·학습을 실행하지 않습니다. Git clone에는 이미지/가중치가 없으므로 전체 ZIP 복원이 필요합니다.

## 환경

기록 환경: Windows, Python 3.11.9, RTX 5060 Laptop 8GB, NVIDIA driver 610.88, Torch 2.12.1+cu130, torchvision 0.27.1+cu130, Ultralytics 8.4.120. [전체 버전](environment/runtime.json)과 [패키지 고정 목록](environment/requirements-lock.txt)을 보존했습니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r .\environment\requirements-lock.txt
.\.venv\Scripts\python.exe .\scripts\train_reproduction.py --action check
```

설치 명령은 복원 안내이며 새 가상환경 설치 자체를 이번 보관 작업에서 시험하지 않았습니다. 기존 학습 환경에서 bundle 무결성, 초기 모델 로드, GPU 확인을 수행했습니다. OS/드라이버/라이브러리 버전 차이 때문에 재학습의 가중치나 점수가 비트 단위로 같다고 보장하지 않습니다.

## 사진 추론

```python
from ultralytics import YOLO
model = YOLO('boards8-1000images-yolo11s-best.pt')
model.predict(source='board_photo.jpg', imgsz=640, conf=0.25, save=True)
```

## 별도의 새 학습

기존 결과를 덮어쓰지 않는 외부 출력 폴더와 새 이름을 지정합니다. 이 명령을 이번 업로드 과정에서 실행하지 않았습니다.

```powershell
.\.venv\Scripts\python.exe .\scripts\train_reproduction.py --action train --output C:\BoardExperiments --name reproduction_01
```

이 도우미는 동봉한 공식 초기 가중치와 configs/train.json으로 새 학습을 시작합니다. 검증으로 best를 선택하고 종료하며 test를 자동 재실행하지 않습니다. 예전 test는 계속 개발 데이터로 취급하고, 최종 성능은 별도 미사용 실물/촬영 세션으로 평가해야 합니다.

## 보관 범위

- 현재 best, epoch 50 last, 공식 초기 가중치, 정확한 1,000장/YOLO 라벨.
- 실제 실행 코드·전체 args·설정·분할/클래스 매핑·선택 해시·학습 로그·50행 CSV·test 결과·곡선/혼동행렬.
- 이전 10에폭 모델·설정·지표와 실패/집계 정정 기록. 이전 파일럿의 원자료 1,620장 전체는 이 ZIP에 포함하지 않았습니다.
- code/의 과거 절대경로와 원래 data.yaml은 증거로 보존했습니다. scripts/는 다른 경로에서 복원하기 위한 별도 도우미입니다.
- 가상환경 전체, 인증정보, 라이브러리 계정 설정, Commons 원본, Roboflow 원 ZIP은 포함하지 않았습니다.
- 원래 로컬 ARTIFACT_MANIFEST는 당시 결과 묶음의 기록이고, 현재 Release의 기준은 BUNDLE_MANIFEST입니다.
