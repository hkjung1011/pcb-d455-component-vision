# R03 오프라인 추론

`scripts/infer_rpi.py`는 이미지 파일 또는 폴더를 읽고 각 이미지의 `annotated.png`, `prediction.json`, 선택적 보드 마스크 PNG를 저장합니다. 카메라는 열지 않습니다. 기존 A/H/I 라벨 오류가 확인된 역사적 YOLO11m 체크포인트는 SHA256으로 거부합니다.

보드는 `raspberry_pi_sbc` 검출이며, 선택적 segmentation 모델은 **보드 전체 윤곽**입니다. 부품 n/s 모델의 출력은 `resistor/capacitor/ic/connector` **bounding box**입니다. 보드 마스크를 부품 마스크로 사용하거나 표시하지 않습니다.

## 검증 후 선택 파일 사용

릴리스 루트의 `selected_models.json`이 있으면 자동으로 읽습니다. 기본 장치는 CPU이며 `--device 0`으로 GPU를 지정할 수 있습니다. 파일 또는 폴더 모두 `--input`으로 지정합니다.

```powershell
python scripts/infer_rpi.py --input 'C:/images/board.jpg' --output 'C:/results/run01' --device 0
```

아직 선택 파일이 없다면 체크포인트와 confidence를 각각 명시해야 합니다. 다음 경로와 confidence 숫자는 CLI 작성 예시이며 검증에서 선택된 값으로 바꿔야 합니다.

```powershell
python scripts/infer_rpi.py --input 'C:/images' --output 'C:/results/run02' --board-weights 'C:/models/board-best.pt' --board-conf 0.50 --parts-weights 'C:/models/parts-best.pt' --parts-conf 0.40 --parts-pipeline native-tiles --parts-tile-size 1024 --parts-imgsz 1024 --device 0
```

선택적 보드 윤곽은 `--board-seg-weights PATH --board-seg-conf VALUE`로 추가합니다. 선택 파일의 보드 윤곽 역할을 생략하려면 `--no-board-mask`를 사용합니다. 출력 폴더는 비어 있어야 합니다.

## 타일·좌표·보드 ROI

선택 파일이 없을 때 부품 기본값은 원본 전체 영상을 1024픽셀 타일, overlap 0.2로 자르는 방식입니다. **전체 영상을 먼저 축소하지 않습니다.** 각 원본 타일은 모델의 `imgsz`에 맞는 letterbox/resize를 거치며, 모델이 되돌려 준 타일 좌표에 원본 타일 위치를 더합니다. 이 전처리와 crop bounds를 JSON에 기록합니다. `whole`은 모델이 전체 영상을 resize할 수 있는 별도 파이프라인입니다. 선택 파일에 파이프라인이 있으면 그 값을 사용합니다.

타일 사이 중복 검출은 평가 스크립트와 같은 torchvision 클래스별 box NMS(IoU 0.5)로 제거합니다. 모델 내부 prediction NMS는 IoU 0.6입니다. 평가와 동일하게 confidence 0.001부터 예측·NMS를 수행한 뒤, 선택된 운영 confidence 이상인 결과만 남깁니다. 기본 cap은 모델/타일과 최종 타일 병합 모두1000개이며, `--max-det` 변경은 별도 override로 기록합니다. 모든 출력 좌표는 디코딩된 원본 이미지의 픽셀 좌표입니다.

`--board-roi`는 검출한 보드 box에 각 방향 10% 여백을 더한 범위에서만 부품을 찾습니다. 이 조합은 `UNBENCHMARKED_BOARD_TO_PARTS_ROI_CASCADE`로 표시합니다. 보드가 없으면 부품을 실행하지 않고 `SKIPPED_NO_BOARD_ROI`를 남깁니다. 기본 전체 프레임 부품 추론은 보드 검출 결과로 제한하지 않습니다.

예측이 없으면 `NO_DETECTIONS_AT_SELECTED_CONFIDENCE`로 기록합니다. 이는 해당 threshold에서 남은 예측이 없다는 의미이며, 실제 물체의 부재를 보증하지 않습니다. confidence도 보정된 확률로 주장하지 않습니다.

보드 mask PNG는 원본 해상도의 이진 마스크를 필요한 영역만 자른 파일입니다. JSON의 `origin_xy`와 `bounds_xyxy`로 원본 프레임에 배치합니다. 구멍과 분리된 영역은 PNG가 보존하며, JSON exterior contour는 표시용입니다. 파일 안의 `mask_scope`는 보드 전체라는 사실을 명시합니다.

## 선택 registry 계약

```json
{
  "schema": "r03-inference-selection-v1",
  "board_detector": {
    "checkpoint": "models/board-best.pt",
    "sha256": "ACTUAL_CHECKPOINT_SHA256",
    "confidence": 0.5,
    "imgsz": 1024
  },
  "parts": {
    "checkpoint": "models/parts-best.pt",
    "sha256": "ACTUAL_CHECKPOINT_SHA256",
    "confidence": 0.4,
    "imgsz": 1024,
    "pipeline": "native-tiles",
    "tile_size": 1024,
    "overlap": 0.2
  }
}
```

위 수치는 예시입니다. 실제 선택 JSON은 validation 결과로 결정합니다. `board_segmenter`는 보드 검출기와 같은 필드를 가진 선택적 역할입니다. 상대 checkpoint 경로는 registry 파일 폴더 기준입니다. CLI override는 출력에 명시하며 registry와 동일한 검증 결과라고 자동 주장하지 않습니다.

`python scripts/infer_rpi.py --help`는 모델·torch를 불러오지 않습니다. 실제 실행에는 해당 환경의 Ultralytics, torch/torchvision, NumPy, OpenCV가 필요합니다. geometry 모듈은 우선 릴리스 `src/pcb_components`에서 찾고 개발 폴더에서는 `../pcb_components/src`를 사용합니다.

합성 좌표·CPU NMS 테스트 9개를 통과했고, 실제 Pi3B 원본 한 장의 부품 16개 class/bbox/score가 저장된 평가 예측과 정확히 일치했습니다. 근거는 `reports/inference_equivalence.json`입니다. 추가 Commons24장에는 예측 예시를 생성했으며 정답 대조 점수를 붙이지 않았습니다. 이 검증은 D455 실기 검증을 뜻하지 않습니다.

현재 PC에서는 릴리스 폴더의 `Run-Inference.ps1 -InputPath 'C:/images/board.jpg' -OutputPath 'C:/results/r03_run01'`로 기존 학습 Python과 GPU0을 사용할 수 있습니다. 다른 환경에서는 `-PythonPath`와 `-Device`를 지정합니다. 라즈베리파이는 촬영 대상이며 이 명령이 라즈베리파이 보드에서 실행된다는 뜻은 아닙니다.
