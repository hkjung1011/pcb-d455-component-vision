# PCB 부품 원본 자료 확보·검증 기록

확인일: 2026-09-28. 이 폴더는 로컬 연구 작업 자료이며 원본 재배포 허가를 뜻하지 않는다.

## 실제 확보한 자료

[Kuo et al., WACV 2019 저자 페이지](https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection)가 직접 연결한 [Georgia Tech 원본 ZIP](https://ripl.cc.gatech.edu/data/pcb_wacv_2019.zip)을 로그인 없이 받았다. ZIP 298,093,226 bytes, SHA-256 `d24cf0e3688a5c7a4d827b2363a2a5d6e3c88c189203876a9e9ef3ab6f45b01a`. README에는 논문 인용 요청이 있고 명시적 라이선스는 없다. 원본·변환 라벨·원본이 포함된 QA 그림의 공개 재배포 권한은 확인되지 않았다.

실제 내려받은 파일을 세면 JPEG 47장, XML 47개, 보드 이름 기준 29그룹이다. XML 전체 18,201개 항목에는 텍스트와 비대상 부품이 섞여 있다. 홈페이지의 약 62,000개 수치와 현재 ZIP 계수가 다르므로 이 실험은 아래 실측값만 사용한다. 부품 bbox가 있으며 mask/polygon 정답은 없다. 원본 ZIP에 들어 있는 일부 PNG는 부품 인스턴스 마스크로 간주하지 않았다.

| class ID | 이번 이름 | 정확히 포함한 원본 type | 고유 annotation 수 |
|---:|---|---|---:|
| 0 | resistor | resistor | 2,111 |
| 1 | capacitor | capacitor, electrolytic capacitor | 2,778 |
| 2 | ic | ic | 386 |
| 3 | connector | connector | 621 |

`resistor network`, `resistor jumper`, `capacitor jumper`, `potentiometer`, `unknown` 등을 이름만 보고 자동으로 합치지 않았다. 텍스트 designator도 class를 바꾸는 근거로 사용하지 않았다. 이 ID 체계는 기존 smd_v1와 별개다.

## 고정 분할과 학습 입력

| 분할 | 보드 이름 그룹 | 원본 사진 | target4 bbox |
|---|---:|---:|---:|
| train | 18 | 27 | 3,484 |
| val | 5 | 7 | 875 |
| 일반 test | 5 | 11 | 1,348 |
| Raspberry Pi 3B test | 1 | 2 | 189 |

원본 이름의 `_Top`, `_Bottom`, 뒤 숫자를 제거한 보드 이름으로 묶고 seed 42로 고정했다. 같은 보드의 면·분할 사진이 다른 split에 들어가지 않는다. 실물 시리얼 번호를 검증한 것은 아니다. `Spartan6`와 `Spartan6Redux`는 비슷한 이름 때문에 따로 육안 점검했으며 실제 사진의 PCB 설계가 다르다.

RPI3B 2장은 모두 최종 Pi 시험용이다. train/val에서 제외했다. 사람 정답 189개는 저항 57, 커패시터 104, IC 16, 커넥터 12개다. 별도 `unknown` 67개가 있어 완전한 BOM 정답이나 모든 물리 부품의 정답이라는 주장을 할 수 없다. source target4 annotation 기준 AP로만 해석한다. 이 두 면이 여러 Raspberry Pi 모델이나 촬영 조건을 대표하지 않는다.

Train에만 원본 픽셀 1024 타일·overlap 0.2를 적용했다. 425타일이며 원본 부품 3,484개의 노출은 총 6,733회다. 노출 수는 저항 2,515, 커패시터 2,898, IC 523, 커넥터 797이다. 양수 너비·높이 1px 이상 조각을 모두 보존했다. 짧은 변 8px 미만 조각도 206개 유지했다. `instance_id`와 `visible_fraction`으로 고유 annotation 수와 반복 노출을 구분한다. 빈 타일도 유지했다. Val/test GT는 원본 크기이며 타일별 GT로 시험하지 않는다.

클래스별 추가 손실 가중치는 모두 1.0이고 역빈도 가중치나 클래스 oversampling은 적용하지 않았다. 생성된 타일은 균일하게 샘플링하므로 큰 보드와 겹침 영역이 더 자주 노출된다. 실제 손실 gain과 optimizer는 실행 설정/기록을 기준으로 한다.

## 준비 파일

- `component_records.json`: 원본 절대 경로·사진/XML 해시·class·raw VOC bbox·0-based half-open xyxy·그룹·분할·비대상 source objects.
- `component_tile_records.json`: 모든 train 타일의 원본 좌표, 원래 instance ID, 잘린 비율, 노출 bbox.
- `componentdetectdata.yaml`: train 타일, val 원본, 일반 test 원본.
- `componentdetect_pi_test.yaml`: Pi test를 가리키는 별도 설정. 모델 선택에 사용하지 않는다.
- `component_data_manifest.json`: 분할별 수·학습 타일 수·가중 정책.
- `preparation_verification.json`: 445개 YOLO 라벨 파일/9,145개 bbox 왕복 검사. 최대 좌표 오차 4.11e-7px, train 누락 0, 그룹 교차 0.
- `wacv_coordinate_review.png`: 원본 정답 상자와 Pi 중앙 ROI를 직접 확인한 QA 이미지. 예측 또는 mask가 아니다.
- `storage_budget.json`: ZIP·원본·파생 타일·hardlink를 중복 계수해 약 831 MB, 초기 1 GB 한도 이내.

재현 스크립트: `../scripts/prepare_component_data.py`. 사진 해상도 확인과 좌표 QA는 전문가가 모든 부품 종류를 검증했다는 뜻이 아니다.

## 다른 후보의 현재 상태

| 자료 | 실제 라벨 범위 | 접근·권리 확인 | 이번 사용 |
|---|---|---|---|
| [Pi3B+ Face Detection/Roboflow](https://universe.roboflow.com/face-detection-0dlvp/raspberry-pi-3-model-b-2tary) | GPIO/USB/HDMI/RAM 등 기능부품 14종 instance segmentation. 원본27장, 공개v1 61장 | 작성자 MIT 표기. Chrome 공개 UI에서 Download가 로그인/계정 생성 요구. 무인증 ZIP 미확보 | 미사용. 14종을 target4로 임의 매핑하지 않음 |
| [PCBVision](https://zenodo.org/records/10617721) | 53개 PCB RGB, IC·전해 커패시터·커넥터 semantic masks | CC BY 4.0, 이전 세션에 원본53장+General/Monoseg 확보 | 기존 source3 참고. 저항 없음, instance IDs 없음, Pi/D455 전용 아님 |
| [FPIC 원 배포처](https://physicaldb.ece.ufl.edu/index.php/fics-pcb-image-collection-fpic/) | SMD polygon/부품·텍스트 annotation | 원본 등록 요구 확인. 미러 라이선스를 원본 권한으로 가정하지 않음 | 미확보·미사용 |
| [PCB DSLR](https://zenodo.org/records/3886553) | IC bbox 및 보드 전체 mask | 원본 비상업 연구 조건. 부품4종 mask가 아님 | 추가 다운로드하지 않음 |
| [PCB-SAID 논문](https://openaccess.thecvf.com/content/ICCV2025W/VISION%2725/html/Mineo_PCB-SAID_A_Low-Cost_Camera-Based_Dataset_for_Few-Shot_SMD_Assembly_Inspection_ICCVW_2025_paper.html) | 부품/조립상태 bbox·polygon | 공개 무인증 파일·라이선스 미확인 | 미사용 |
| [Labellerr Pi4 사례](https://www.labellerr.com/blog/ai-electronics-detection-system/) | Pi4 주요 기능부품 segmentation 사례 | 원본 라벨 다운로드·라이선스 미확인 | 성능 또는 시험 정답으로 사용하지 않음 |

Pi3B+ Roboflow v1의 공개 화면은 train51/val8/test2, train 예제당3개 출력, 640×640 stretch, shear±15°, saturation±31%였다. 공개 mAP를 이번 실험 성능으로 인용하지 않는다. 다운로드 로그인 요구를 우회하지 않았다.

PCBVision 기존 경로: `../../assets_research/PCBVision_source/PCBDataset/RGB`. RGB53·General53·Monoseg53의 확보 해시는 기존 `pcbvision_acquisition_manifest.json`에 있다. Semantic connected regions를 물리 인스턴스/부품 수로 부르지 않는다. 이미 사용한 시험셋을 새 모델 선택에 재사용하지 않는다.
