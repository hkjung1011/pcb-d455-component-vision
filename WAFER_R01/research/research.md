# 웨이퍼 표면 결함 데이터 조사 · 2026-09-29

사용 목적은 **카메라 영상에서 웨이퍼의 표면 결함 위치를 검출**하는 것이다. 전기검사 결과를 색으로 표시한 wafer bin map을 실제 표면 영상으로 취급하지 않는다.

## 실행 가능한 자료와 선택

이번 로컬 연구용 파일럿에는 저자가 공개한 **Wafer-Datas의 표면 현미경 사진과 bbox 라벨**을 확보했다. 2,132장, bbox 3,693개를 읽었고 파일 4,264개의 Git blob SHA가 고정 커밋과 일치한다. 공개된 전체 데이터에는 미리 생성된 증강본이 포함되므로, 이름에 변형 접미사가 없는 원본 후보만 골랐다. 원본 후보가 전부 서로 다른 웨이퍼라는 뜻은 아니다. 추가 검사에서 완전 중복 49그룹 모두 라벨이 서로 달랐으므로 **98장을 격리하고 2,034장을 학습 분할 대상으로 삼는다.** 최종 라벨·분할 수는 `work/wafer_r01/reports/data_audit.json` 기준이다.

원논문 §3.1은 촬영 방법을 `electron microscopy and an attached camera`라고 설명한다. 따라서 이 자료를 일반 RGB/D455 광학 영상으로 단정하지 않는다. **현미경 영상에서 표면 결함을 찾는 알고리즘 파일럿**이며, D455 현장 성능은 별도 촬영으로 검증해야 한다. bin map은 아니다.

| 후보 | 영상과 라벨 | 규모·분할 | 접근·라이선스 확인 | 이번 판단 |
|---|---|---|---|---|
| [Wafer-Datas](https://github.com/ztao3243/Wafer-Datas) | 현미경 표면 사진, Pascal VOC bbox; 6종 | 저장소 이미지 6,228장 중 원본 후보 2,132장 확보. 저자 제공 split 파일 없음 | [원논문 출판사](https://fcds.cs.put.poznan.pl/FCDS/ArticleDetails.aspx?articleId=521)가 데이터 공개 저장소를 직접 지정. 인증 없이 파일 수신 완료. 데이터 라이선스 문구 없음 | 로컬 연구 파일럿에 사용. raw 이미지·원본 라벨은 GitHub에 재배포하지 않음. 실제 D455 실사용 검증 아님 |
| [Roboflow Wafer Defect v2](https://universe.roboflow.com/wafer-irhuv/wafer-defect-rv1vx/dataset/2) | 색상 패턴 웨이퍼 현미경 사진, 7종 instance polygons | 4,532장. train 3,173 / val 907 / test 452. 공개 페이지상 augmentation 없음 | CC BY 4.0 표시. 실제 browser export는 로그인 요구, 비인증 API는 HTTP401. bulk 미수신 | **추후 instance segmentation 우선 후보**. 계정으로 정식 export 확보 후 변환·누수 검사 |
| [INU Semiconductor Scratch Segmentation](https://bridge.inu.ac.kr/dataset/semiconductor_scratch_segmentation_dataset) | 반도체 표면 scratch pixel segmentation으로 설명 | 페이지에 장수·클래스별 수량 없음 | ZIP 링크 존재, 페이지에 '라이센스를 제공하지 않음' 표시 | bbox 후 mask 단계 대안. 라이선스·촬영 방식·파일 실체 미확인, 이번에는 미수신 |
| [Carinthia v2](https://zenodo.org/records/10715190) | SEM 영상의 6종 이미지 분류 | 4,591장, data.zip 133.8MB | Infineon/KAI 연구자가 Zenodo에 공개. 이 조사에서 라이선스 상세 추출은 미확정 | 카메라 표면 검출 목적과 차이 큼. 이번 훈련에서 제외 |
| [WDD / DAS-YOLOv13](https://pmc.ncbi.nlm.nih.gov/articles/PMC12987287/) | 산업 카메라 TD-4KH로 촬영한 표면 bbox 검출, 6종 | 논문상 5,605장, 640×640; train 4,484 / val 561 / test 560 | Data Availability는 교신저자에게 요청. 공개 다운로드 확인 안 됨 | 광학 검사 목적에 맞지만 즉시 확보 불가 |

### 직접 확보한 Wafer-Datas

- 저장소 커밋: `17dbc7e19f4e354f9b2b436af66e5abdaf2ea4b1`
- 파일: `work/wafer_r01/raw/VOC2007/JPEGImages` / `Annotations`
- 원본 후보 2,132장 모두 1280×720. 총 279,385,715 bytes.
- 라벨 3,693개. `scratch` 1,054, `edge_bite` 306, `stains_enbedded` 1,403, `gray_line` 154, `open` 237, `Short_circuit` 535, `short_cricuit` 4.
- source의 두 short-circuit 표기는 원본을 보존하고 학습용에서 하나로 명시적 매핑할 수 있다. 다른 철자도 source/raw 이름을 기록한다.
- 이미지/라벨 불일치 0, 좌표 범위 오류 0. VOC의 1-based inclusive 가정에 모두 적합하며, 변환 시 `(xmin-1, ymin-1, xmax, ymax)`를 사용한다. 이 좌표 관례는 자료를 검사해 정한 것이며 저자 문서에 명시된 계약은 아니다.
- 바이트 및 디코딩 픽셀 완전 중복 49그룹. pHash Hamming≤4 후보 1,586쌍. pHash 유사성만으로 같은 웨이퍼임을 단정할 수 없다.
- 동일 픽셀 49그룹 전부 canonical class·정렬 bbox signature가 불일치한다. 예를 들어 `scratch_102`와 `waferImg_2144`는 같은 사진인데 전자에 있는 오염 객체 하나가 후자에는 없고 박스 크기도 다르다. 두 원본을 보존하되 98장 모두 split 전에 격리한다. 나머지 사진에 라벨 누락이 없다는 보장은 아니며, 남은 원본 전체의 전문가 검수는 수행하지 않았다.
- 물리 웨이퍼 ID·촬영 세션 ID·카메라 보정·실제 결함 치수 없음. `waferImg2_*` 같은 계열과 완전/근접 중복을 분할 전에 묶되, 보지 않은 물리 웨이퍼의 성능이 검증됐다고 표현하지 않는다.
- 원논문은 원사진 2,278장·증강 후 4,330장을 설명한다. 현재 저장소 6,228장 및 이번 원본 후보 2,132장과 다르다. 원논문의 결과나 분할을 재현했다고 주장하지 않는다.
- mask 없음. bbox를 임의로 채워서 정답 instance mask라고 만들지 않는다.

정확한 선택 정규식:

```text
^(edge_bite\d+|gray_line_\d+|open_\d+|scratch_\d+|waferImg_\d+|waferImg\d+_\d+)$
```

증강명에는 blur, crop, noise 외에 brighter, darker, rot, flip의 오기 `filp`도 있어, 단순히 세 접미사만 제거하면 증강이 남는다.

**사용 조건:** 저자가 연구 데이터로 공개한 출처는 확인했다. 별도 데이터 라이선스는 미명시이므로 '오픈소스/상업 사용 자유/재배포 허용'이라고 표현하지 않는다. 논문의 CC BY-NC-ND 표시를 데이터 라이선스로 옮겨 적지 않는다. 이번에 받은 원본과 원본 라벨은 로컬에 보관한다.

### Roboflow 라벨 실제 확인

공개 [샘플 13534_image2.jpg](https://universe.roboflow.com/wafer-irhuv/wafer-defect-rv1vx/images/01hqBhJ347Q0pgV7nkge)를 브라우저로 확인했다. 742×576 색상 표면 사진에 SCRATCH 인스턴스 2개가 폴리곤으로 표시된다. Raw Data에서 원본 COCO annotation임을 확인했다.

- 클래스: BLOCK ETCH, COATING BAD, PARTICLE, PIQ PARTICLE, PO CONTAMINATION, SCRATCH, SEZ BURNT.
- 일부 원본 점이 y=-5.3 또는 y=591.2로 이미지 경계 밖에 있다.
- 원본 annotation의 분리된 네 polygon이 플랫폼 변환본에서 한 연결 polygon으로 바뀐 예가 있다.
- 추후 export 시 원본 COCO와 변환된 YOLO mask의 경계·다중 영역 처리 차이를 검사한다.
- 페이지의 72.5% mAP50은 게시자의 모델 값이며 우리 모델의 측정값이 아니다.
- 프로젝트 설명은 비어 있어 원 촬영기관·물리 웨이퍼 ID·분할 독립성을 확인할 수 없다.

## 이번 파일럿과 D455 적용 경계

현재 환경에서 가벼운 YOLO11n bbox 검출을 10에폭 이내로 시작할 수 있는 파일은 확보했다. 최종 학습 설정·split·실제 step·평가는 `work/wafer_r01`의 실행 기록을 기준으로 한다. 이번 조사 담당자는 GPU 학습을 실행하지 않았다.

실제 실행 프로토콜은 bbox mAP50 및 mAP50–95와 추론 시간을 기록하고, validation AP로 best 가중치만 선택한다. confidence를 validation에서 최적화하지 않았다. 별도 보조 분석은 test 전에 정한 confidence 0.25 / IoU 0.5에서 클래스별 P/R·TP/FP/FN을 계산한다. 이 값은 현장에 맞춰 보정된 운용점이 아니다. holdout은 가중치 선택 이후 한 번 평가한다. split은 중복/파일 계열을 묶어 고정하며, 처음 보지 않은 물리 웨이퍼 및 실제 D455라는 두 가지 검증은 별도로 남는다.

D455로는 먼저 고정 거리·확산 조명에서 표면의 큰 스크래치·오염·에지 파손이 원본 RGB 픽셀에서 보이는지 확인한다. 실제 최소 결함 크기와 보드/웨이퍼 점유 픽셀을 측정한 뒤, 여러 각도 프레임이 같은 물체의 train/test를 넘나들지 않게 물체·세션 단위로 나눈다. 현미경 자료에 보이는 세부 결함이 D455에도 보인다고 가정하지 않는다.

## 목적이 다른 자료 / 탈락 이유

- WM-811K 및 MixedWM38은 die pass/fail 배열로 구성된 wafer bin map이다. 카메라 표면 결함 학습으로 대체하지 않았고 수신·학습하지 않았다.
- [HiHiAllen Chip-surface-defect](https://github.com/HiHiAllen/Chip-surface-defect-dataset): 실제 정상 2,270 / 실제 결함 1,241 / 합성 7,250으로 설명. LFS quota 문제로 Drive 링크를 제공하나 데이터 라이선스·원래 웨이퍼 광학 조건 불명확. 이번 미수신.
- [MCDA-DETR](https://github.com/HQU-ADL/MCDA-DETR): 공식 README가 Mini-LED dataset coming soon. 코드가 존재한다고 dataset이 이미 공개됐다고 처리하지 않음.
- 세그멘테이션 판매성 재게시 저장소의 이름·README만으로 원데이터 또는 라이선스를 신뢰하지 않음.

## 증거 파일

- `download_manifest.json`: 수신한 4,264파일, source Git SHA, SHA-256, 원본 후보 선택 정책.
- `wafer_datas_tree.json`: 고정 source의 전체 트리 목록.
- `wafer_datas_source_audit.json`: 전체 2,132개 이미지/라벨 및 중복 집계. source 라벨·메타정보를 포함하므로 로컬 전용.
- `duplicate_annotation_audit.json`: 동일 픽셀 49그룹의 불일치 라벨 비교. source 라벨이 포함되므로 로컬 전용.
- `training_script_review.md`: 설치된 Ultralytics와 훈련/선택 callback의 정적 호환성 검사. GPU 학습 성공 증거와 구분.
- `download_wafer_datas.py`, `audit_wafer_datas_source.py`: 재현 코드. 외부 저장소의 실행 코드는 사용하지 않음.
- `research.json`: 후보·출처·접근상태의 간단한 구조화 기록.

### 원출처

1. [Wafer-Datas 연구 공개 선언](https://fcds.cs.put.poznan.pl/FCDS/ArticleDetails.aspx?articleId=521)
2. [CC-De-YOLO 원논문 PDF · §3.1](https://reference-global.com/pdf/10.2478/fcds-2024-0014)
3. [Wafer-Datas 저장소](https://github.com/ztao3243/Wafer-Datas)
4. [Roboflow Wafer Defect v2](https://universe.roboflow.com/wafer-irhuv/wafer-defect-rv1vx/dataset/2)
5. [INU Scratch 데이터](https://bridge.inu.ac.kr/dataset/semiconductor_scratch_segmentation_dataset)
6. [Carinthia v2](https://zenodo.org/records/10715190)
7. [WDD 원논문과 제공 조건](https://pmc.ncbi.nlm.nih.gov/articles/PMC12987287/)
