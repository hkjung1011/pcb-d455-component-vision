# R03 Raspberry Pi 보드 자료 인계

대상은 **라즈베리파이 보드 전체**이다. R02의 저항·콘덴서·IC 등 부품 segmentation과 다른 라벨 범위다. 여기서는 모델 설치·학습·추론을 실행하지 않았다.

## 확보한 라벨 자료

`board_records.json`은 원저자 배포 IoTKITs COCO ZIP을 정규화한 목록이다. 이미지 경로는 절대경로이며 `objects` 안에 원래 클래스명, native category ID, bbox, polygon, 원 annotation ID를 보존했다. Raspberry Pi는 category 18–29를 `raspberry_pi_sbc` 후보로 매핑하고, 세부 모델 코드는 논문 Table 1을 근거로 별도로 적었다. 원본 A-/B- 문자열은 변경하지 않았다.

- 원저자: AnhTuấn Đỗ Nguyễn, [Mendeley Data v1](https://data.mendeley.com/datasets/x5thzmkxhy/1), DOI 10.17632/x5thzmkxhy.1.
- [논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC12149570/), [Roboflow 원 배포 프로젝트](https://universe.roboflow.com/donguyenanhtuan/32-fzmxa).
- 원본 archive: `iotkits_v1/32.v1i.coco.zip`, 141,166,460 bytes. SHA256 `5a22c88daafa8cc21b4fda46259cb9dbb13e2abff9ac6745b81d933c5d2b3bf5`.
- 라이선스: publisher 페이지와 archive의 `native/README.dataset.txt` 모두 CC BY 4.0 표시.
- 실제 archive: 3,108 이미지, 3,266 annotation, 32개 실제 라벨 클래스와 annotation 없는 category 0. 발표 설명 3,200과 실제 archive 수를 구분한다.
- RPi: 1,200 이미지, 1,206 개체. 그중 1,205 native polygon, 1 bbox-only. 사각형을 임의의 segmentation mask로 만들지 않았다.
- 전체 polygon 3,262 / bbox-only 4. 모든 이미지의 dimensions 및 bbox/polygon 경계 검사를 통과했다.
- 전체 exact SHA unique 3,095장. provider train/valid는 2,488/620이지만 평가용 분할로 승인한 것은 아니다.

인터넷 검색·Kaggle·Roboflow·저자 사진이 혼합되어 있다. 개별 파일 원천과 물리 보드 ID는 제공되지 않았다. `group_id`는 `.rf.` 이전 filename family를 보수적으로 묶은 값이다. 독립 촬영 그룹의 증명은 아니다. `per_file_origin`과 source-independence 한계를 각 record에 넣었다. 다른 보드 클래스는 Raspberry Pi 음성 후보이며, 미라벨 Raspberry Pi가 없는지 시각 검수해야 한다. 원본 native 폴더는 변경하지 않았다.

## 새 외부 도메인 사진

`commons_records.json`에는 **라벨 없는 24장**이 있다. `objects: null`이므로 빈 라벨/negative로 해석하면 안 된다. 개별 사진의 원본 URL, Commons page ID, photographer, license URL, credit, 해시, 16개 촬영자 그룹을 포함한다.

- 공식 Wikimedia API imageinfo/extmetadata로 출처를 읽고 공개 CC 사진만 수집했다.
- 최초 다운로드 44장 중 24장을 사진 내용과 근접중복 화면으로 선별했다. 검색군은 Pi4/Pi5/Zero/Pi3 각 6장이다. 세대명은 아직 사진별 하드웨어 검증 결과가 아니다.
- 원본 구도를 유지한 Commons 제공 thumbnail(너비 요청 1600px, 원본이 작으면 원본)을 받았다. 원본 publisher SHA1과 실제 내려받은 bytes SHA256을 구분한다.
- 기존 mcu-vision의 Commons 5개 page ID와 겹치지 않는다.
- IoTKITs 3,108장을 기준으로 64bit pHash 및 Commons의 8개 회전·반전 변형을 비교했다. distance <=8 또는 exact SHA overlap 후보는 제외했다.
- 원래 pHash 후보 2장, 상자 사진·가려진 보드·부품 crop·의심되는 stock뷰도 제외했다. 나머지 중 24장을 선택하고, 모든 선택/제외 사유를 `commons_holdout/selection_review.json`에 기록했다.
- pHash는 crop, 배경 제거, 다른 시점의 동일 물리 보드까지 완전히 판별하지 못한다. IoTKITs의 개별 인터넷 원천이 없으므로 완전한 물리·촬영 독립성은 아직 미검증이다.
- 최초 freeze 시에는 모델 예측이나 bbox가 없었다. 이후 별도 파일로 assistant bbox 초안을 만들었으며, 아래 설명처럼 정량 평가에는 사용할 수 없다.

`commons_holdout/frozen_holdout_candidates.json`은 라벨 초안/모델 예측 전에 선택한 이미지와 metadata hash를 고정한 기록이다. 원본 다운로드 44장 metadata는 `commons_records_all.json`에 남겼다. `selected_24_contactsheet.jpg`와 `nearest_source_contactsheet_01`~`04.jpg`는 검수용이다. 초기 시각 선별은 assistant가 수행했으며 사람이 독립 검증한 것은 아니다.

## Assistant가 만든 라벨 초안 — 정량 평가 사용 불가

후속 요청에 따라 `commons_draft_annotations.json`을 추가했다. 각 사진을 개별적으로 열어 본 후 기록한 **24장/28개 bbox 초안**이며 모델 예측을 보거나 사용하지 않았다. `reviewer: assistant_visual`, `annotation_status: draft_requires_human_review`, `human_review_performed: false`, `quantitative_evaluation_eligible: false`를 문서 및 각 record/object에 명시했다. 사람이 만든 독립 benchmark 정답이 아니다. mAP/IoU 산출용 정답으로 사용하면 안 된다.

좌표는 native pixel xyxy와 normalized xyxy 모두 제공한다. 보드 본체와 native port의 보이는 범위만 표시하고 손/상자/케이스/케이블/그림자와 별도 부속품을 제외한다. 가려진 영역을 임의로 복원하지 않는다. `commons_145417507`은 개별 원본 확인 결과 **4개 보드**(위·아래 partial 2개 포함)이며, 86471022는 2개이다. 패키징/인쇄된 보드 그림은 개체로 세지 않았다. 초점 흐림·microSD 경계·부속 보드·가림 등의 불확실한 사진에는 검토 사유를 기록했다.

`commons_holdout/draft_overlays/`와 `draft_annotation_contactsheet.jpg`에 DRAFT 표식이 있는 검토 이미지를 저장했다. `make_commons_draft_annotations.py`는 눈으로 기록한 좌표를 원본 크기로 환산하는 코드이며 detector를 호출하지 않는다. 원본 사진·라벨 없는 Commons metadata·초기 freeze hash는 그대로 유지했다.

## 접근하지 못하거나 제외한 후보

- [raspberryassembly/Raspberry Pi](https://universe.roboflow.com/raspberryassembly/raspberry-pi-qygyn): 제작자 페이지는 42장, raspberry/usb_wifi bbox, CC BY 4.0으로 표시한다. 이 세션의 비인증 HTTP는 403이므로 다운로드하지 않았다. 로그인 우회나 인증정보 사용은 하지 않았다.
- [OnTheBall 3B/3B+](https://universe.roboflow.com/ontheball/3b-3b): 100장, bbox, CC BY 4.0 표시. IoTKITs 원천과 연관 가능성이 있어 별도 독립 source로 승격하지 않았다.
- Hugging Face SyntheticFuture/real-raspberry-pi는 README만 있고 이미지 파일이 없었다. introvoyz041/Raspberrypi는 TensorFlow 실행자료로 대상 보드 이미지셋이 아니다.
- Commons 일부 파일은 HTTP 429를 반환했으며 재시도·우회하지 않고 제외했다. download_exclusions에 기록했다.

재현 스크립트는 `../source_acquisition.py`, `acquire_commons_holdout.py`, `select_commons_holdout.py`이다. 새로 획득한 bytes는 대략 185 MB로 초기 2 GB 예산 안에 있다. 원본 파일 재배포 시 데이터의 CC BY 및 사진별 CC BY/CC BY-SA/CC0 조건과 크레딧을 각각 보존해야 한다.
