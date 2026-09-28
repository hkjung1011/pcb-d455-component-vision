# R04 중간 저장 · 2026-09-28

**사용자 요청으로 PC 종료 전 중단·저장했습니다.** 기준/개선 YOLO11s 학습은 각각20에폭이내·실제1,165회갱신으로완료됐고, 검증은22/40조합까지완료했습니다. 최종선택·일반/Pi시험은아직미완료입니다.

[중간 상태·재개 안내](R04_PAUSED/RESUME.md) · [저장 검증](R04_PAUSED/snapshot.json) · [현재까지 검증 지표](R04_PAUSED/partial_validation.csv) · [검토 반영](R04_PAUSED/reports/CLAUDE_REVIEW_RESPONSE.md)

이 중간 기록에는 코드·설정·학습상태·검증지표가 있습니다. **R04 가중치20개와 원본자료는 로컬PC에 저장되어 있으며, 이번 중간 Git 기록에는 업로드하지 않았습니다.** 재부팅 후 학습을 다시 하지 않고 남은 평가부터 이어갑니다. 아래 R03는 이전에 완료한 기록입니다.

---

# Raspberry Pi PCB · D455 인식 개발 기록

**2026-09-28 R03 스냅샷 · 비공개 연구 기록용 저장소**

Raspberry Pi는 촬영·인식 대상이다. RTX 5060 Laptop 8GB에서 공개자료로 네 모델을 학습하고, 학습·검증에 쓰지 않은 자료를 평가했다. **현재 세부 부품 인식은 실사용 수준에 미달하며, D455 실물 영상과 저항 포함 4종 부품 instance segmentation은 미검증/미완료다.**

## 현재 상태와 실제 지표

| 과제 | 시험 사진 / 정답 객체 | bbox mAP50–95 | mask mAP50–95 | 고정 threshold의 box Recall |
|---|---:|---:|---:|---:|
| 보드 검출 | 895 / 426 | 74.45% | 해당 없음 | 97.65% |
| 보드 외곽 분할 | 490 / 166 | 91.07% | 96.28% | 96.99% |
| 부품 검출 n / Pi3B | 2 / 189 | 22.00% | 해당 없음 | 43.39% |
| 부품 검출 s / Pi3B | 2 / 189 | 23.15% | 해당 없음 | 49.21% |

COCO maxDets=100 기준이며, 서로 다른 과제·시험셋의 숫자를 한 모델 순위로 비교하지 않는다. 기본 부품 모델은 **YOLO11s, native tile1024, confidence0.35**로 검증셋에서 선택했다. Pi3B 두 면의 target4 정답189개 중93개를 찾았고 IC는3/16개, 커넥터는0/12개였다(box IoU≥0.5). Pi는 한 보드그룹뿐이고 unknown67개가 제외돼 전체 BOM 인식률이나 새 설계 일반화를 뜻하지 않는다.

- 보드 검출 자료5,970장, 보드 mask 자료3,094장. 부품 자료는 원본47장·bbox5,896개다.
- 보드 모델2개는 각각5에폭, 부품 n/s는 각각20에폭. 실제 optimizer 갱신·클래스별 라벨 수·손실 가중치·체크포인트 SHA를 기록했다.
- 구조·라벨·학습·평가 그래프14종과 추가 공개 사진24장의 정성 예측 예시를 보존했다.
- 평가·선택·추론 CPU 검사28개와 공식 COCO 재계산의 일치를 확인했다. 이는 실사용 합격 판정이 아니다.
- D455 조회 당시 장치0대, 신규 실물 사진0장, target4 검수 mask0개다. 합성 축소·블러 결과는 카메라 실측이 아니다.
- 기존 micro-PCB의 Raspberry Pi 클래스 설명 A/H/I 오류를 G/H/M으로 수정했다. 과거 잘못된 매핑의 AP는 비교 기준에서 제외한다. 기존 `mcu-vision` 저장소는 변경하지 않았다.

## 기록 찾아보기

- [R03 전체 보고서와 그래프](R03/README.md)
- [현재 상태 JSON](CURRENT_STATUS.json), [시험 지표 CSV](metrics/r03_test_metrics.csv), [모델·에폭·가중치 SHA CSV](metrics/r03_models.csv)
- [모델 선택과 상세 registry](R03/model_registry.json), [전체 평가 집계](R03/reports/evaluation_suite/aggregate.json)
- [클로드 검토 반영표](R03/claude_review_application.md), [독립 평가 감사](R03/reports/final_evaluation_audit.json)
- [학습 미사용 공개 사진24장 예측 예시](R03/reports/qualitative_external24/README.md)
- [D455 촬영·라벨링 가이드](R03/annotation/target4_labeling_guide.md), [픽셀 예산 계산](R03/annotation/D455_PIXEL_BUDGET.md)
- [R01 계획](history/PCB_부품검출_세그멘테이션_계획_R01_Claude검토용.md), [R02 과거 보조 실험](history/PCB_D455_R02_개발결과.md), [진행 이력](CHANGELOG.md)

## 전체 모델 파일과 실행

**모델 가중치와 큰 데이터 manifest는 [비공개 Release](https://github.com/hkjung1011/pcb-d455-component-vision/releases/tag/r03-2026-09-28)의 `PCB_D455_R03.zip`에 포함돼 있다.** Git에서 보는 `R03/`는 보고서·코드·평가 기록을 위한 일부 파일 구성이다. 그대로 clone한 폴더만으로 가중치가 복원되지는 않는다.

Release에서 ZIP을 내려받아 새 폴더에 압축을 풀면 전체 `PCB_D455_R03` 폴더가 생긴다. 이 안의 [추론 사용법](R03/INFERENCE.md)을 따른다. 로컬 기존 Python 환경 또는 기록된 의존성이 필요하다. 기록된 절대 경로는 실행 당시의 증거이며 다른 PC에서는 유효하지 않을 수 있다.

원본 데이터 사진·다운로드 ZIP·Python 가상환경·인증 설정은 업로드하지 않았다. 학습/평가를 재현하려면 [원본 로컬 캐시 기록](R03/local_data_paths.json)과 출처에 따라 자료를 확보하고 경로를 설정해야 한다. 원본 ZIP 파일과 생략 목록·SHA256은 [아카이브 명세](ARCHIVE_MANIFEST.json)에 있다. `R03/SHA256SUMS.json`은 Release의 완전한 원본 패키지 기준이다.

## 다음 작업

실제 D455 원본30장으로 초점·거리·원본 부품 픽셀·조명을 확인한다. 이후 조건부200장 예산을 실물/설계/세션 기준으로 나누고 사람이 검수한 target4 mask를 확보한다. 카메라 미사용·공개 사진 평가 결과를 D455 실사용 성능으로 바꾸어 기록하지 않는다.

자료와 모델의 출처·라이선스는 [R03 보고서 출처](R03/README.md#출처), 각 metadata와 ATTRIBUTION.json에 보존한다. 비공개 저장소라는 이유로 제3자 자료에 새로운 재배포 권한을 부여하지 않는다.
