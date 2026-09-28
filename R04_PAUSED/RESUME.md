# PCB D455 R04 중간 저장 · 재부팅 후 재개

**2026-09-28: 사용자의 PC 종료 준비 요청으로 계산을 중단하고 저장했다. 지금 PC를 종료해도 된다.**

- 기준/개선 YOLO11s 학습 모두 완료. 각각 20번째 epoch의 90/112배치까지, 실제 optimizer 1,165회, draw 8,872회다.
- 2에폭 간격 체크포인트는 각10개, 합계20개이며 SHA256을 확인했다.
- 검증은 **22/40조합 완료**. baseline20개, improved2개 결과를 해시로 검증했다.
- 23번째 검증인 `improved__epoch_04__native__val`의 중단 기록은 별도 보존했다. 재개 시 이 검증부터 다시 계산한다.
- 최종 모델·confidence 선택은 아직 미확정이다. 일반/Pi 최종 개발 holdout과 합성 해상도 시험도 아직 수행하지 않았다. 현재 숫자를 최종 시험 mAP로 사용하면 안 된다.
- R04 평가·학습 GPU 프로세스를 종료했다. 사용자가 직접 PC를 종료하면 된다.

## 가장 간단한 재개

재부팅 후 **이 채팅에서 “PCB R04 이어서 진행해줘. 학습은 완료했고 검증 22/40부터 재개해”**라고 말하면 된다.

직접 남은 평가만 실행하려면 다음 명령을 사용한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\hkjun\Documents\Codex\2026-09-27\new-chat\outputs\Resume-PCB-R04.ps1"
```

이 명령은 완료한 검증22개를 해시 대조 후 재사용하고, 나머지18개 검증과 선택 고정 후 일반/Pi 개발 holdout4개를 계산한다. **재학습은 하지 않는다.** 에러가 나면 기존 결과를 지우지 말고 이 채팅에서 확인한다.

## 보존할 폴더

작업 전체: `C:\Users\hkjun\Documents\Codex\2026-09-27\new-chat`

특히 `work\r03`의 원본 자료 캐시, `work\r04`의 가중치·manifest·코드·평가 결과, 기존 Python 환경 `C:\Users\hkjun\Documents\mcu-vision\.venv-yolo11`을 유지한다. 모두 저장장치에 기록돼 있으며 메모리에만 남은 학습 상태에 의존하지 않는다. Temp의 첨부 이미지가 없어도 현재 학습 가중치와 평가 재개에는 영향이 없다.

가중치: `work\r04\runs\baseline\candidates` 및 `work\r04\runs\improved\candidates`.

완료 평가: `work\r04\reports\evaluation_suite`. 중단된 1개 작업: `work\r04\paused_evaluations`.

[중간 저장 검증 JSON](PCB_D455_R04_중간저장.json) · [재개 PowerShell](Resume-PCB-R04.ps1)

## 후속 작업 순서

1. `run_r04_evaluation.py --root work/r04`로 남은 주 평가 완료. evaluator/runner/protocol/후보 가중치를 수정하지 않는다.
2. `evaluate_resolution_stress.py --run`으로 사전 동결한 별도 합성 해상도4조건 작업. 저해상도 보드 crop의 민감도이며 D455 실측/상하한이 아니다.
3. `audit_r04_results.py`로 저장 COCO·매칭·선택·학습 예산을 독립 재계산.
4. `prepare_inference_selection.py`, `check_inference_equivalence.py`로 고정 모델의 오프라인 실행을 확인. 원본 사진 preview는 `work/r04_runtime_smoke`에만 두며 배포하지 않는다.
5. `analyze_r04_errors.py`, `make_r04_figures.py` 실행 후 최종9개 그림을 시각 검토.
6. `package_r04.py`로 원본 사진을 제외한 최종 패키지 생성. 현재는 미완료 결과이므로 실행하지 않는다.
7. `prepare_github_r04.py`로 최종 R04 Git 기록을 준비하고 비공개 Release를 올린 뒤 `verify_github_r04.py`로 원격 SHA 검증.

R03 Release와 원본 시험 GT는 보존한다. 현재 R04는 bbox 검출이며 D455 실물 검증과 저항 포함4종 instance mask는 미완료다. 이후 실물30장 파일럿·별도200장 수집은 계획량이다.
