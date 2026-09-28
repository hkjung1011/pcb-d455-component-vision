# R04 오프라인 실행

이 명령은 이미지 파일 또는 폴더를 읽어 4종 bbox를 그린 PNG와 원본좌표 JSON을 새 폴더에 저장한다. 카메라를 열지 않는다. 예측은 정확도 측정이 아니며 4종 부품 mask 출력은 없다.

```powershell
python scripts/infer_r04.py --input 'C:/images/board.jpg' --output 'C:/results/r04_01' --device 0
```

패키지 루트에서 실행한다. `--arm baseline` 또는 `--arm improved`로 비교 모델을 선택할 수 있다. 기본값은 검증셋으로 고정한 recommended 모델이다. checkpoint SHA, confidence, native/dual pipeline은 selected_models.json에서 읽는다. CPU는 `--device cpu`다.

```powershell
.\Run-Inference.ps1 -InputPath 'C:/images' -OutputPath 'C:/results/r04_02'
```

PowerShell wrapper는 현재 PC의 기존 Python을 기본 사용한다. 다른 PC에서는 `-PythonPath`를 지정한다. 필요한 버전은 environment.json에 있다. 별도 설치는 자동 수행하지 않는다.

native는1024px 원본 타일, dual은1024/2048px 원본 타일을 모델 입력1024px로 처리한다. overlap0.2, 모델 내부NMS0.6, 클래스별 병합NMS0.5, raw confidence0.001, 최대1000검출이다. 모델 반환 xyxy는 이미 원본 crop 좌표라서 crop offset만 더한다. 운영 threshold는 병합 후 적용한다.

재학습은 원자료를 확보한 후 prepare_r04.py의 source 경로를 새 작업폴더에 맞추고 두 arm을 구성해야 한다. 기록 폴더에 덮어쓰지 않는다. train_r04.py는 독립 데이터 감사·loss 검증과 hash를 확인한 뒤에만 실행한다. run_r04_evaluation.py는 두 학습이 완료되어야 실행된다. 기존 완료 holdout은 입력·출력 hash가 같은 경우 재사용하며 무조건 다시 추론하지 않는다.
