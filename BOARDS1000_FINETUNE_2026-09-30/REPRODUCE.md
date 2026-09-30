# 미세조정 결과 복원

비공개 저장소 접근 권한이 있는 계정에서 두 Release를 다운로드합니다.

```powershell
gh release download boards1000-finetune-2026-09-30 --repo hkjung1011/pcb-d455-component-vision --pattern BOARDS1000_FINETUNE_FULL_SNAPSHOT.zip --pattern SHA256SUMS.txt --dir .\finetune_assets
gh release download boards1000-2026-09-30 --repo hkjung1011/pcb-d455-component-vision --pattern BOARDS1000_FULL_SNAPSHOT.zip --pattern SHA256SUMS.txt --dir .\baseline_assets
Get-FileHash .\finetune_assets\BOARDS1000_FINETUNE_FULL_SNAPSHOT.zip -Algorithm SHA256
Get-Content .\finetune_assets\SHA256SUMS.txt
Expand-Archive -LiteralPath .\finetune_assets\BOARDS1000_FINETUNE_FULL_SNAPSHOT.zip -DestinationPath .\restored_finetune
Expand-Archive -LiteralPath .\baseline_assets\BOARDS1000_FULL_SNAPSHOT.zip -DestinationPath .\restored_baseline
```

새 ZIP의 `BOARDS1000_FINETUNE_2026-09-30/local_package/BOARDS1000_FINETUNE20_2026-09-30/`에 후보·선택 가중치, 원래 README, 실제 설정·코드·검증·test 기록과 그래프가 있습니다. `ZIP_MANIFEST.json`은 ZIP 전체의 파일별 SHA-256이며 `github_documents/GIT_MANIFEST.json`은 가중치를 제외한 Git 문서만의 해시입니다. ZIP 바이트 자체는 외부 `SHA256SUMS.txt`와 비교합니다.

이미지·YOLO 라벨·공식 초기값은 기존 ZIP의 `BOARDS1000_2026-09-30/datasets/boards_v1/`와 `weights/`에 있습니다. 현재 데이터 fingerprint는 `66284009264318497a8ba16e8d532fc7f7bc0c6c96bdf1d4eae1c3d3525f5922`이며 split은 650 / 150 / 200입니다. 기존 [복원 도우미](https://github.com/hkjung1011/pcb-d455-component-vision/blob/main/BOARDS1000_2026-09-30/REPRODUCE.md)로 데이터 무결성을 확인할 수 있습니다.

새 optimizer와 스케줄로 기존 best에서 시작한 실험입니다. `resume=false`, lr0 `0.0001`, warmup `1.0`, mosaic `0.0`, patience `8`입니다. 원래 실행 코드는 당시 절대경로를 포함한 증거 스냅샷이며 경로 수정 없이 다른 PC에서 그대로 실행되지 않습니다. 새 학습에는 별도 출력 경로를 지정하고 기존 결과를 보존해야 합니다. 환경 설치나 재학습을 이 보관 도우미가 실행하지 않습니다.

선택 SHA-256: `fa0b8fb5fc11de748e9d8051b93fc57dde5c6cc34db6167045e3fee9e4912752`. 선택 판단은 `evidence/selection.json`의 검증 규칙을 따릅니다. test는 이미 재사용된 개발 자료이며 최종 성능 판단은 새로운 보드·촬영 세션 자료에서 해야 합니다.
