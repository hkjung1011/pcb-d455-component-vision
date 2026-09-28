# R03 개발 스냅샷 · 2026-09-28

공개자료 네 모델 학습·평가 기록과 이전 R02 보조 실험을 보존하는 비공개 Release다.

- `PCB_D455_R03.zip`: 네 모델 best/last, 초기 가중치, 코드·설정·라벨 집계·manifest·원본 평가 지표·그래프14종·정성24장 예시.
- `PCB_D455_R02.zip`: 이전 source3 semantic 보조 실험의 전체 로컬 산출물.
- 신규 D455 실물 촬영0장, target4 사람 mask0개. 부품 실사용 준비 상태는 미달이다.
- 기본 s 모델 Pi3B bbox AP50–95 23.15%, 고정 threshold recall49.21%, IC3/16·connector0/12. 한 보드그룹2장에 대한 제한 시험이다.
- 원본 데이터셋 사진/ZIP과 가상환경은 포함하지 않으며 학습 재현에는 별도 데이터 캐시가 필요하다.

SHA256:

```text
eaeec97c92058c1bbba4a94dece82087ec84c9a41b25a1a3cf8675107a15e2b1  PCB_D455_R03.zip
aaaa4b76e640dcd7f0d1780dace512ed15b612839643c13fdc0044d87cb890f2  PCB_D455_R02.zip
```
