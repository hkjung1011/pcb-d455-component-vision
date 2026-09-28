# D455 원본 픽셀 확인

기본 캡처 요청은 RGB 1280×800, 30fps이며, 실제 SDK가 열거한 지원 profile에 정확히 있는 경우에만 사용한다. [제조사 비교표](https://www.realsenseai.com/compare-depth-cameras/)의 RGB 사양과 실제 연결 상태는 별개다. 이번 R03은 장치가 검색되지 않아 실제 profile·intrinsics·초점을 확인하지 못했다.

정면 평면의 근사식은 `부품 폭(px) ≈ fx(px) × 실물 폭(mm) / 거리(mm)`다. 실제 intrinsics의 fx와 실물 자로 확인해야 하며 기울어진 보드는 위치별 값이 달라진다.

아래는 **설명용 fx=640px 가정**이며 D455 실측값이나 권장 촬영거리가 아니다.

| 실물 부품 폭 | 거리200mm일 때 | 거리400mm일 때 |
|---|---:|---:|
| 0.5mm | 1.6px | 0.8px |
| 1.0mm | 3.2px | 1.6px |
| 2.0mm | 6.4px | 3.2px |

거리만 줄여도 실제 초점이 맞지 않으면 도움이 되지 않는다. 작은 부품의 모양이 원본에 몇 픽셀로 기록되는지 확인한 다음, 해당 크기의 종류 구분·경계 라벨링이 재현되는지 시험한다. depth의 최소 거리/정확도와 RGB 초점·가독성은 동일한 사양이 아니다.

원본 ROI와 타일은 모델 입력의 불필요한 축소를 줄인다. 디지털 확대나 1024 입력은 원본 2px에 없던 실제 윤곽 정보를 복원하지 않는다. mAP의 크기 구간은 원본 정답의 짧은 변 기준으로 기록하고, 선명도 수치 하나를 촬영 합격 기준으로 사용하지 않는다.

촬영 파일럿은 실제 초점이 확인되는 거리 범위에서 가까움/중간/멀어짐 조건을 선택하고 거리를 실측한다. 비슷한 구도를 유지한 채 조명·기울기·반사·노출 변화도 기록한다. 같은 보드를 연속 촬영한 30프레임을 독립 보드 30개로 세지 않는다.

```powershell
# 조회만 수행
& 'C:/Users/hkjun/Documents/RealSense-Local-Preview/.venv/Scripts/python.exe' -B scripts/capture_d455.py --output 'C:/results/d455_inventory01'

# 대상 보드가 화면에 배치된 뒤 직접 실행. 거리 숫자는 실제 측정값으로 추가한다.
& 'C:/Users/hkjun/Documents/RealSense-Local-Preview/.venv/Scripts/python.exe' -B scripts/capture_d455.py --capture --count 10 --interval 1 --board-id pi01 --design-id pi3b --session-id session01 --side top --lighting-id diffuse01 --output 'C:/results/d455_pi01_session01'
```

새로 추가한 촬영 메타데이터/품질 기록 코드는 CLI만 확인했다. 실제 D455 프레임에서의 동작과 프레임별 metadata 지원 여부는 미검증이다. 지원되지 않는 프레임 metadata는 null로 남긴다.
