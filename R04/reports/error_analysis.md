# R04 저장 예측의 객체별 오류 분석

새 추론 없이 저장된 예측과 원본 GT를 비교했다. 두 arm의 기존 전역 confidence를 각각 그대로 사용했다. 독립적인 score순·같은 클래스·IoU≥0.5·1:1 매칭 결과가 원 평가 함수 및 저장된 TP/FP/FN과 모두 일치한 뒤 분석을 작성했다.

모델 선택, threshold, 원본 GT는 변경하지 않았다. R03에서 이미 본 개발 holdout을 재사용한 진단이며 새로운 최종시험 또는 D455 실측이 아니다.

## 두 모델이 같은 객체를 찾았는가

| 자료 | 원본 GT | 둘 다 찾음 | 개선안만 | 기준안만 | 둘 다 놓침 |
|---|---:|---:|---:|---:|---:|
| general_test | 1348 | 641 | 201 | 52 | 454 |
| pi_test | 189 | 99 | 47 | 1 | 42 |

분모는 고유 원본 GT이며 오검출 수를 이 표의 분모에 섞지 않았다. 개선안만 찾은 수와 기준안만 찾은 수는 바뀐 학습 구성·선택 checkpoint·추론 방식·confidence를 함께 반영하므로 특정 변경 하나의 효과로 해석하지 않는다.

### general_test 클래스별

고정 confidence: 기준안 0.75, 개선안 0.65.

| 클래스 | 전체 GT | 둘 다 찾음 | 개선안만 | 기준안만 | 둘 다 놓침 |
|---|---:|---:|---:|---:|---:|
| resistor | 443 | 202 | 56 | 20 | 165 |
| capacitor | 636 | 369 | 103 | 14 | 150 |
| ic | 93 | 38 | 16 | 1 | 38 |
| connector | 176 | 32 | 26 | 17 | 101 |

### pi_test 클래스별

고정 confidence: 기준안 0.75, 개선안 0.65.

| 클래스 | 전체 GT | 둘 다 찾음 | 개선안만 | 기준안만 | 둘 다 놓침 |
|---|---:|---:|---:|---:|---:|
| resistor | 57 | 37 | 9 | 0 | 11 |
| capacitor | 104 | 54 | 34 | 0 | 16 |
| ic | 16 | 8 | 1 | 1 | 6 |
| connector | 12 | 0 | 3 | 0 | 9 |

## Pi3B 혼동 사례 후보

해당 GT를 놓쳤고, 다른 클래스의 IoU≥0.5 상자가 기존 운영 confidence 이상인 경우만 추렸다. 아래 사례는 서로 배타적인 오류 원인 분류가 아니다.

| 모델 | 원본 객체 | GT 클래스 | 겹친 다른 클래스 후보 |
|---|---|---|---|
| baseline | RPI3B_Bottom:00118 · connector J9 | connector | ic · score 0.9375 / IoU 0.933 |
| improved | RPI3B_Bottom:00118 · connector J9 | connector | ic · score 0.8858 / IoU 0.950 |

## Pi3B IC·커넥터의 원본 객체별 후보

같은 클래스 후보는 최고 IoU 및 IoU≥0.5 안에서의 최고 score를 따로 기록했다. 다른 클래스 후보는 양의 overlap 중 IoU가 가장 큰 것을 표시한다. 이 표의 후보는 이미 crop/global NMS와 최대 1,000개 제한을 통과한 저장 후보이며, 네트워크의 모든 출력은 아니다. raw confidence floor는 0.001이다.

score는 보정된 정답 확률이 아니다. 높은 점수의 다른 클래스 후보가 겹친다는 것은 클래스 혼동의 사례 후보이며, 원본 GT의 의미가 항상 옳다고 입증하지 않는다. 기존 미실장/unknown 정답은 임의로 바꾸지 않았다.

### RPI3B_Bottom:00069 · ic U18 · 둘 다 놓침

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | ic · score 0.0013 / IoU 0.540 | ic · score 0.0013 / IoU 0.540 | capacitor · score 0.0033 / IoU 0.535 |
| improved | 놓침 | ic · score 0.1660 / IoU 0.777 | ic · score 0.1660 / IoU 0.777 | capacitor · score 0.0629 / IoU 0.841 |

### RPI3B_Bottom:00071 · ic U15 · 둘 다 찾음

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 찾음 | ic · score 0.9514 / IoU 0.916 | ic · score 0.9514 / IoU 0.916 | capacitor · score 0.0029 / IoU 0.971 |
| improved | 찾음 | ic · score 0.9288 / IoU 0.940 | ic · score 0.9288 / IoU 0.940 | resistor · score 0.0017 / IoU 0.035 |

### RPI3B_Bottom:00106 · ic U14 · 둘 다 찾음

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 찾음 | ic · score 0.8158 / IoU 0.846 | ic · score 0.8158 / IoU 0.846 | connector · score 0.0186 / IoU 0.008 |
| improved | 찾음 | ic · score 0.9137 / IoU 0.820 | ic · score 0.9137 / IoU 0.820 | resistor · score 0.1364 / IoU 0.702 |

### RPI3B_Bottom:00118 · connector J9 · 둘 다 놓침

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | connector · score 0.1428 / IoU 0.132 | 저장 후보 없음 | ic · score 0.9375 / IoU 0.933 |
| improved | 놓침 | connector · score 0.0030 / IoU 0.930 | connector · score 0.0030 / IoU 0.930 | ic · score 0.8858 / IoU 0.950 |

J9는 이전 검토에서 microSD 소켓으로 지목한 위치다. 이번 스크립트는 사진을 읽지 않아 subtype을 새로 검증한 것은 아니다.

### RPI3B_Bottom:00130 · ic U19 · 둘 다 놓침

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | ic · score 0.2116 / IoU 0.767 | ic · score 0.2116 / IoU 0.767 | connector · score 0.0235 / IoU 0.015 |
| improved | 놓침 | ic · score 0.3840 / IoU 0.867 | ic · score 0.3840 / IoU 0.867 | 저장 후보 없음 |

### RPI3B_Bottom:00371 · connector unknown · 둘 다 놓침

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | connector · score 0.0039 / IoU 0.832 | connector · score 0.0039 / IoU 0.832 | 저장 후보 없음 |
| improved | 놓침 | connector · score 0.2884 / IoU 0.842 | connector · score 0.2884 / IoU 0.842 | ic · score 0.0057 / IoU 0.005 |

### RPI3B_Top:00012 · ic U20 · 둘 다 찾음

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 찾음 | ic · score 0.9468 / IoU 0.916 | ic · score 0.9468 / IoU 0.916 | 저장 후보 없음 |
| improved | 찾음 | ic · score 0.9130 / IoU 0.926 | ic · score 0.9130 / IoU 0.926 | capacitor · score 0.0025 / IoU 0.851 |

### RPI3B_Top:00023 · connector J1 · 둘 다 놓침

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | connector · score 0.0377 / IoU 0.353 | 저장 후보 없음 | capacitor · score 0.0331 / IoU 0.822 |
| improved | 놓침 | connector · score 0.0033 / IoU 0.637 | connector · score 0.0033 / IoU 0.637 | capacitor · score 0.0018 / IoU 0.626 |

### RPI3B_Top:00028 · ic U4 · 둘 다 놓침

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | ic · score 0.0085 / IoU 0.560 | ic · score 0.0085 / IoU 0.560 | 저장 후보 없음 |
| improved | 놓침 | ic · score 0.1718 / IoU 0.574 | ic · score 0.1718 / IoU 0.574 | 저장 후보 없음 |

### RPI3B_Top:00030 · ic U9 · 기준안만 찾음

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 찾음 | ic · score 0.7885 / IoU 0.787 | ic · score 0.7885 / IoU 0.787 | 저장 후보 없음 |
| improved | 놓침 | ic · score 0.4162 / IoU 0.567 | ic · score 0.4162 / IoU 0.567 | capacitor · score 0.4415 / IoU 0.031 |

### RPI3B_Top:00035 · connector J6 · 둘 다 놓침

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | connector · score 0.4721 / IoU 0.899 | connector · score 0.4721 / IoU 0.899 | capacitor · score 0.0660 / IoU 0.858 |
| improved | 놓침 | connector · score 0.2651 / IoU 0.816 | connector · score 0.2651 / IoU 0.816 | capacitor · score 0.0032 / IoU 0.915 |

### RPI3B_Top:00039 · ic U16 · 둘 다 놓침

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | ic · score 0.0250 / IoU 0.882 | ic · score 0.0250 / IoU 0.882 | capacitor · score 0.5907 / IoU 0.008 |
| improved | 놓침 | ic · score 0.4151 / IoU 0.833 | ic · score 0.4151 / IoU 0.833 | capacitor · score 0.8797 / IoU 0.002 |

### RPI3B_Top:00043 · ic U8 · 둘 다 놓침

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | ic · score 0.0098 / IoU 0.017 | 저장 후보 없음 | 저장 후보 없음 |
| improved | 놓침 | ic · score 0.0037 / IoU 0.002 | 저장 후보 없음 | 저장 후보 없음 |

### RPI3B_Top:00050 · ic U3 · 둘 다 찾음

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 찾음 | ic · score 0.9484 / IoU 0.659 | ic · score 0.9484 / IoU 0.659 | 저장 후보 없음 |
| improved | 찾음 | ic · score 0.9365 / IoU 0.691 | ic · score 0.9365 / IoU 0.691 | 저장 후보 없음 |

### RPI3B_Top:00051 · connector J4 · 둘 다 놓침

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | connector · score 0.5094 / IoU 0.833 | connector · score 0.5094 / IoU 0.833 | ic · score 0.0020 / IoU 0.018 |
| improved | 놓침 | connector · score 0.1048 / IoU 0.785 | connector · score 0.1048 / IoU 0.785 | ic · score 0.0024 / IoU 0.670 |

### RPI3B_Top:00059 · ic U13 · 둘 다 찾음

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 찾음 | ic · score 0.9573 / IoU 0.952 | ic · score 0.9573 / IoU 0.952 | capacitor · score 0.0011 / IoU 0.135 |
| improved | 찾음 | ic · score 0.9272 / IoU 0.951 | ic · score 0.9272 / IoU 0.951 | resistor · score 0.0010 / IoU 0.851 |

### RPI3B_Top:00065 · connector RUN · 둘 다 놓침

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | connector · score 0.0062 / IoU 0.938 | connector · score 0.0062 / IoU 0.938 | 저장 후보 없음 |
| improved | 놓침 | connector · score 0.1433 / IoU 0.960 | connector · score 0.1433 / IoU 0.960 | ic · score 0.0013 / IoU 0.007 |

### RPI3B_Top:00066 · connector J12 · 개선안만 찾음

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | connector · score 0.3404 / IoU 0.774 | connector · score 0.3404 / IoU 0.774 | capacitor · score 0.0549 / IoU 0.749 |
| improved | 찾음 | connector · score 0.8116 / IoU 0.554 | connector · score 0.8116 / IoU 0.554 | capacitor · score 0.2465 / IoU 0.865 |

### RPI3B_Top:00067 · connector J11 · 개선안만 찾음

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | connector · score 0.4215 / IoU 0.819 | connector · score 0.4215 / IoU 0.819 | capacitor · score 0.0089 / IoU 0.716 |
| improved | 찾음 | connector · score 0.6903 / IoU 0.587 | connector · score 0.6903 / IoU 0.587 | capacitor · score 0.0011 / IoU 0.710 |

### RPI3B_Top:00068 · connector J10 · 둘 다 놓침

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | connector · score 0.7172 / IoU 0.670 | connector · score 0.7172 / IoU 0.670 | ic · score 0.1627 / IoU 0.873 |
| improved | 놓침 | connector · score 0.1236 / IoU 0.742 | connector · score 0.1236 / IoU 0.742 | capacitor · score 0.0175 / IoU 0.931 |

### RPI3B_Top:00079 · ic U11 · 둘 다 찾음

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 찾음 | ic · score 0.9320 / IoU 0.785 | ic · score 0.9320 / IoU 0.785 | capacitor · score 0.9173 / IoU 0.011 |
| improved | 찾음 | ic · score 0.9243 / IoU 0.735 | ic · score 0.9243 / IoU 0.735 | capacitor · score 0.0022 / IoU 0.175 |

### RPI3B_Top:00084 · ic U10 · 둘 다 놓침

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | ic · score 0.6581 / IoU 0.779 | ic · score 0.6581 / IoU 0.779 | connector · score 0.0052 / IoU 0.013 |
| improved | 놓침 | ic · score 0.6307 / IoU 0.699 | ic · score 0.6307 / IoU 0.699 | resistor · score 0.0308 / IoU 0.251 |

### RPI3B_Top:00089 · connector J7 · 둘 다 놓침

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | connector · score 0.3389 / IoU 0.847 | connector · score 0.3389 / IoU 0.847 | resistor · score 0.0027 / IoU 0.910 |
| improved | 놓침 | connector · score 0.1093 / IoU 0.889 | connector · score 0.1093 / IoU 0.889 | capacitor · score 0.0077 / IoU 0.930 |

### RPI3B_Top:00093 · connector J3 · 개선안만 찾음

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | connector · score 0.3963 / IoU 0.775 | connector · score 0.3963 / IoU 0.775 | capacitor · score 0.0095 / IoU 0.264 |
| improved | 찾음 | connector · score 0.5893 / IoU 0.798 | connector · score 0.6815 / IoU 0.509 | ic · score 0.0037 / IoU 0.058 |

### RPI3B_Top:00110 · ic U2 · 둘 다 찾음

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 찾음 | ic · score 0.9713 / IoU 0.777 | ic · score 0.9713 / IoU 0.777 | connector · score 0.0019 / IoU 0.147 |
| improved | 찾음 | ic · score 0.9352 / IoU 0.788 | ic · score 0.9352 / IoU 0.788 | resistor · score 0.0011 / IoU 0.001 |

### RPI3B_Top:00126 · ic U17 · 개선안만 찾음

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | ic · score 0.0518 / IoU 0.800 | ic · score 0.0518 / IoU 0.800 | resistor · score 0.0014 / IoU 0.824 |
| improved | 찾음 | ic · score 0.9021 / IoU 0.767 | ic · score 0.9021 / IoU 0.767 | resistor · score 0.0011 / IoU 0.535 |

### RPI3B_Top:00138 · connector J8 · 둘 다 놓침

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 놓침 | connector · score 0.3970 / IoU 0.759 | connector · score 0.3970 / IoU 0.759 | ic · score 0.0117 / IoU 0.052 |
| improved | 놓침 | connector · score 0.4935 / IoU 0.683 | connector · score 0.5376 / IoU 0.610 | ic · score 0.0013 / IoU 0.152 |

### RPI3B_Top:00272 · ic U1 · 둘 다 찾음

| 모델 | 고정 threshold에서 매칭 | 같은 클래스 최고 IoU 후보 | 같은 클래스 IoU≥0.5 중 최고 score | 다른 클래스 최고 overlap 후보 |
|---|---|---|---|---|
| baseline | 찾음 | ic · score 0.9507 / IoU 0.968 | ic · score 0.9507 / IoU 0.968 | capacitor · score 0.0036 / IoU 0.004 |
| improved | 찾음 | ic · score 0.9459 / IoU 0.978 | ic · score 0.9459 / IoU 0.978 | connector · score 0.0020 / IoU 0.971 |

## 재현 기록

원본 instance ID별 전체 일반/Pi 비교, 매칭된 예측 좌표·score, 오류 분류, 후보 정보 및 파일 SHA-256은 `error_analysis.json`에 있다. 픽셀 프리뷰·원본 이미지·새 마스크는 만들지 않았다.
