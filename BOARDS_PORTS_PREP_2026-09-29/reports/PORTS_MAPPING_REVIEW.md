# 포트 데이터셋 매핑 검수

검수일: 2026-09-29

**대표 사진에서 클래스의 뜻과 박스 단위는 확인했지만, 포트 데이터셋의 전체 라벨 완전성은 아직 검증하지 못했습니다.** 클래스 매핑 확정은 모든 사진에서 보이는 대상이 빠짐없이 라벨링되었다는 뜻이 아닙니다. 두 Raspberry Pi 자료의 `incomplete_requires_review` 상태를 유지하며, 이 검수에서는 학습하지 않았습니다.

## 1. Raspberry Pi 클러스터 자료 — dpiwf v3

[공개 원본 및 버전 3](https://universe.roboflow.com/kerchick-gmail-com/raspberry-pi-dpiwf/dataset/3), CC BY 4.0. 별도 검수 에이전트가 아래 포트 6종의 대표 crop 시트와 클러스터 관련 3종 시트를 직접 확인했습니다. 포트별 대표 시트는 48개 crop이며, 원본 사진 3장도 함께 확인했습니다.

| 원본 클래스 | 대상 클래스 | 확인한 의미·단위 |
|---|---|---|
| `usb-2`, `usb-3` | `usb_stack` | USB-A **2단 적층 금속 하우징 1개**. 소켓 구멍 하나 또는 두 하우징 전체를 한 박스로 묶는 정의가 아님 |
| `gigabit ethernet` | `ethernet_port` | RJ45 커넥터 하우징 전체 |
| `gpio header` | `gpio_header` | GPIO 헤더 핀열 전체 |
| `usb-c 3a-5v` | `usb_c_power` | USB-C 전원 커넥터 |
| `2.5a micro usb` | `micro_usb` | micro USB 전원 커넥터 |

`raspberry-pi` 박스는 개별 보드가 아닌 6대 Pi 클러스터 전체를 감쌉니다. `server`와 `server slot`도 이 클러스터의 개별 Pi/슬롯입니다. 일반 서버랙 사진이라는 기존 설명은 정정했으며, 이 세 클래스는 현재 보드 종류 학습용 박스로 사용하지 않습니다.

**확인된 라벨 누락:** `deg_095`와 `deg_115` 원본에서 micro HDMI 쌍과 하단 보드의 표준 HDMI 소켓이 보이지만, 원본 클래스 목록에 HDMI류가 없습니다. 따라서 `micro_hdmi`와 `hdmi` 보완이 필요합니다. `heatsink` 클래스도 없으나, 512px 원본에서 실제 방열판 존재와 누락까지 확정하지는 못했습니다. 상단 냉각팬이 보인다는 사실만으로 방열판 존재를 확정하지 않았습니다.

**분할 주의:** 동일한 클러스터를 여러 각도에서 촬영한 자료입니다. 파일명이 다르거나 pHash가 멀어도 독립 장면으로 취급하지 않고 동일 장면 그룹에 묶어 train/val/test 사이에 나누지 않아야 합니다.

## 2. Raspberry Pi 단일 보드 자료 — vniye v1

[공개 원본](https://universe.roboflow.com/raspberrypi-4jqsd/raspberrypi-vniye), CC BY 4.0. 아래 내용은 **주 작업자가 수행한 대표 시트 검수 결과**와 현재 `class_map.json`을 정리한 것입니다. 이 보고서 작성자의 별도 전수 검수 결과는 아닙니다. 주 작업자는 12개 클래스별 48개 대표 crop을 검수했으며, 사진은 Pi 3B+ 배치로 확인했습니다.

| 실제 export의 원본 이름 | 현재 대상 | 대표 사진에서 확인한 대상 |
|---|---|---|
| `0` | `gpio_header` | 40핀 GPIO 헤더 |
| `4` | `ethernet_port` | RJ45 하우징 |
| `6` | `hdmi` | 표준 크기 HDMI 소켓 |
| `7` | `micro_usb` | micro USB 전원 소켓 |
| `8` | `usb_stack` | 2단 USB-A 하우징; 사진에 두 묶음 |
| `1`, `2`, `3` | 제외 | 오디오 잭, CSI, DSI 커넥터 |
| `5`, `9`, `10`, `11` | 제외 | SoC, LAN 컨트롤러, 무선부 실드, 실크 로고 |

웹의 클래스 표시 순서로 숫자 이름을 추정해서는 안 됩니다. **YOLO 라벨의 클래스 ID는 `names` 배열 위치이며, 숫자 문자열로 된 이름 자체의 값과 다를 수 있습니다.** 이 export의 `6`을 `micro_hdmi`로, `7`을 USB-C로 변환하면 잘못된 매핑입니다.

이 자료도 각 사진의 대상 누락을 전수 검사하지 않았습니다. 방열판 클래스 부재와 실제 방열판 미라벨 여부를 구분하고, 동일 촬영 세션을 분할 사이에 나누지 않아야 합니다.

## 3. STM32 Nucleo 자료 상태

[공개 원본](https://universe.roboflow.com/computervisionaihub-n6cdj/stm32), CC BY 4.0. 주 작업자 확인에 따르면 원본 107장 기반 fork/version 생성은 완료했으나 **Chrome이 ZIP 다운로드를 차단하여 로컬 가져오기와 전체 검증은 완료되지 않았습니다.** 학습 가능한 Nucleo 로컬 자료로 집계하면 안 됩니다.

공개 미리보기 2장의 표본 검수에서는 `STM32_board`가 NUCLEO-H723ZG 보드 전체, `Nucleo`가 로고로 확인되었습니다. `ST-LINK V3`는 디버거 영역 전체가 아닌 디버거 MCU 칩을 감싸므로 `st_link`의 정의를 먼저 정해야 합니다. `Connector_power`도 전체 export에서 커넥터 종류를 추가 확인해야 합니다. 미리보기에는 polygon 라벨이 있으며, 샘플 판단을 전체 export 검증으로 확대하지 않았습니다.

## 확인 경로

- 기준 작업 폴더: `C:\Users\hkjun\Documents\Codex\2026-09-27\new-chat\work\boards_ports`
- 현재 매핑: 위 폴더의 `class_map.json`
- dpiwf 검수 인덱스: 위 폴더의 `staging\rf_raspberry_pi_dpiwf\review\acda4597dcda181c\1f59306fcd17c18e\index.json`
- dpiwf 원본 위치: 위 폴더의 `raw\extracted\acda4597dcda181cda154ca94195d36e625fa78a85940b68c1e9cac81ee50782\train\images`
- 직접 확인한 원본: `deg_095_jpg.rf.5c256cdfeae72202bccb19ea464193ca.jpg`, `deg_115_jpg.rf.6b51a770651856cb537d4844d0c1a7d4.jpg`, `deg_195_jpg.rf.06b65bb4e8e8482b1221fe4c4f4f50d8.jpg`
- vniye 검수 인덱스: 위 폴더의 `staging\rf_raspberrypi_vniye\review\15bf590aa817174d\676caf133492cfae\index.json` 및 같은 상위 폴더의 `25a4582fb4d694b6\index.json`

이 문서는 표본 시각 검수의 범위와 한계를 기록합니다. 데이터셋 수량, 클래스 매핑 파일, 분할 결과는 이 문서를 작성하면서 변경하지 않았습니다.
