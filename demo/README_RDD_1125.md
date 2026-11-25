# 야간 B1·B2 평가 준비 및 차로 경고 시연

이 코드는 AI Hub 178의 `41776` 묶음에서 팀 검증 이미지 386장을 추출하고 Zero-DCE 보정본을 준비합니다. 영상 시연은 B1 검출 박스의 아래쪽 중앙점을 사다리꼴 차로 영역과 비교합니다. 차로 안 판정으로 전환될 때 UDP 경고를 한 번 전송합니다.

## 실행 환경

Windows PowerShell 예시입니다. `data/`의 `extract_night_val.py`, `prepare_night_eval.py`와 `demo/`의 `prepare_zero_dce_b2.py`, `lane_demo.py`, `udp_warning_test.py`를 로컬 `p_project` 폴더 바로 아래에 복사한 뒤 같은 위치에서 실행합니다. 해당 스크립트는 데이터 폴더를 스크립트 위치 기준으로 찾습니다.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install "ultralytics==8.4.207"
```

## 검증 데이터 준비

팀 공유 `images.csv`, `objects.csv`를 `metadata`에 두고 파일키 `41776`의 `1.Frontback_N02.zip`을 프로젝트 하위 폴더에 둡니다.

```powershell
.\.venv\Scripts\python.exe .\extract_night_val.py
.\.venv\Scripts\python.exe .\prepare_night_eval.py
.\.venv\Scripts\python.exe .\prepare_zero_dce_b2.py --preview
.\.venv\Scripts\python.exe .\prepare_zero_dce_b2.py
```

원본은 `b2_night/images/val`, 보정본은 `b2_night_zero_dce/images/val`에서 확인합니다. 보정 중 중단한 경우 같은 전체 보정 명령으로 다시 실행하면 저장된 이미지부터 재사용합니다. B1·B2 수치는 팀 평가 패키지의 `make_list.py`와 `07_eval.py`로 산출합니다. 양쪽 모두 동일한 검증 목록, `b1_best.pt`, 입력 크기 1280, 채점 신뢰도 0.21을 사용합니다.

## 차로 판단과 UDP 경고

첫 번째 터미널에서 수신을 시작합니다.

```powershell
py .\udp_warning_test.py receive 127.0.0.1
```

두 번째 터미널에서 B1 가중치와 주간 시험로의 예시 영상으로 실행합니다. 아래 사다리꼴 좌표는 주간 시험로 클립에 맞춘 예시입니다. 다른 영상에서는 영상 속 차로를 보고 좌표를 조정합니다.

```powershell
.\.venv\Scripts\python.exe .\lane_demo.py .\in_lane_cones.mp4 .\for_junhyeok\b1_best.pt --lane 0.51 0.58 0.52 0.46 0.70 0.72 --udp-host 127.0.0.1 --output .\results\lane_udp_in_lane.mp4
```

수신 터미널에 송신과 같은 `sent_at`의 경고가 찍히는지 확인합니다. 영상 입력 처리 속도와 프레임 처리 시작부터 UDP 송신까지의 시간은 `lane_demo.py`가 출력합니다.

- Zero-DCE 모델과 공식 가중치는 Guo 외, CVPR 2020 연구에서 가져옵니다. [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE)
- 주간 시험로 예시 영상의 출처입니다. [Drive Anywhere](https://drive-anywhere.github.io/)
