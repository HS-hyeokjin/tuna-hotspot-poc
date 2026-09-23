# tuna-hotspot-poc

참치 선망 조업데이터에서 **어장 탐색에 활용할 수 있는 신호가 실제로 존재하는지 검증**하기 위한 Python PoC입니다.

> V1은 "내일 특정 좌표에서 몇 톤 잡힌다"를 예측하는 시스템이 아닙니다.
> 과거 조업데이터의 품질과 패턴을 검증하고, 어종·조업방법·선박·선장·환경 변수에 설명 가능한 신호가 있는지 확인합니다.

## V1
- 다단 헤더 Excel 로딩
- S0925 / E15958 형식 위경도 변환 및 오류 탐지
- 수온·조류·좌표·복수 투망방법 품질 점검
- S/J, Y/F, B/E, S/H(PS) 분석
- school fish / log fish / PA / mixed 비교
- 투망 성공률 및 투망 1회당 어획량(CPUE 유사 지표)
- 1도 격자 Historical Hotspot
- CatBoost 시간분할 baseline
- 2023~2025 학습 / 2026 검증 우선

## 실행
Python 3.11+ 권장.

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

또는 Windows에서 `run.bat` 실행.

화면에서 Excel을 업로드하거나 로컬에 다음 파일을 배치합니다.

    data/선망_조업보고 데이터.xlsx

## 보안
**회사 조업 원본 데이터는 GitHub에 올리지 않습니다.**

이 저장소는 원본 데이터, CSV/XLSX, 모델 산출물을 .gitignore로 차단합니다.

## 한계
현재 조업일지는 선장이 이미 후보 어장을 선택한 이후의 기록이므로 selection bias가 있습니다. 또한 미조업 위치는 실패 데이터가 아니며, V1에는 미래 SST, SST gradient, Chl-a, 해류/와류 예보, 풍속/파고, FAD 생체량, 소나/어탐, 경쟁선박 정보가 없습니다.

따라서 V1은 미래 어획 보장값이 아니라 **다음 연구 투자 여부를 결정하기 위한 baseline**입니다.

## V2
외부 해양환경 데이터를 날짜/좌표별로 결합하고, 현재 조업정보 대비 실제 개선이 있는지 검증합니다.

과거 조업 + SST/SST gradient/Chl-a/current/eddy/wind/wave → 24~48시간 후보 해역 ranking → 실제 조업 결과로 Top-K 성능 검증
