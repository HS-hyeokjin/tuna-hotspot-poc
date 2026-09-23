# tuna-hotspot-poc

참치 선망 조업데이터와 외부 해양환경 데이터를 이용해 **어장 탐색에 활용할 수 있는 신호가 실제로 존재하는지 검증**하는 Python PoC입니다.

> 이 프로젝트는 특정 좌표의 미래 어획량을 확정하는 시스템이 아닙니다.
> 시간분할 검증으로 어떤 데이터가 실제 예측력을 추가하는지 단계적으로 확인합니다.

## 현재 브랜치

`feature/v0.3-ocean-data`

- `main`: V1 baseline
- `feature/v0.2-model-validation`: 조업방법/선박/선장 효과 분리 검증
- `feature/v0.3-ocean-data`: Copernicus Marine 외부 해양데이터 결합

병합은 하지 않고 버전별 브랜치로 관리합니다.

## V3 핵심

Copernicus Marine의 다음 일자료를 조업 날짜·좌표에 결합합니다.

- SST
- Current U/V
- Current speed / direction
- Sea Surface Height
- Chlorophyll-a
- SST Gradient 근사치

그리고 동일한 검증 방식으로 아래를 비교합니다.

1. V2 기존 변수 모델
2. 외부 해양 환경 Only
3. 외부 해양 + 선박/선장/조업방법
4. 기존 수온/조류 + 외부 해양 전체

평가지표:

- ROC-AUC
- PR-AUC
- Brier Score
- MAE / Median AE
- Top 10% / 20% Catch Lift
- SHAP
- Walk-forward validation

## 설치

Python 3.11 권장.

    git fetch origin
    git switch feature/v0.3-ocean-data

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

Copernicus Marine 로그인이 아직 저장되지 않았다면:

    copernicusmarine login

## 1. 조업데이터

화면 업로드 또는 로컬에 배치:

    data/선망_조업보고 데이터.xlsx

회사 원본 데이터는 GitHub에 커밋하지 않습니다.

## 2. 외부 해양데이터 생성

먼저 100건 연결 테스트:

    python scripts/build_ocean_features.py --limit 100

정상 확인 후 전체:

    python scripts/build_ocean_features.py

생성 파일:

    data/external/processed/fishing_ocean_features.parquet

월/공간 블록별 캐시:

    data/external/cache/

기존 캐시를 무시하고 새로 조회:

    python scripts/build_ocean_features.py --force

## 3. 실행

    streamlit run app.py

Ocean Data 탭에서 외부 데이터 Coverage와 조업일지 수온/Copernicus SST 비교를 먼저 확인하세요.

그 다음 AI 실험실에서 V2/V3 모델을 동일 조건으로 비교합니다.

## 날짜변경선

조업좌표가 E/W 180도 부근을 오가므로 전체 좌표를 하나의 최소/최대 경도 bbox로 요청하지 않습니다.

V3는 월별 + 20도 공간 블록으로 Copernicus 요청을 분할해 날짜변경선 문제와 과도한 다운로드를 줄입니다.

## 문서

- `docs/V0.2_MODEL_VALIDATION.md`
- `docs/V0.3_OCEAN_DATA.md`

## 다음 단계

V3에서 외부 해양데이터의 성능 개선이 확인된 경우에만 V4에서
Copernicus +24h/+48h forecast 기반 후보 해역 Ranking으로 확장합니다.
