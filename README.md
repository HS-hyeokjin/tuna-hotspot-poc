# tuna-hotspot-poc

참치 선망 조업데이터에서 **어장 탐색에 활용할 수 있는 신호가 실제로 존재하는지 검증**하기 위한 Python PoC입니다.

> 이 프로젝트는 "내일 특정 좌표에서 몇 톤 잡힌다"를 확정 예측하는 시스템이 아닙니다.
> 과거 조업데이터의 품질과 패턴을 검증하고, 어떤 데이터가 실제로 예측력을 추가하는지 확인하는 연구용 PoC입니다.

## 현재 브랜치

`feature/v0.2-model-validation`

V2에서는 V1의 단일 CatBoost 결과가 실제 어장/환경 신호인지, 아니면 조업방법·선박·선장 효과를 학습한 결과인지 분리해서 검증합니다.

## v0.2.0 핵심 기능

- 전체 모델 vs 위치/환경 Only 모델 비교
- School Fish 단독 모델
- School Fish 위치/환경 Only 모델
- PA 단독 모델
- S/J · Y/F · B/E 어종별 모델 비교
- ROC-AUC / PR-AUC / Brier Score
- MAE / Median AE
- Top 10% · 20% Catch Lift
- CatBoost Feature Importance
- SHAP Global Importance
- SHAP 변수별 방향 확인
- 연도별 Walk-forward validation
- 당일 총어획량 vs 어종별 합계 데이터 정의 점검
- 기존 Historical Hotspot / CPUE 분석 유지

자세한 검증 설계는 `docs/V0.2_MODEL_VALIDATION.md`를 참고하세요.

## 브랜치 실행

    git fetch origin
    git switch feature/v0.2-model-validation

Windows:

    run.bat

직접 실행:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

## 데이터

화면에서 Excel을 업로드하거나 로컬에 다음 파일을 배치합니다.

    data/선망_조업보고 데이터.xlsx

## 보안

**회사 조업 원본 데이터는 GitHub에 올리지 않습니다.**

`.gitignore`가 원본 데이터, CSV/XLSX 및 모델 산출물을 차단합니다.

## 해석 시 주의

현재 조업일지는 선장이 이미 후보 어장을 선택한 이후의 기록이므로 selection bias가 있습니다.

또한 아직 정의 확인이 필요한 항목이 있습니다.

- PA
- S/H (PS)
- Y/F (PS)
- Y/F (GG)
- 중량구간(+7.5, +4, +3, -3)의 정확한 범위
- 조류 컬럼의 단위 및 의미
- 당일 어획량과 어종별 합계의 관계

V2에서는 이러한 미확정 정의를 임의로 확정하지 않습니다.

## V3 후보

V2에서 위치·환경 신호가 유효하다고 판단되면 다음 데이터를 날짜/좌표 기준으로 결합합니다.

    SST
    SST Gradient
    Chlorophyll-a
    Current U/V
    Sea Surface Height / Eddy
    Wind
    Wave

그 후 동일한 Walk-forward 검증으로 **외부 데이터를 추가했을 때 실제 성능이 개선되는지** 비교합니다.
