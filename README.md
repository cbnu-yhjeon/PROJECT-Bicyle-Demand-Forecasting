# 🚲 Bicycle Demand Forecasting Project  
### 자전거 수요 예측 머신러닝 프로젝트  
**Project-Bicycle-Demand-Forecasting**

---

## 📌 프로젝트 개요

본 프로젝트는 다음 데이터를 활용하여  
**도시형 자전거 공유 서비스의 시간대별 수요를 예측하는 머신러닝 프로젝트**입니다.

- **UCI Bike Sharing Dataset (Washington, D.C.)**
- **서울시 따릉이 시간대별 대여 데이터**
- **기상 데이터 (Open-Meteo / 기상청 API 기반 가공 데이터)**

본 프로젝트는 **4인 협업 프로젝트**로 진행되었으며,  
데이터 수집부터 전처리, 모델링, 앙상블, 해석(SHAP)까지의 **전체 ML 파이프라인 경험**을 목표로 합니다.

---

## 🧭 프로젝트 진행 흐름

1. 데이터 수집 및 관리 (Google Drive)
2. 데이터 전처리 및 EDA
3. 단일 모델 학습 (XGBoost / LightGBM / RandomForest)
4. 앙상블 및 성능 비교
5. 결과 해석 및 문서화

---

## 📂 프로젝트 폴더 구조 (요약)

```
Project-Bicycle-Demand-Forecasting/
│
├── Data/
│   ├── raw/                    # 원본 데이터 (Google Drive 관리)
│   ├── interim/                # 중간 전처리 데이터
│   ├── processed/              # 모델링 최종 데이터
│   └── data_sources_and_license.md
│
├── docs/                       # 주차별 문서, 리포트, PPT
├── notebooks/                  # EDA / 모델링 / SHAP 분석 노트북
├── models/                     # 학습 완료된 모델(pkl)
├── ensemble_result/             # 앙상블 예측 결과
├── logs/                        # 학습 및 평가 로그
├── shap_result/                 # SHAP 결과
│
├── src/
│   ├── Models/                 # 학습/평가 스크립트
│   ├── ensemble/               # 앙상블 로드 및 실행
│   └── utils/                  # 전처리 / 모델 유틸
│
└── README.md
```

---

## 📌 데이터 관리 정책 (Google Drive 기반)

📂 **Google Drive 링크**  
https://drive.google.com/drive/u/1/home

### ✔ 핵심 원칙

- **모든 데이터는 GitHub에 업로드하지 않음**
- GitHub에는 코드, 문서, 로그만 포함
- 데이터는 반드시 Google Drive에서만 관리
- `.gitignore`에 `/Data/` 전체 제외 규칙 적용

---

## 📊 데이터 단계별 구분 규칙

### 1️⃣ Raw Data – 원본 데이터
```
/Data/raw/
```
- 다운로드한 원본 그대로 저장
- 수정 및 가공 **절대 금지**
- UCI / 서울 따릉이 / 기상 원본 데이터

### 2️⃣ Interim Data – 중간 전처리 데이터
```
/Data/interim/
```
- 스키마 통합
- 결측치 처리
- 날짜/시간 정규화
- 날씨 데이터 병합
- 1차 가공 데이터

### 3️⃣ Processed Data – 모델링 최종 데이터
```
/Data/processed/
```
- Feature Engineering 완료
- Encoding / Scaling
- 시간 기반 Feature
- 모델 학습에 직접 사용되는 공식 데이터셋

---

## 📚 데이터 출처 및 라이선스

모든 데이터 출처 및 라이선스 정보는 아래 문서에 정리되어 있습니다.

📄 `Data/data_sources_and_license.md`

- UCI Bike Sharing Dataset
- 서울 열린데이터 광장 (따릉이)
- Open-Meteo API / 기상청 데이터

---

## 🧪 프로젝트 수행 단계

### 📍 Week 1 – 데이터 수집 및 설계
- 데이터 수집 (UCI / 서울 / 날씨)
- Google Drive 구조 설계
- Raw / Interim / Processed 기준 정의

### 📍 Week 2 – 전처리 및 EDA
- 데이터 병합
- Feature 탐색
- 시계열 패턴 분석
- 이상치(IQR) 실험

### 📍 Week 3 – 모델링
- 단일 모델 학습
  - XGBoost
  - LightGBM
  - RandomForest
- IQR 적용 vs 미적용 비교

### 📍 Week 4 – 앙상블 및 해석
- 앙상블(Averaging) 구성
- 성능 비교 및 시각화
- SHAP 기반 모델 해석
- 최종 보고서 및 PPT 정리

---

## 📓 Notebooks 운영 규칙

### ✔ 파일명 규칙
```
01_eda.ipynb
02_preprocessing.ipynb
03_modeling_xgboost.ipynb
04_modeling_lgbm_rf.ipynb
05_ensemble.ipynb
06_shap_analysis.ipynb
```

### ✔ 운영 원칙
- Notebook에서 로컬 CSV 직접 로드 금지
- 데이터 로딩은 `src/utils/data_preprocessing` 또는 공용 로더만 사용
- 결과는 문서(`docs/`) 또는 결과 폴더에 저장

---

## ⚙️ 모델 구성

- **단일 모델**
  - XGBoost
  - LightGBM
  - RandomForest

- **비교 실험**
  - IQR 이상치 제거 적용 / 미적용

- **앙상블**
  - 단순 평균 기반 앙상블

- **해석**
  - SHAP Summary / Dependence Plot

---

## 🤝 협업 규칙

### ✔ Git 브랜치 전략
```
main        : 최종 결과
dev         : 통합 개발
feature/*   : 개인 작업
```

### ✔ Commit 메시지 규칙
```
feat: Add feature engineering logic
fix: Fix model loading bug
docs: Update README
refactor: Cleanup preprocessing code
```

---

## 🎯 프로젝트 목표

- 도시형 자전거 수요 예측 문제 해결
- 시계열 기반 머신러닝 파이프라인 경험
- Google Drive 기반 데이터 관리 실습
- Feature Engineering → 모델링 → 앙상블 전 과정 이해
- 협업형 데이터 사이언스 프로젝트 수행 경험

---

