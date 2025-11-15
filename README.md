# 🚲 Bicycle Demand Forecasting Project  
### 자전거 수요 예측 머신러닝 프로젝트  
**Project-Bicycle-Demand-Forecasting**

---

# 📌 프로젝트 개요  

본 프로젝트는 미국 **UCI Bike Sharing Dataset (Washington DC)** 과  
**서울시 따릉이 시간대별 대여/반납 데이터**,  
그리고 **기상청 API Weather Data**를 활용하여  

도시형 자전거 공유 서비스의 **시간대별 수요를 예측하는 머신러닝 프로젝트**이다.

4명이 협업하며 다음 프로세스로 진행한다:

1. **데이터 수집 및 저장 (Google Drive)**
2. **전처리 및 EDA**
3. **개별 모델링**
4. **앙상블 및 결론**

---

# 📂 1. 프로젝트 폴더 구조  

```
Project-Bicycle-Demand-Forecasting/
│
├── Data/                         
│   ├── raw/                      
│   ├── interim/                  
│   ├── processed/                
│   └── data_sources_and_license.md
│
├── docs/                         
│
├── notebooks/                    
│
├── src/                          
│   ├── data/                     
│   ├── features/                 
│   ├── models/                   
│   └── visualization/            
│
└── .gitignore
```

---

# 📌 2. 데이터 관리 정책 (Google Drive 기반)

📂 **Google Drive 링크:**  
https://drive.google.com/drive/u/1/home

### ✔ 모든 데이터는 GitHub 아닌 Google Drive에서만 관리  
GitHub에는 **절대 어떠한 데이터 파일도 업로드하지 않는다.**  
`.gitignore`에 `/Data/` 전체 제외 규칙 포함.

---

# 📌 3. 데이터 구분 규칙

데이터는 반드시 아래 3단계로 구분하여 Google Drive에서 관리한다.

---

## 1️⃣ Raw Data – 원본 데이터  
```
/Data/raw/
```
- 다운로드한 원본 그대로 저장  
- 수정 **절대 금지(Immutable)**  
- UCI Bike Sharing / 서울 따릉이 / 기상청 API 원본

---

## 2️⃣ Interim Data – 중간 전처리 데이터  
```
/Data/interim/
```
- 스키마 통합  
- Null 처리  
- 날짜/시간 정규화  
- 기상 데이터 병합  
- 타입 변환 등 1단계 가공

---

## 3️⃣ Processed Data – 모델링 가능 최종 데이터  
```
/Data/processed/
```
- Feature Engineering 완료  
- Scaling / Encoding  
- Lag / Rolling Feature 포함  
- Train/Test Split 반영  
- **모든 모델이 사용하는 공식 데이터셋**

---

# 📌 4. Google Drive 데이터 사용 규칙

- 모든 팀원은 Drive에서 데이터를 다운로드/업로드한다.  
- Raw → 수정 금지  
- Interim → 중간 버전 저장  
- Processed → 재현 가능한 코드 기반으로 생성  

---

# 📚 5. 데이터 출처  

모든 데이터 출처는 아래 문서에 기록한다:

📄 `Data/data_sources_and_license.md`

- UCI Bike Sharing Dataset  
- 서울 열린데이터 광장 따릉이  
- 기상청(동네예보, 단기예보) API  

---

# 🧪 6. 프로젝트 수행 단계

### 📍 Week 1  
- 데이터 수집(UCI/따릉이/기상청)  
- Google Drive 구조 세팅  
- Raw/Interim/Processed 설계  

### 📍 Week 2  
- 전처리  
- 스키마 통합  
- EDA  
- 시계열 분석  

### 📍 Week 3  
- 개인별 모델링(XGBoost/LGBM/RF/LSTM 등)  
- Hyperparameter Tuning  

### 📍 Week 4  
- Voting/Stacking 앙상블  
- 결과 시각화  
- 보고서 작성  

---

# 📁 7. notebooks/ 운영 규칙  

### ✔ 파일명 규칙  
```
01_eda.ipynb
02_preprocessing.ipynb
03_modeling_xgboost.ipynb
04_modeling_lstm.ipynb
05_ensemble.ipynb
```

### ✔ 규칙  
- Notebook에서 로컬 CSV 직접 로드 금지  
- 데이터 로딩은 `src/data/loader.py`만 사용  

---

# 📁 8. src/ 코드 구조  

## src/data  
- loader.py  
- weather_api.py  
- schema_unify.py  

## src/features  
- preprocess.py  
- feature_engineering.py  

## src/models  
- train_xgboost.py  
- train_lgbm.py  
- train_rf.py  
- evaluate.py  

## src/visualization  
- plot_utils.py  
- eda_plots.py  

---

# 🤝 팀 협업 규칙  

## ✔ Git 브랜치 전략  
```
main        → 최종 결과
dev         → 통합 개발
feature/*   → 개인 개발
```

## ✔ Commit 규칙  
```
feat: Add preprocessing logic
fix: Resolve merge bug
docs: Update README
```

---

# 🎯 프로젝트 목표  

- 도시형 자전거 수요 예측  
- 시계열 기반 ML 경험  
- Google Drive 기반 데이터 관리 체계 구축  
- Feature Engineering → 모델링 → 앙상블 경험  
- 협업형 머신러닝 프로젝트 프로세스 이해  

---
