# 📊 Ensemble SHAP Interpretation Report (Washington DC)

본 문서는 Washington DC 자전거 대여 수요 예측 앙상블 모델에 대해  
**모델별 SHAP 분석 결과를 종합하여 앙상블 관점에서 해석한 보고서**이다.

---

## 1. 앙상블 관점 종합 해석

Washington DC 앙상블 모델은 LightGBM, XGBoost, RandomForest 세 모델의 예측값을 평균하여
최종 수요를 산출한다. 이 구조의 핵심 특징은 다음과 같다.

- 세 모델이 **공통적으로 중요하다고 판단한 요인**이 앙상블에서 가장 강하게 반영됨
- 특정 모델에서만 두드러지는 신호는 평균 과정에서 자연스럽게 완화됨
- 결과적으로 앙상블은 **시간·기상·공간 접근성이라는 보편적 수요 결정 요인**에 안정적으로 의존

이는 단일 모델 대비 과적합 위험을 낮추고, 지역 특성에 덜 민감한 예측을 가능하게 한다.

---

## 2. 모델별 Top5 중요 피처 및 영향 방향 비교

(노트북 SHAP summary / dependence plot 기준)

### 2.1 LightGBM (LGBM)

**Top5 중요 피처**
- N_of_stations_within_2000m  
- n_station_dis(m)  
- temperature  
- quarter of day  
- humidity  

**해석**
- 중·대규모 반경 내 대여소 밀도가 가장 중요한 공간 요인
- 가까운 대여소 접근성은 수요 증가 방향
- 기온은 비선형적으로 작용하며 쾌적 구간에서 수요 증가
- 시간대별 수요 패턴이 뚜렷하게 반영됨

---

### 2.2 XGBoost (XGB)

**Top5 중요 피처**
- N_of_stations_within_2000m  
- n_station_dis(m)  
- temperature  
- quarter of day  
- cloud_cover / precipitation 계열  

**해석**
- LGBM과 매우 유사한 중요 피처 구조
- 공간 인프라 + 시간대 + 기상 조건의 결합 효과를 강하게 학습
- 날씨 변수(구름량·강수)가 특정 조건에서 수요 감소 방향으로 작용

---

### 2.3 Random Forest (RF)

**Top5 중요 피처**
- n_station_dis(m)  
- month  
- temperature  
- quarter of day  
- N_of_stations_within_2000m  

**해석**
- 접근성(n_station_dis)이 가장 강한 단일 요인
- month를 통해 **거친 계절성**을 직접적으로 반영
- 시간대와 기온은 부스팅 모델과 동일한 방향성

---

## 3. 앙상블이 공통적으로 의존하는 요인 도출

### 3.1 세 모델 공통 핵심 요인 (교집합)

- **시간 요인**: quarter of day  
- **기상 요인**: temperature  
- **접근성 요인**: n_station_dis(m)  
- **공간 인프라 요인**: N_of_stations_within_2000m  

➡️ Washington DC 앙상블은  
**“언제(시간대) · 얼마나 쾌적한지(기온) · 얼마나 접근하기 쉬운지 · 주변 인프라가 어떤지”**에
가장 안정적으로 반응한다.

---

### 3.2 모델 계열별 보강 요인

- **부스팅 계열(LGBM/XGB)**  
  - 대여소 밀도, 기상 변수의 비선형 효과를 세밀하게 반영

- **Random Forest**  
  - month 기반 계절성, 단순하고 직관적인 패턴을 강조

앙상블에서는 이 두 성향이 결합되어
**세밀함과 안정성을 동시에 확보**한다.

---

## 4. 한 문장 결론 (보고서용)

Washington DC 앙상블 모델은 시간대와 기온을 중심으로 대여소 접근성과 주변 인프라 밀도를 결합해 수요를 예측하며, 세 모델이 공통적으로 동의하는 핵심 요인이 강화되어 단일 모델 대비 안정적이고 일관된 예측 성능을 보인다.
