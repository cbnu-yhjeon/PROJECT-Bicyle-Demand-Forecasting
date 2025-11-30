# 🚲 서울시 따릉이 수요 예측 프로젝트 — EDA 문서

본 문서는 최종 통합 데이터프레임(시간 기반 + 날씨 + POI 기반)을 활용하여 진행하는 **EDA(Exploratory Data Analysis)** 가이드 및 코드 예시를 포함한다.

## 📌 1. 분석 목적
따릉이 대여량(`rental_count`)에 영향을 미치는 **시간, 날씨, 공간(POI)** 요인을 탐색하고 모델링 전에 반드시 확인해야 할 **데이터 패턴·이상치·상관관계·계절성**을 사전에 파악하는 것이 목적이다.

## 📌 2. 데이터 컬럼 요약

### ✔ 핵심 타깃  
- rental_count

### ✔ 시간 기반  
- quarter_flag, month, weekend

### ✔ 운영/거리 기반  
- used_time(avg), used_dis(avg)
- 거치대수
- n_station_dis(m), n_bus_dis(m), n_school_dis(m), n_park_dis(m)

### ✔ POI Count (반경별)
- N_of_stations_within_*
- N_of_bus_within_*
- N_of_school_within_*
- N_of_park_within_*

### ✔ 날씨 기반  
- temperature, Precipitation, windspeed, humidity, sunshine, snowcover, cloudcover, weathersit

---

## 📌 3. EDA 전체 로드맵

### ✔ 1) 데이터 기본 구조 점검
- 결측치, 이상치, 스케일  
- rental_count 0 비율  
- 거리기반 값 검증  

### ✔ 2) 수요량(rental_count) 분석
- 분포(히스토그램/KDE)
- 월·주말·분기별 boxplot
- station-level 분포 비교

### ✔ 3) 시간 기반 패턴 분석
- 월·분기 변화  
- 주말 vs 평일  
- 시계열 추세

### ✔ 4) 날씨 기반 영향도
- 온도/강수/풍속/습도 vs rental_count  
- 날씨 코드별 박스플롯  

### ✔ 5) POI 기반 영향도
- 거리 기반 변수  
- 반경 POI count  
- POI 상관 히트맵  

### ✔ 6) 전체 상관관계 분석 (heatmap)

### ✔ 7) Station clustering 

---

