
# 📘 서울시 따릉이 최종 데이터 스키마 요약 및 분석 보고서

본 문서는 제공된 **Seoul Bike Sharing Final Dataset**의 스키마와  
각 변수의 의미, 데이터 유형, 분석에서의 활용 가능성을 정리한 문서입니다.

---

# 🏙 1. 데이터 개요

- **도시:** 서울(Seoul)
- **행(row):** 14,581,612 (원본 기준)
- **특징(feature):** 시간대, POI(Points of Interest), 날씨(Weather), 운영 정보 혼합
- **타깃 변수:** `rental_count` (시간대별 대여량)

---

# 🧩 2. 최종 스키마 (Feature Schema)

| 변수명 | 설명 | 데이터 타입 | 카테고리 |
|--------|------|-------------|-----------|
| `date` | 관측 날짜 | datetime | 시간 |
| `quarter_flag` | 하루를 4등분한 시간구간(0~3) | float | 시간 |
| `month` | 월(1~12) | float | 시간 |
| `weekend` | 주말 여부(0 weekday, 1 weekend) | int | 시간 |
| `rental_count` | 시간대별 자전거 실제 대여량 | int | 타깃 |
| `used_time(avg)` | 평균 사용 시간(초 단위) | float | 운영 정보 |
| `used_dis(avg)` | 평균 이동 거리(m) | float | 운영 정보 |
| `Holding quantity` | 해당 스테이션의 보유 자전거 수 | int | 운영 정보 |

---

# 🚏 3. POI(주변 환경 요인) 관련 스키마

이 데이터셋은 대여소 주변의 주요 POI까지의 거리와 개수를 포함한다.

## 🔹 **거리 기반 피처**

| 변수명 | 설명 | 데이터 타입 |
|--------|------|-------------|
| `n_station_dis(m)` | 가장 가까운 대여소까지 거리 | float |
| `n_bus_dis(m)` | 가장 가까운 버스정류장까지 거리 | float |
| `n_school_dis(m)` | 가장 가까운 학교까지 거리 | float |
| `n_park_dis(m)` | 가장 가까운 공원까지 거리 | float |

---

## 🔹 **반경 내 개수 기반 피처**

### (1) 대여소(Stations)
| 반경 | 변수명 |
|------|---------|
| 100m | `N_of_stations_within_100m` |
| 500m | `N_of_stations_within_500m` |
| 1000m | `N_of_stations_within_1000m` |
| 1500m | `N_of_stations_within_1500m` |
| 2000m | `N_of_stations_within_2000m` |

### (2) 버스정류장(Bus Stops)
| 반경 | 변수명 |
|------|---------|
| 100m | `N_of_bus_within_100m` |
| 500m | `N_of_bus_within_500m` |
| 1000m | `N_of_bus_within_1000m` |
| 1500m | `N_of_bus_within_1500m` |
| 2000m | `N_of_bus_within_2000m` |

### (3) 학교(Schools)
| 반경 | 변수명 |
|------|---------|
| 100m | `N_of_school_within_100m` |
| 500m | `N_of_school_within_500m` |
| 1000m | `N_of_school_within_1000m` |
| 1500m | `N_of_school_within_1500m` |
| 2000m | `N_of_school_within_2000m` |

### (4) 공원(Parks)
| 반경 | 변수명 |
|------|---------|
| 100m | `N_of_park_within_100m` |
| 500m | `N_of_park_within_500m` |
| 1000m | `N_of_park_within_1000m` |
| 1500m | `N_of_park_within_1500m` |
| 2000m | `N_of_park_within_2000m` |

---

# 🌤 4. 날씨(Weather) 스키마

| 변수명 | 설명 | 데이터 타입 |
|--------|------|-------------|
| `temperature` | 기온(°C) | float |
| `Precipitation` | 강수량(mm) | float |
| `windspeed` | 풍속(m/s) | float |
| `humidity` | 상대습도(%) | float |
| `sunshine` | 일조시간(hr) | float |
| `snowcover` | 적설량(cm) | float |
| `cloudcover` | 구름 양(0~10) | float |
| `weathersit` | 날씨 상태 코드(1~4) | int |

---

# 🧠 5. 데이터 분석 요약

## ✔ 시간 요인은 수요 예측에서 핵심
- `quarter_flag`, `month`, `weekend` 은 매우 강한 패턴을 가짐
- 하루 시간대 변화가 수요 변동을 크게 설명함

## ✔ 주변 환경(POI) 요인의 영향력도 큼
- 특히 반경 내 대여소 수, 버스정류장 수가 수요와 높은 상관관계
- 도심 / 주거지 / 공원 인접 여부가 수요 패턴을 명확히 구분

## ✔ 날씨는 모든 도시에서 공통으로 중요한 변수
- 기온, 강수량, 풍속, 습도는 대여량에 직접적인 영향을 미침  
  (특히 기온은 비선형적 영향 존재)

## ✔ 운영 정보(usage features)
- `used_time(avg)`, `used_dis(avg)`는 스테이션의 트래픽 수준을 반영  
- 하지만 이상치가 많아 모델링 시 사전 정규화 필요

---

# 🎯 6. 피처 그룹 구성 요약

| 그룹 | 포함 피처 |
|------|------------|
| **시간 기반(Time)** | date, quarter_flag, month, weekend |
| **운영 기반(Operational)** | Holding quantity, used_time(avg), used_dis(avg) |
| **POI 기반(Location)** | 거리 기반 + 반경 내 개수 기반 |
| **날씨 기반(Weather)** | temperature, humidity, windspeed, cloudcover 등 |
| **타깃(Target)** | rental_count |

---

# 🏁 7. 결론

서울시 따릉이 데이터는  
- **시간 패턴이 매우 강하고**,  
- **POI 특징이 도시 구조를 잘 설명하며**,  
- **날씨가 수요 변동의 핵심 외생 변수**인  
우수한 수요 예측용 데이터셋입니다.

해당 스키마는 향후 다음 작업에 활용될 수 있습니다:

- 모델 피처링 표준화  
- SHAP 기반 변수 중요도 분석  
- 서울 vs 워싱턴 DC 비교 분석  
- 수요 예측 모델의 고도화/앙상블 모델 구성

---

# 📁 작성 파일 정보
본 문서는 자동 생성된 마크다운 분석 보고서입니다.
