
# 📊 DC Bike Rental Dataset – Schema Summary

본 데이터셋은 **Washington DC 자전거 대여 데이터**를 기반으로  
**시간 정보, 대여 수요, 대여소 메타정보, 주변 POI 밀도, 날씨 정보**를 통합한 분석용 데이터입니다.

---

## 1️⃣ 대여소 기본 정보 (Station Metadata)

| 컬럼명 | 타입 | 설명 |
|------|----|----|
| NAME | string | 대여소 이름 |
| STATION_ID | string | 대여소 고유 ID |
| STATION_TYPE | string | 대여소 유형 (예: classic) |
| STATION_STATUS | string | 대여소 상태 |
| REGION_ID | float | 지역 ID |
| REGION_NAME | string | 지역명 |
| CAPACITY | int | 대여소 최대 수용 가능 자전거 수 |
| RENTAL_METHODS | string | 대여 방식 (KEY, CREDITCARD 등) |

---

## 2️⃣ 위치 / 공간 정보 (Geospatial)

| 컬럼명 | 타입 | 설명 |
|------|----|----|
| lat / lon | float | 위도 / 경도 |
| LATITUDE / LONGITUDE | float | GIS 기준 위도 / 경도 |
| X / Y | float | 투영 좌표계 좌표 |
| GIS_ID | string | GIS 객체 ID |
| OBJECTID | int | GIS 객체 고유 번호 |
| GIS_LAST_MOD_DTTM | datetime | GIS 정보 마지막 수정 시각 |

---

## 3️⃣ 시간 관련 변수 (Temporal Features)

| 컬럼명 | 타입 | 설명 |
|------|----|----|
| date | date | 날짜 |
| month | int | 월 (1~12) |
| quarter of day | int | 하루를 4구간으로 나눈 시간대 (0~3) |
| weekend | int | 주말 여부 (1=주말) |
| LAST_REPORTED | datetime | 대여소 상태 마지막 보고 시각 |

---

## 4️⃣ 대여 수요 및 이용 정보 (Target & Usage)

| 컬럼명 | 타입 | 설명 |
|------|----|----|
| rental_count | int | **해당 시간대 대여 횟수 (Target)** |
| used_time(avg) | timedelta | 평균 이용 시간 |
| used_dis(avg) | float | 평균 이동 거리 (m) |

---

## 5️⃣ 대여소 실시간 상태 정보 (Bike Availability)

| 컬럼명 | 타입 | 설명 |
|------|----|----|
| NUM_DOCKS_AVAILABLE | int | 사용 가능한 거치대 수 |
| NUM_DOCKS_DISABLED | int | 비활성 거치대 수 |
| NUM_BIKES_AVAILABLE | int | 사용 가능한 자전거 수 |
| NUM_EBIKES_AVAILABLE | int | 사용 가능한 전기자전거 수 |
| NUM_BIKES_DISABLED | int | 비활성 자전거 수 |
| IS_INSTALLED | string | 설치 여부 |
| IS_RETURNING | string | 반납 가능 여부 |
| IS_RENTING | string | 대여 가능 여부 |
| HAS_KIOSK | string | 키오스크 여부 |

---

## 6️⃣ POI 기반 공간 피처 (주변 시설 밀도)

- 지하철역, 버스정류장, 공원, 학교에 대해  
  **최근접 거리 + 반경(100~2000m) 내 개수** 제공

---

## 7️⃣ 날씨 정보 (Weather Features)

| 컬럼명 | 타입 | 설명 |
|------|----|----|
| temperature | float | 기온 (°C) |
| humidity | float | 습도 (%) |
| windspeed | float | 풍속 |
| cloud_cover | float | 구름량 (%) |
| shortwave_radiation | float | 태양복사량 |
| precipitation | float | 강수량 |
| rain | float | 강우량 |
| snowfall | float | 적설량 |
| snow_depth | float | 적설 깊이 |
| weathersit | int | 날씨 상태 코드 |

---

## 🎯 모델링 관점 요약

- **Target 변수**: `rental_count`
- **Feature 축**
  - 시간 기반
  - 공간 기반 (POI)
  - 상황 기반 (날씨)
  - 인프라 상태 기반

➡️ **자전거 수요 예측을 위한 고차원 통합 데이터셋**
