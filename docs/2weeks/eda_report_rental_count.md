# 따릉이 rental_count 예측을 위한 강화된 EDA 보고서
본 EDA는 rental_count 예측 모델 구축을 목표로 수행되었습니다.

## 0. rental_count 및 주요 변수 요약 통계
|       |   rental_count |   quarter_flag |     거치대수 |   temperature |   N_of_stations_within_100m |   N_of_stations_within_500m |        month |   N_of_school_within_2000m |   N_of_school_within_1500m |   used_time(avg) |    used_dis(avg) |
|:------|---------------:|---------------:|-------------:|--------------:|----------------------------:|----------------------------:|-------------:|---------------------------:|---------------------------:|-----------------:|-----------------:|
| count |    1.45816e+07 |    1.45816e+07 |  1.45816e+07 |   1.45816e+07 |                 1.45816e+07 |                 1.45816e+07 |  1.45816e+07 |                1.45816e+07 |                1.45816e+07 |      1.45816e+07 |      1.45815e+07 |
| mean  |   11.9519      |    1.54441     | 12.4319      |  14.2784      |                 0.130506    |                 0.668757    |  6.28918     |               20.8998      |               12.223       |     20.7325      |   2373.57        |
| std   |   15.8485      |    1.13399     |  5.58388     |  10.7083      |                 0.364241    |                 0.822142    |  3.37207     |                7.68575     |                5.27544     |     21.4867      |   1758.17        |
| min   |    1           |    0           |  2           | -17.0833      |                 0           |                 0           |  1           |                0           |                0           |    -13           |      0           |
| 25%   |    3           |    0           | 10           |   5.6         |                 0           |                 0           |  3           |               16           |                8           |     11.2308      |   1329.74        |
| 50%   |    7           |    2           | 10           |  15.5167      |                 0           |                 0           |  6           |               21           |               12           |     17.7419      |   1943.5         |
| 75%   |   15           |    3           | 15           |  23.4167      |                 0           |                 1           |  9           |               26           |               15           |     26           |   2894.62        |
| max   |  932           |    3           | 62           |  35.5667      |                 2           |                 5           | 12           |               48           |               31           |  25284           | 160643           |

---

# 1. 목표변수 rental_count 분석

- rental_count의 전체 분포 및 이상치를 시각적으로 확인함.

# 2. 월 계절 기반 패턴 분석

- 월 계절 별 rental_count 변동 확인.

# 3. 범주형 변수별 rental_count 비교

- season별 rental_count 평균 비교 완료.
- weekend별 rental_count 평균 비교 완료.
# 4. 수치형 변수 상관관계 분석

## rental_count과 상관관계가 높은 상위 10개 변수
- **quarter_flag: 0.220**
- **거치대수: 0.209**
- **temperature: 0.208**
- **N_of_stations_within_100m: 0.110**
- **N_of_stations_within_500m: 0.088**
- **month: 0.067**
- **N_of_school_within_2000m: 0.065**
- **N_of_school_within_1500m: 0.046**
- **used_time(avg): 0.036**
- **used_dis(avg): 0.034**
# 5. 주요 피처 ↔ rental_count 관계 시각화

- quarter_flag ↔ rental_count 회귀 분석 완료
- 거치대수 ↔ rental_count 회귀 분석 완료
- temperature ↔ rental_count 회귀 분석 완료
- N_of_stations_within_100m ↔ rental_count 회귀 분석 완료
- N_of_stations_within_500m ↔ rental_count 회귀 분석 완료
- month ↔ rental_count 회귀 분석 완료
- N_of_school_within_2000m ↔ rental_count 회귀 분석 완료
- N_of_school_within_1500m ↔ rental_count 회귀 분석 완료
- used_time(avg) ↔ rental_count 회귀 분석 완료
- used_dis(avg) ↔ rental_count 회귀 분석 완료