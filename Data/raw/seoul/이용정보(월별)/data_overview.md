# 🚲 공공자전거 이용정보(월별) – 출력 스키마

## 📌 공통 필드
| 구분 | 필드명 | 설명 |
|------|---------|--------|
| 공통 | list_total_count | 총 데이터 건수(정상 조회 시 출력됨) |
| 공통 | RESULT.CODE | 요청결과 코드 |
| 공통 | RESULT.MESSAGE | 요청결과 메시지 |

---

## 📂 월별 이용정보 필드 목록
| 번호 | 필드명 | 설명 |
|------|---------|--------|
| 1 | RENT_NM | 년월(YYYYMMDD) |
| 2 | RENT_TYPE | 대여종류 |
| 3 | STATION_NO | 대여소번호 |
| 4 | STATION_NAME | 대여소명 |
| 5 | GENDER_CD | 성별 |
| 6 | AGE_TYPE | 연령 |
| 7 | USE_CNT | 건수 |
| 8 | EXER_AMT | 운동량 |
| 9 | CARBON_AMT | 탄소절감량 |
| 10 | MOVE_METER | 이동거리(M) |
| 11 | MOVE_TIME | 이용시간(분) |
