# 🚲 서울시 따릉이 **대여소 정보 API – 스키마 설명**

다음은 서울시 열린데이터광장(따릉이) **대여소 정보 API**의 출력값(Output) 스키마를 정리한 표입니다.

---

## 📌 공통 출력 필드(Common Fields)

| 구분 | 필드명 | 설명 |
|------|--------|------|
| 공통 | `list_total_count` | 총 데이터 건수 (정상 조회 시 출력) |
| 공통 | `RESULT.CODE` | 요청 결과 코드 (하단 메시지 설명 참고) |
| 공통 | `RESULT.MESSAGE` | 요청 결과 메시지 |

---

## 📌 대여소 정보 필드(Station Metadata Fields)

| 번호 | 필드명 | 설명 |
|------|--------|------|
| 1 | `STA_LOC` | 대여소 그룹명 |
| 2 | `RENT_ID` | 대여소 ID |
| 3 | `RENT_NO` | 대여소 번호 |
| 4 | `RENT_NM` | 대여소 이름 |
| 5 | `RENT_ID_NM` | 대여소 번호명 |
| 6 | `HOLD_NUM` | 거치대 수(총 거치 가능 대수) |
| 7 | `STA_ADD1` | 주소1 |
| 8 | `STA_ADD2` | 주소2 |
| 9 | `STA_LAT` | 대여소 위도 (latitude) |
| 10 | `STA_LONG` | 대여소 경도 (longitude) |
