import pandas as pd
from pathlib import Path


# -----------------------------------------
#  공통: CSV 안전 로더
# -----------------------------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    """여러 인코딩을 시도하면서 CSV를 안전하게 읽는 함수"""
    encodings_to_try = ["utf-8", "cp949", "euc-kr", "latin1"]
    last_err = None

    for enc in encodings_to_try:
        try:
            print(f"   ↳ {path.name} 인코딩 시도: {enc}")
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError as e:
            print(f"   ⚠ 실패 (인코딩 문제): {enc}")
            last_err = e
        except FileNotFoundError:
            print(f"❌ 파일을 찾을 수 없습니다: {path}")
            raise

    raise last_err if last_err else RuntimeError(f"CSV를 읽지 못했습니다: {path}")


# -----------------------------------------
# 1) 날짜 + 쿼터별 날씨 요약 테이블 생성
# -----------------------------------------
def build_weather_quarter_table(weather_csv: str) -> pd.DataFrame:
    """
    날씨 CSV를 읽어서 (join_date, quarter_flag) 단위로 요약한 테이블 생성
    """
    df_w = safe_read_csv(Path(weather_csv))

    # 컬럼 이름 정리 (앞뒤 공백 제거)
    df_w.columns = [c.strip() for c in df_w.columns]

    # dt → datetime, join_date(YYYY-MM-DD) 생성
    if "dt" not in df_w.columns:
        raise KeyError("날씨 데이터에 'dt' 컬럼이 없습니다.")

    df_w["dt"] = pd.to_datetime(df_w["dt"], errors="coerce")
    df_w["join_date"] = df_w["dt"].dt.strftime("%Y-%m-%d")

    if "quarter_flag" not in df_w.columns:
        raise KeyError("날씨 데이터에 'quarter_flag' 컬럼이 없습니다.")

    # 변수별 대표값 전략
    agg_rule = {
        "temperature": "mean",
        "humidity": "mean",
        "windspeed": "mean",
        "atmosphericpressure": "mean",
        "sunshine": "mean",
        "cloudcover": "mean",
        "Precipitation": "sum",   # 쿼터 동안 총 강수량
        "snowcover": "max",       # 쿼터 동안 최대 적설
        "weathersit": "max",      # 가장 나쁜 날씨 상태
    }
    # 실제 존재하는 컬럼만 사용
    agg_rule = {k: v for k, v in agg_rule.items() if k in df_w.columns}

    # (join_date, quarter_flag) 단위 요약
    weather_q = (
        df_w
        .groupby(["join_date", "quarter_flag"], as_index=False)
        .agg(agg_rule)
    )

    # 🔢 소수점 둘째 자리까지 반올림
    weather_q = weather_q.round(2)

    print("✅ 날씨 쿼터 요약 테이블 생성 완료 (round(2) 적용)")
    return weather_q


# -----------------------------------------
# 2) 날씨를 대여 데이터에 JOIN
# -----------------------------------------
def join_weather_to_rentals(
    weather_csv: str,
    rental_csv: str,
    output_csv: str,
    rental_date_col: str = "date",          # 🔹 여기에 'Date'처럼 원본 컬럼명 넣으면 됨
    rental_quarter_col: str = "quarter of day",
    use_quarter_mapping: bool = True,
):
    """
    - weather_csv : 쿼터 플래그/날씨 피처가 들어있는 날씨 CSV
    - rental_csv  : 대여소/일자/쿼터별 대여 정보 CSV
    - output_csv  : 날씨가 join된 결과 CSV

    🔥 중요한 점:
      - rental_date_col 컬럼은 "절대" 안 지움/안 바꿈
      - 조인용으로만 join_date 라는 컬럼을 따로 만들어서 사용
    """
    weather_path = Path(weather_csv)
    rental_path = Path(rental_csv)
    output_path = Path(output_csv)

    # 1) 날씨 쿼터 요약 테이블
    print("📂 날씨 데이터 로딩 및 쿼터 요약...")
    weather_q = build_weather_quarter_table(str(weather_path))

    # 2) 대여 데이터 로딩
    print("📂 대여 데이터 로딩...")
    df_r = safe_read_csv(rental_path)

    # 3) 렌탈 원본 date 컬럼 존재 확인
    if rental_date_col not in df_r.columns:
        raise KeyError(f"대여 데이터에 '{rental_date_col}' 컬럼이 없습니다.")

    # 4) 조인용 join_date 컬럼 따로 생성 (원본 date는 건드리지 않음)
    df_r["join_date"] = (
        pd.to_datetime(df_r[rental_date_col], errors="coerce")
        .dt.strftime("%Y-%m-%d")
    )

    # 5) 쿼터 컬럼 정리 (원본 quarter_of_day는 그대로 두고, 조인용만 사용)
    if rental_quarter_col not in df_r.columns:
        raise KeyError(f"대여 데이터에 '{rental_quarter_col}' 컬럼이 없습니다.")

    df_r[rental_quarter_col] = df_r[rental_quarter_col].astype(int)
    weather_q["quarter_flag"] = weather_q["quarter_flag"].astype(int)

    # 🔁 rental 쿼터(0,1,2,3)를 weather 쿼터(0,2,3,4)로 매핑
    if use_quarter_mapping:
        # 0: 00–05 / 1: 06–11 / 2: 12–17 / 3: 18–23 (rental)
        # 0: 00–05 / 2: 06–11 / 3: 12–17 / 4: 18–23 (weather)
        mapping = {0: 0, 1: 1, 2: 2, 3: 3}
        df_r["quarter_join"] = df_r[rental_quarter_col].map(mapping)
        join_left_keys = ["join_date", "quarter_join"]
    else:
        # rental 쿼터 값이 weather_quarter랑 이미 동일한 경우
        join_left_keys = ["join_date", rental_quarter_col]

    # 6) LEFT JOIN (대여 기준으로 날씨 붙이기)
    print("🔗 조인 수행 (LEFT JOIN)...")
    df_merged = df_r.merge(
        weather_q,
        left_on=join_left_keys,
        right_on=["join_date", "quarter_flag"],
        how="left",
        suffixes=("", "_weather"),
    )

    # 🔥 여기서 원본 rental_date_col은 건드리지 않는다
    #    조인용으로 만든 join_date / quarter_flag 만 정리
    df_merged = df_merged.drop(columns=["join_date", "quarter_flag", "quarter_join"], errors="ignore")

    # 7) 저장
    df_merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 조인 완료: {output_path}")
    print(f"✅ 최종 컬럼 목록: {list(df_merged.columns)}")


# -----------------------------------------
#  실행 예시
# -----------------------------------------
if __name__ == "__main__":
    join_weather_to_rentals(
        weather_csv=(
            r"/home/avg/PROJECT-Bicyle-Demand-Forecasting/"
            r"Data/processed/seoul/Contextual Data/weather/"
            r"merged_with_time_features_weathersit.csv"
        ),
        rental_csv=(
            r"/home/avg/PROJECT-Bicyle-Demand-Forecasting/"
            r"Data/processed/rental_data/"
            r"bicycle rental_info.csv"
        ),
        output_csv=(
            r"/home/avg/PROJECT-Bicyle-Demand-Forecasting/"
            r"Data/processed/join/"
            r"merged_with_weather.csv"
        ),
        # 🔴 여기 컬럼명은 "렌탈 CSV에 실제로 적힌 이름"을 써야 한다
        #    만약 렌탈 CSV에 'Date' 라고 되어 있으면 이렇게 써:
        #    rental_date_col="Date",
        rental_date_col="date",
        rental_quarter_col="quarter of day",
        use_quarter_mapping=True,
    )
