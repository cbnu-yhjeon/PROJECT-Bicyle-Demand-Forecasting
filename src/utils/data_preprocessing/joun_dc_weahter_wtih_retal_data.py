import pandas as pd
from pathlib import Path


# -----------------------------------------
#  공통: CSV 안전 로더
# -----------------------------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
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
# 1) 시간별 날씨 → 날짜+쿼터별 요약 테이블 생성
# -----------------------------------------
def build_weather_quarter_table(
    weather_csv: str,
    time_col: str = "time",
):
    """
    컬럼 구조:
    time, temperature, humidity, precipitation, rain, snowfall, snow_depth,
    cloud_cover, windspeed, shortwave_radiation,
    year, month, day, hour, quarter_flag, weathersit
    """

    df_w = safe_read_csv(Path(weather_csv))

    df_w.columns = [c.strip() for c in df_w.columns]

    if time_col not in df_w.columns:
        raise KeyError(f"날씨 데이터에 '{time_col}' 컬럼이 없습니다.")

    # time → datetime
    df_w[time_col] = pd.to_datetime(df_w[time_col], errors="coerce")
    if df_w[time_col].isna().all():
        raise RuntimeError(f"'{time_col}'을 datetime으로 변환하지 못했습니다.")

    # join_date 생성
    df_w["join_date"] = df_w[time_col].dt.strftime("%Y-%m-%d")

    # quarter_flag 는 이미 있음 (0~3)
    if "quarter_flag" not in df_w.columns:
        raise KeyError("날씨 데이터에 'quarter_flag'가 없습니다.")

    df_w["quarter_flag"] = df_w["quarter_flag"].astype(int)

    # 집계 규칙
    agg_rule = {
        "temperature": "mean",
        "humidity": "mean",
        "windspeed": "mean",
        "cloud_cover": "mean",
        "shortwave_radiation": "mean",
        "precipitation": "sum",
        "rain": "sum",
        "snowfall": "sum",
        "snow_depth": "max",
        "weathersit": "max",
    }

    # 존재하는 컬럼만 사용
    agg_rule = {k: v for k, v in agg_rule.items() if k in df_w.columns}

    weather_q = (
        df_w.groupby(["join_date", "quarter_flag"], as_index=False)
             .agg(agg_rule)
    ).round(2)

    print("✅ 날씨 쿼터 요약 테이블 생성 완료")
    print(f"   · 행 수: {len(weather_q)}")
    print(f"   · 컬럼: {list(weather_q.columns)}")

    return weather_q


# -----------------------------------------
# 2) 날씨를 대여 데이터에 JOIN
# -----------------------------------------
def join_weather_to_rentals(
    weather_csv: str,
    rental_csv: str,
    output_csv: str,
    rental_date_col: str = "date",
    rental_quarter_col: str = "quarter of day",
):
    """
    rental CSV: DC 자전거 대여 데이터
    weather CSV: 위에서 만든 DC weather 요약 데이터
    """

    weather_path = Path(weather_csv)
    rental_path = Path(rental_csv)
    output_path = Path(output_csv)

    # 1) 날씨 쿼터 요약 생성
    print("📂 날씨 데이터 로딩 & 쿼터 요약...")
    weather_q = build_weather_quarter_table(weather_csv, time_col="time")

    # 2) 렌탈 데이터 로딩
    print("📂 렌탈 데이터 로딩...")
    df_r = safe_read_csv(rental_path)

    # date 확인
    if rental_date_col not in df_r.columns:
        raise KeyError(f"대여 데이터에 '{rental_date_col}' 컬럼이 없습니다.")

    # 조인용 join_date 생성 (원본 date 유지)
    df_r["join_date"] = (
        pd.to_datetime(df_r[rental_date_col], errors="coerce")
        .dt.strftime("%Y-%m-%d")
    )

    # quarter 컬럼 확인
    if rental_quarter_col not in df_r.columns:
        raise KeyError(f"대여 데이터에 '{rental_quarter_col}' 컬럼이 없습니다.")

    df_r[rental_quarter_col] = df_r[rental_quarter_col].astype(int)
    weather_q["quarter_flag"] = weather_q["quarter_flag"].astype(int)

    # JOIN KEY (동일 구조)
    join_left_keys = ["join_date", rental_quarter_col]

    print("🔗 LEFT JOIN 수행 (렌탈 기준)...")
    df_merged = df_r.merge(
        weather_q,
        left_on=join_left_keys,
        right_on=["join_date", "quarter_flag"],
        how="left",
        suffixes=("", "_weather"),
    )

    # 조인용 컬럼 제거
    df_merged = df_merged.drop(
        columns=["join_date", "quarter_flag"],
        errors="ignore"
    )

    df_merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ 조인 완료 → {output_path}")
    print(f"   · 최종 행 수: {len(df_merged)}")
    print(f"   · 최종 컬럼 수: {len(df_merged.columns)}")


# -----------------------------------------
# 실행 예시
# -----------------------------------------
if __name__ == "__main__":
    join_weather_to_rentals(
        weather_csv=(
            r"/home/avg/PROJECT-Bicyle-Demand-Forecasting/"
            r"Data/interim/washington/"
            r"dc_weather.csv"
        ),
        rental_csv=(
            r"/home/avg/PROJECT-Bicyle-Demand-Forecasting/"
            r"Data/interim/washington/"
            r"bicycle_final_analysis_WDC.csv"
        ),
        output_csv=(
            r"/home/avg/PROJECT-Bicyle-Demand-Forecasting/"
            r"Data/interim/washington/"
            r"dc_rental_data.csv"
        ),
        rental_date_col="date",
        rental_quarter_col="quarter of day",
    )
