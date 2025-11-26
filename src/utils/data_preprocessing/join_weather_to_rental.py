import pandas as pd
from pathlib import Path


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
# 1) 날짜 + 쿼터별 날씨 요약 테이블 생성
# -----------------------------------------
def build_weather_quarter_table(weather_csv: str) -> pd.DataFrame:

    df_w = safe_read_csv(Path(weather_csv))
    df_w.columns = [c.strip() for c in df_w.columns]

    # dt → 날짜 컬럼 생성
    df_w["dt"] = pd.to_datetime(df_w["dt"], errors="coerce")
    df_w["date"] = df_w["dt"].dt.strftime("%Y-%m-%d")

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
        "Precipitation": "sum",
        "snowcover": "max",
        "weathersit": "max"
    }

    # 실제 있는 컬럼만 사용
    agg_rule = {k: v for k, v in agg_rule.items() if k in df_w.columns}

    # 쿼터별 요약
    weather_q = (
        df_w.groupby(["date", "quarter_flag"], as_index=False)
        .agg(agg_rule)
    )

    # ⭐ 소수점 둘째 자리까지 반영
    weather_q = weather_q.round(2)

    print("✅ 날씨 쿼터 요약 테이블 생성 완료 (소수점 둘째자리 적용)")
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

    print("📂 날씨 쿼터 요약 테이블 생성 중...")
    weather_q = build_weather_quarter_table(weather_csv)

    print("📂 대여 데이터 로딩 중...")
    df_r = safe_read_csv(Path(rental_csv))

    # 날짜 형식 통일
    df_r[rental_date_col] = (
        pd.to_datetime(df_r[rental_date_col], errors="coerce")
        .dt.strftime("%Y-%m-%d")
    )

    # 쿼터 타입 통일
    df_r[rental_quarter_col] = df_r[rental_quarter_col].astype(int)
    weather_q["quarter_flag"] = weather_q["quarter_flag"].astype(int)

    # 조인 수행 (LEFT JOIN)
    print("🔗 조인 수행...")
    df_merged = df_r.merge(
        weather_q,
        left_on=[rental_date_col, rental_quarter_col],
        right_on=["date", "quarter_flag"],
        how="left",
        suffixes=("", "_weather"),
    )

    df_merged = df_merged.drop(columns=["date", "quarter_flag"], errors="ignore")

    df_merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 조인 완료: {output_csv}")


# -----------------------------------------
# 실행 예시
# -----------------------------------------
if __name__ == "__main__":
    join_weather_to_rentals(
        weather_csv=(
            r"/mnt/c/projects/PROJECT-Bicyle-Demand-Forecasting/"
            r"Data/processed/seoul/Contextual Data/weather/"
            r"merged_with_time_features_filled_flagged_weathersit.csv"
        ),
        rental_csv=(
            r"/mnt/c/projects/PROJECT-Bicyle-Demand-Forecasting/"
            r"Data/processed/rental_data/"
            r"bicycle rental_info.csv"
        ),
        output_csv=(
            r"/mnt/c/projects/PROJECT-Bicyle-Demand-Forecasting/"
            r"Data/processed/join/"
            r"merged_with_weather.csv"
        ),
        rental_date_col="date",
        rental_quarter_col="quarter of day"
    )
