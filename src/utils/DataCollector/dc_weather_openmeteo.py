import requests
import pandas as pd
from datetime import datetime

# -----------------------------
# 1) 공통 설정
# -----------------------------
LAT = 38.9072   # Washington, DC
LON = -77.0369

START_YEAR = 2021
END_YEAR = 2025  # 2025년은 6월 30일까지

# Open-Meteo Historical API 엔드포인트
BASE_URL = "https://archive-api.open-meteo.com/v1/era5"

# 우리가 받고 싶은 hourly 변수들
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
]

OUTPUT_CSV = "dc_weather_2021_2025_hourly.csv"


def fetch_yearly_data(year: int) -> pd.DataFrame:
    """
    해당 연도의 데이터를 Open-Meteo API에서 받아서 pandas DataFrame으로 반환.
    2025년은 6월 30일까지, 나머지는 1년 전체.
    """
    if year == 2025:
        start_date = f"{year}-01-01"
        end_date = "2025-06-30"
    else:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        # 시간대 설정: 모델 기본은 UTC, 필요하면 여기서 timezone="America/New_York" 등으로 변경 가능
        "timezone": "UTC",
    }

    print(f"📡 Fetching {year} data: {start_date} ~ {end_date}")
    resp = requests.get(BASE_URL, params=params, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"❌ API 요청 실패 ({year}): {resp.status_code} {resp.text[:200]}")

    data = resp.json()

    # JSON 구조에서 hourly 데이터 꺼내기
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        raise RuntimeError(f"❌ {year} 데이터에 'hourly.time'이 비어 있음")

    df = pd.DataFrame({"time": times})
    for var in HOURLY_VARS:
        df[var] = hourly.get(var, [None] * len(times))

    # time을 datetime으로 변환
    df["time"] = pd.to_datetime(df["time"])

    return df


def main():
    all_df_list = []

    for y in range(START_YEAR, END_YEAR + 1):
        df_year = fetch_yearly_data(y)
        all_df_list.append(df_year)

    # 모두 concat
    df_all = pd.concat(all_df_list, ignore_index=True)

    # 시간 정렬
    df_all = df_all.sort_values("time").reset_index(drop=True)

    # 원하는 컬럼 순서로 정리
    cols = ["time"] + HOURLY_VARS
    df_all = df_all[cols]

    # 2021-01-01 00:00 ~ 2025-06-30 23:00까지만 필터 (UTC 기준)
    start_dt = datetime(2021, 1, 1, 0, 0)
    end_dt = datetime(2025, 6, 30, 23, 0)
    df_all = df_all[(df_all["time"] >= start_dt) & (df_all["time"] <= end_dt)]

    # CSV로 저장
    df_all.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved to {OUTPUT_CSV}")
    print(df_all.head())
    print(df_all.tail())


if __name__ == "__main__":
    main()
