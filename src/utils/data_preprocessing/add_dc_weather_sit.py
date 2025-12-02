import pandas as pd
import numpy as np
from pathlib import Path


def _pick_first_existing_column(df: pd.DataFrame, candidates, default_name=None):
    """
    candidates 리스트 중에서 실제 df에 존재하는 첫 번째 컬럼명을 찾아서 반환.
    없으면 default_name 반환 (또는 None).
    """
    for c in candidates:
        if c in df.columns:
            return c
    return default_name


def add_weathersit_flag(input_csv, output_csv):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    # 1) CSV 읽기
    df = pd.read_csv(input_csv)

    # ---------------------------
    # 2) 컬럼 매핑 (이름 자동 매칭)
    # ---------------------------
    col_precip = _pick_first_existing_column(
        df,
        ["Precipitation", "precipitation", "rain"],
    )
    col_snow = _pick_first_existing_column(
        df,
        ["snowcover", "snow_depth", "snowfall"],
    )
    col_cloud = _pick_first_existing_column(
        df,
        ["cloudcover", "cloud_cover"],
    )
    col_sun = _pick_first_existing_column(
        df,
        ["sunshine", "shortwave_radiation"],
    )
    col_hum = _pick_first_existing_column(
        df,
        ["humidity", "relative_humidity_2m"],
    )

    # 존재 여부 출력 (디버깅용)
    print("📌 Column mapping:")
    print(f"  Precipitation  -> {col_precip}")
    print(f"  Snowcover      -> {col_snow}")
    print(f"  Cloudcover     -> {col_cloud}")
    print(f"  Sunshine       -> {col_sun}")
    print(f"  Humidity       -> {col_hum}")

    # 필수 컬럼 체크 (비/눈/구름/습도는 있어야 날씨 구분이 의미 있음)
    required = {
        "Precipitation": col_precip,
        "Snow/SnowDepth": col_snow,
        "Cloudcover": col_cloud,
        "Humidity": col_hum,
    }
    missing_required = [k for k, v in required.items() if v is None]
    if missing_required:
        raise RuntimeError(f"❌ 필수 날씨 컬럼 부족: {missing_required} (input columns={list(df.columns)})")

    # ---------------------------
    # 3) NaN 방어 & 값 가져오기
    # ---------------------------
    # 기본 채움값은 네가 준 예시 기준 + 단위 감안해서 조금 조정
    df[col_precip] = df[col_precip].fillna(0.0)
    df[col_snow] = df[col_snow].fillna(0.0)
    df[col_cloud] = df[col_cloud].fillna(0.0)
    df[col_hum] = df[col_hum].fillna(50.0)

    # sunshine / shortwave_radiation 처리
    if col_sun is not None:
        df[col_sun] = df[col_sun].fillna(df[col_sun].median())
    else:
        # 컬럼이 아예 없으면 중간값 비슷한 0.5로 가정
        df["__fake_sun"] = 0.5
        col_sun = "__fake_sun"

    # 편의상 변수로 잡기
    rain = df[col_precip].astype(float)
    snow = df[col_snow].astype(float)
    cloud = df[col_cloud].astype(float)
    hum = df[col_hum].astype(float)

    # sunshine/shortwave_radiation: 단위가 다를 수 있으니 0~1로 정규화
    sun_raw = df[col_sun].astype(float)

    # 만약 이미 0~1 범위라면 그대로 쓰고, 아니라면 (예: 0~1000 W/m²) 스케일링
    if sun_raw.max() <= 1.5:
        sun = sun_raw.clip(0.0, 1.0)
    else:
        # 대략 0~1000 W/m² 가정, 1000으로 나눠 0~1로 스케일링
        sun = (sun_raw / 1000.0).clip(0.0, 1.0)

    # ---------------------------
    # 4) 조건 정의 (weathersit)
    # ---------------------------
    # cloud는 0~100% 기준으로 설정 가정
    #   - "매우 흐림" ~ 80% 이상
    #   - "다소 흐림" ~ 50% 이상
    cloud_very_high = cloud >= 80.0
    cloud_high = cloud >= 50.0

    # 4: Heavy Rain / Snow / Mix
    cond_4 = (
        (rain >= 5.0) |           # 강한 비
        (snow >= 5.0) |           # 눈 많이
        ((rain > 0) & (snow > 0) & cloud_very_high)  # 비+눈+구름 잔뜩
    )

    # 3: Light Rain / Light Snow
    cond_3 = (
        (rain > 0) | (snow > 0)
    ) & (~cond_4)  # 4번에 해당되는 건 제외

    # 2: Mist / Cloudy 계열 (비/눈은 없고, 습하고 구름 많은 날)
    cond_2 = (
        (rain == 0) &
        (snow == 0) &
        (
            ((hum >= 85.0) & cloud_high) |      # 습도 높고 구름 많은 날
            ((sun <= 0.4) & cloud_very_high)    # 해는 약하고 구름은 많은 날
        )
    )

    # 기본값 1로 채워 두고 조건 순서대로 덮어쓰기
    weathersit = np.ones(len(df), dtype=int)
    weathersit[cond_2] = 2
    weathersit[cond_3] = 3
    weathersit[cond_4] = 4

    df["weathersit"] = weathersit

    # 5) 저장
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ weathersit 추가 완료: {output_csv}")


if __name__ == "__main__":
    # 예시: Open-Meteo 결과 CSV에 weathersit 추가
    add_weathersit_flag(
        input_csv=r"/home/avg/PROJECT-Bicyle-Demand-Forecasting/Data/interim/washington/dc_weather_2021_2025_hourly_with_time_features.csv",
        output_csv=r"/home/avg/PROJECT-Bicyle-Demand-Forecasting/Data/interim/washington/dc_weather.csv",
    )
