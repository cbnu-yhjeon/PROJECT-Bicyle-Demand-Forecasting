import pandas as pd
import numpy as np
from pathlib import Path


def add_weathersit_flag(input_csv, output_csv):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    # 1) CSV 읽기
    df = pd.read_csv(input_csv)

    # 2) 사용할 컬럼들 NaN 방어 (있으면 0.0 또는 중립값으로)
    for col, fill_value in [
        ("Precipitation", 0.0),
        ("snowcover", 0.0),
        ("cloudcover", 0.0),
        ("sunshine", 0.5),
        ("humidity", 50.0),
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
        else:
            print(f"⚠ 컬럼 없음: {col}")

    # 편의상 변수로 잡기
    rain = df["Precipitation"]
    snow = df["snowcover"]
    cloud = df["cloudcover"]
    sun = df["sunshine"]
    hum = df["humidity"]

    # ===== 조건 정의 =====

    # 4: Heavy Rain / Snow / Mix
    cond_4 = (
        (rain >= 5.0) |           # 강한 비
        (snow >= 5.0) |           # 눈 많이
        ((rain > 0) & (snow > 0) & (cloud >= 8.0))  # 비+눈+구름 잔뜩
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
            ((hum >= 85.0) & (cloud >= 4.0)) |      # 습도 높고 구름 많은 날
            ((sun <= 0.4) & (cloud >= 6.0))         # 해는 약하고 구름은 많은 날
        )
    )

    # 기본값 1로 채워 두고 조건 순서대로 덮어쓰기 해도 됨
    weathersit = np.ones(len(df), dtype=int)
    weathersit[cond_2] = 2
    weathersit[cond_3] = 3
    weathersit[cond_4] = 4

    df["weathersit"] = weathersit

    # 3) 저장 (기존 컬럼 + weathersit 추가)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ weathersit 추가 완료: {output_csv}")


if __name__ == "__main__":
    add_weathersit_flag(
        input_csv=r"/home/avg/PROJECT-Bicyle-Demand-Forecasting/Data/processed/seoul/Contextual Data/weather/merged_with_time_features.csv",
        output_csv=r"/home/avg/PROJECT-Bicyle-Demand-Forecasting/Data/processed/seoul/Contextual Data/weather/merged_with_time_features_weathersit.csv",
    )
