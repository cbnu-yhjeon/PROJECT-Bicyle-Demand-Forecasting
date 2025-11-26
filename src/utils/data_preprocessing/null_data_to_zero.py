import pandas as pd
from pathlib import Path


def fill_nulls_with_zero(
    input_csv,
    output_csv,
    columns_to_fill_zero,
):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    # 1) CSV 읽기
    df = pd.read_csv(input_csv)

    # 2) 지정된 컬럼들에 대해 NaN → 0.0
    for col in columns_to_fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
            print(f"✅ 컬럼 '{col}' 의 NaN을 0.0으로 채움")
        else:
            print(f"⚠ 컬럼 없음: '{col}' (스킵)")

    # 3) 저장
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"💾 저장 완료: {output_csv}")


if __name__ == "__main__":
    fill_nulls_with_zero(
        input_csv=r"/mnt/c/projects/PROJECT-Bicyle-Demand-Forecasting/Data/processed/seoul/Contextual Data/weather/merged_with_time_features.csv",
        output_csv=r"/mnt/c/projects/PROJECT-Bicyle-Demand-Forecasting/Data/processed/seoul/Contextual Data/weather/merged_with_time_features_filled.csv",
        # 🔽 NaN을 0.0으로 채우고 싶은 컬럼들
        columns_to_fill_zero=[
            "Ob_ID",
            "Ob_NM",
            "dt",
            "temperature",
            "Precipitation",
            "windspeed",
            "atmosphericpressure",
            "humidity",
            "snowcover",
            "sunshine",
            "cloudcover",

        ],
    )

