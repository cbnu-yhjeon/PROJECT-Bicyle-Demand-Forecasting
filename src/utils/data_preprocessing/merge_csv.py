import pandas as pd
from pathlib import Path


def safe_read_csv(path: Path) -> pd.DataFrame:
    """여러 인코딩을 시도하면서 CSV를 안전하게 읽는 함수"""
    encodings_to_try = ["utf-8", "cp949", "euc-kr", "latin1"]

    for enc in encodings_to_try:
        try:
            print(f"   ↳ 인코딩 시도: {enc}")
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            print(f"   ⚠ 실패 (인코딩 문제): {enc}")
            continue

    raise RuntimeError(f"❌ 인코딩 문제로 파일을 읽지 못했습니다: {path}")


def merge_csv_files(input_dir, output_path):
    input_dir = Path(input_dir)
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        print("❌ CSV 파일이 없습니다.")
        return

    merged_df = None

    for i, file in enumerate(csv_files):
        print(f"📂 읽는 중: {file.name}")
        df = safe_read_csv(file)

        if i == 0:
            merged_df = df
        else:
            merged_df = pd.concat([merged_df, df], ignore_index=True)

    output_path = Path(output_path)
    if output_path.is_dir():
        output_path = output_path / "merged.csv"

    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 병합 완료: {output_path}")


if __name__ == "__main__":
    merge_csv_files(
        input_dir=r"/home/avg/PROJECT-Bicyle-Demand-Forecasting/Data/raw/seoul/Contextual Data/weather",
        output_path=r"/home/avg/PROJECT-Bicyle-Demand-Forecasting/Data/processed/seoul/Contextual Data/weather"
    )
