from pathlib import Path
import pandas as pd

def safe_read_csv(path: Path) -> pd.DataFrame:
    """여러 인코딩 시도 후, 마지막에는 깨진 문자 무시하고 강제 로딩"""
    encodings_to_try = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]

    for enc in encodings_to_try:
        try:
            print(f"  ↳ {path.name} 인코딩 시도: {enc}")
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            print(f"    ⚠ {enc} 실패 (UnicodeDecodeError)")

    # 모든 인코딩 시도 실패 → 깨진 문자 무시하고 강제 로딩
    print(f"    ⚠ 모든 기본 인코딩 실패 → errors='ignore'로 강제 로딩: {path.name}")
    with open(path, "r", encoding="cp949", errors="ignore") as f:
        return pd.read_csv(f, low_memory=False)

def main():
    # 이 스크립트 위치: .../src
    script_dir = Path(__file__).resolve().parent
    # 프로젝트 루트: .../PROJECT-Bicyle-Demand-Forecasting
    project_root = script_dir.parent

    # 🔥 진짜 타겟 디렉토리 (리눅스 스타일 경로 + 슬래시로 나눠서 조립)
    target_dir = (
        project_root
        / "Data"
        / "raw"
        / "seoul"
        / "Time-series Data"
        / "Rental_Bike_usage"
        / "20~25"            # ← 여기 숫자만 20,21,22... 바꿔주면 됨
    )

    print("📂 타겟 디렉토리:", target_dir)

    csv_files = sorted(target_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if f.name != "merged.csv"]

    if not csv_files:
        print("❌ CSV 없음")
        return

    print(f"발견한 CSV {len(csv_files)}개")
    df = safe_read_csv(csv_files[0])
    merged = df.copy()

    for f in csv_files[1:]:
        print("➡ 병합 중:", f.name)
        df_tmp = safe_read_csv(f)
        if df_tmp.shape[1] != merged.shape[1]:
            print(" ⚠ 컬럼 수 다름 → 스킵:", f.name)
            continue
        df_tmp.columns = merged.columns
        merged = pd.concat([merged, df_tmp], ignore_index=True)

    output = target_dir / "merged.csv"
    merged.to_csv(output, index=False, encoding="utf-8-sig")
    print("✅ 완료 →", output)


if __name__ == "__main__":
    main()
