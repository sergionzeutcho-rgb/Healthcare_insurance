from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/v1/raw/insurance.csv")
OUT_PATH = Path("data/v1/processed/insurance_clean.csv")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Age group
    bins = [17, 25, 35, 45, 55, 65, 100]
    labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

    # BMI category (standard WHO-ish)
    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ["underweight", "normal", "overweight", "obese"]
    df["bmi_category"] = pd.cut(df["bmi"], bins=bmi_bins, labels=bmi_labels)

    df["is_parent"] = (df["children"] > 0).astype(int)
    return df


def main() -> None:
    df = pd.read_csv(RAW_PATH)

    # Basic quality checks
    df = df.drop_duplicates()

    # Standardise text fields
    for col in ["sex", "smoker", "region"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Ensure numeric types
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["children"] = pd.to_numeric(df["children"], errors="coerce").astype("Int64")
    df["charges"] = pd.to_numeric(df["charges"], errors="coerce")

    # Handle missing (dataset usually has none, but we handle it properly)
    df = df.dropna(subset=["age", "bmi", "children", "charges", "sex", "smoker", "region"])

    df = add_features(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("✅ ETL complete")
    print(f"Saved cleaned dataset to: {OUT_PATH}")
    print("Rows:", len(df), "Columns:", df.shape[1])


if __name__ == "__main__":
    main()
