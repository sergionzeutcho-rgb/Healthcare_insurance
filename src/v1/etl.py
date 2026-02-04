"""
ETL Pipeline for Healthcare Insurance Data
============================================

This script performs Extract, Transform, Load (ETL) operations on the raw insurance dataset.

What it does:
-------------
1. Loads raw data from CSV
2. Cleans and standardizes data (removes duplicates, handles missing values)
3. Adds derived features for analysis (age groups, BMI categories)
4. Saves cleaned dataset for downstream analysis and modeling

Usage:
------
    python src/v1/etl.py

Inputs:
-------
    - data/v1/raw/insurance.csv (raw Kaggle dataset)

Outputs:
--------
    - data/v1/processed/insurance_clean.csv (cleaned dataset with derived features)

Author: Sergio Kadje
Date: February 2026
Project: Healthcare Insurance Cost Analysis (Code Institute Capstone)
"""

from pathlib import Path
import pandas as pd


# ============================================================================
# FILE PATHS
# ============================================================================
# These paths use relative paths from project root
# Assumes script is run from: Healthcare_insurance/ directory

RAW_PATH = Path("data/v1/raw/insurance.csv")          # Input: raw data from Kaggle
OUT_PATH = Path("data/v1/processed/insurance_clean.csv")  # Output: cleaned data


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from existing columns.
    
    Adds three new columns to help with analysis:
    1. age_group: Categorical age ranges (e.g., "18-25", "26-35")
    2. bmi_category: WHO-based BMI classification (underweight, normal, overweight, obese)
    3. is_parent: Binary flag indicating if policyholder has children (1 = yes, 0 = no)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with at least 'age', 'bmi', and 'children' columns
    
    Returns
    -------
    pd.DataFrame
        Original dataframe with three additional feature columns
    
    Examples
    --------
    >>> df = pd.DataFrame({'age': [30], 'bmi': [27.5], 'children': [2]})
    >>> df_with_features = add_features(df)
    >>> df_with_features['age_group'].iloc[0]
    '26-35'
    >>> df_with_features['bmi_category'].iloc[0]
    'overweight'
    >>> df_with_features['is_parent'].iloc[0]
    1
    """
    # --------------------------------------------------
    # Feature 1: Age Groups
    # --------------------------------------------------
    # Bin ages into meaningful life stages
    # Bins: (17, 25], (25, 35], (35, 45], (45, 55], (55, 65], (65, 100]
    bins = [17, 25, 35, 45, 55, 65, 100]
    labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

    # --------------------------------------------------
    # Feature 2: BMI Categories (WHO Classification)
    # --------------------------------------------------
    # Based on World Health Organization BMI ranges:
    #   - Underweight: BMI < 18.5
    #   - Normal: 18.5 ≤ BMI < 25
    #   - Overweight: 25 ≤ BMI < 30
    #   - Obese: BMI ≥ 30
    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ["underweight", "normal", "overweight", "obese"]
    df["bmi_category"] = pd.cut(df["bmi"], bins=bmi_bins, labels=bmi_labels)

    # --------------------------------------------------
    # Feature 3: Parental Status
    # --------------------------------------------------
    # Binary indicator: 1 if policyholder has 1+ children, 0 otherwise
    # Useful for analyzing impact of dependents on insurance costs
    df["is_parent"] = (df["children"] > 0).astype(int)
    
    return df


# ============================================================================
# MAIN ETL PIPELINE
# ============================================================================

def main() -> None:
    """
    Execute the complete ETL pipeline.
    
    Steps:
    ------
    1. Load raw CSV data
    2. Remove duplicate rows
    3. Standardize text fields (lowercase, strip whitespace)
    4. Ensure correct data types for numeric columns
    5. Handle missing values (drop rows with nulls in critical columns)
    6. Add derived features (age_group, bmi_category, is_parent)
    7. Save cleaned data to processed/ folder
    
    Raises
    ------
    FileNotFoundError
        If raw data file doesn't exist at RAW_PATH
    
    Notes
    -----
    - This dataset typically has no missing values, but we handle them defensively
    - All categorical text is normalized to lowercase for consistency
    - Creates output directory if it doesn't exist
    """
    # --------------------------------------------------
    # STEP 1: Load Raw Data
    # --------------------------------------------------
    print("📂 Loading raw data...")
    df = pd.read_csv(RAW_PATH)
    print(f"   Loaded {len(df)} rows, {df.shape[1]} columns")

    # --------------------------------------------------
    # STEP 2: Remove Duplicates
    # --------------------------------------------------
    print("🧹 Removing duplicates...")
    rows_before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = rows_before - len(df)
    print(f"   Removed {duplicates_removed} duplicate rows")

    # --------------------------------------------------
    # STEP 3: Standardize Text Fields
    # --------------------------------------------------
    # Normalize categorical variables:
    # - Convert to string (defensive)
    # - Strip leading/trailing whitespace
    # - Convert to lowercase for consistency
    print("🔤 Standardizing text fields...")
    for col in ["sex", "smoker", "region"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
    print("   Normalized: sex, smoker, region")

    # --------------------------------------------------
    # STEP 4: Ensure Correct Data Types
    # --------------------------------------------------
    # Convert to numeric, coercing errors to NaN (defensive)
    # Int64 allows nullable integers (better than int64 for data with potential nulls)
    print("🔢 Converting to numeric types...")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["children"] = pd.to_numeric(df["children"], errors="coerce").astype("Int64")
    df["charges"] = pd.to_numeric(df["charges"], errors="coerce")
    print("   Converted: age, bmi, children, charges")

    # --------------------------------------------------
    # STEP 5: Handle Missing Values
    # --------------------------------------------------
    # Drop rows with missing values in critical columns
    # Note: This Kaggle dataset typically has no missing values,
    # but we implement this defensively for robustness
    print("🧼 Handling missing values...")
    rows_before = len(df)
    df = df.dropna(subset=["age", "bmi", "children", "charges", "sex", "smoker", "region"])
    missing_removed = rows_before - len(df)
    print(f"   Removed {missing_removed} rows with missing values")

    # --------------------------------------------------
    # STEP 6: Add Derived Features
    # --------------------------------------------------
    print("✨ Engineering features...")
    df = add_features(df)
    print("   Added: age_group, bmi_category, is_parent")

    # --------------------------------------------------
    # STEP 7: Save Cleaned Dataset
    # --------------------------------------------------
    # Create output directory if it doesn't exist
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # --------------------------------------------------
    # Summary Report
    # --------------------------------------------------
    print("\n" + "="*50)
    print("✅ ETL COMPLETE")
    print("="*50)
    print(f"📁 Output file: {OUT_PATH}")
    print(f"📊 Final dataset: {len(df)} rows × {df.shape[1]} columns")
    print(f"🎯 Ready for analysis and modeling!")
    print("="*50)


if __name__ == "__main__":
    main()
