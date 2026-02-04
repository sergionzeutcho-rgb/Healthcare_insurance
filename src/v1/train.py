"""
Machine Learning Model Training Pipeline
=========================================

This script trains a RandomForest regression model to predict healthcare insurance costs.

What it does:
-------------
1. Loads cleaned data from ETL pipeline
2. Splits data into training (80%) and test (20%) sets
3. Builds preprocessing pipeline (OneHotEncoding for categorical features)
4. Trains RandomForest model with 400 trees
5. Evaluates performance on test set (MAE, RMSE, R²)
6. Saves trained model for use in Streamlit dashboard

Model Architecture:
-------------------
- Algorithm: RandomForestRegressor
- Trees: 400 estimators
- Preprocessing: OneHotEncoding for categorical variables (sex, smoker, region)
- Features: age, sex, bmi, children, smoker, region
- Target: charges (insurance costs in USD)

Performance Metrics:
--------------------
- R² Score: ~88.4% (explains 88.4% of cost variance)
- MAE: ~$2,549 (average prediction error)
- RMSE: ~$4,611 (root mean squared error)

Usage:
------
    python src/v1/train.py

Prerequisites:
--------------
    Must run ETL first: python src/v1/etl.py

Inputs:
-------
    - data/v1/processed/insurance_clean.csv (from ETL pipeline)

Outputs:
--------
    - models/v1/insurance_model.joblib (trained sklearn Pipeline)

Author: Sergio Kadje
Date: February 2026
Project: Healthcare Insurance Cost Analysis (Code Institute Capstone)
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


# ============================================================================
# FILE PATHS
# ============================================================================
# Input: cleaned data from ETL pipeline
# Output: trained model saved as joblib file (can be loaded by Streamlit app)

DATA_PATH = Path("data/v1/processed/insurance_clean.csv")
MODEL_PATH = Path("models/v1/insurance_model.joblib")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def rmse(y_true, y_pred) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    RMSE measures the average magnitude of prediction errors, giving more weight
    to large errors compared to MAE. Useful for understanding typical prediction error.
    
    Parameters
    ----------
    y_true : array-like
        Actual target values (true insurance charges)
    y_pred : array-like
        Predicted target values (model predictions)
    
    Returns
    -------
    float
        RMSE value in same units as target (dollars for this project)
    
    Notes
    -----
    Lower RMSE = better model performance
    RMSE is always ≥ MAE due to squaring of errors
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main() -> None:
    """
    Execute the complete model training pipeline.
    
    Steps:
    ------
    1. Load cleaned data (from ETL)
    2. Define features and target variable
    3. Split into train/test sets (80/20)
    4. Build preprocessing pipeline (OneHotEncoder for categorical features)
    5. Configure RandomForest model
    6. Train model on training data
    7. Evaluate on test data (MAE, RMSE, R²)
    8. Save trained pipeline to disk
    
    Raises
    ------
    FileNotFoundError
        If cleaned data doesn't exist (must run ETL first)
    
    Notes
    -----
    - Uses fixed random_state=42 for reproducibility
    - Pipeline prevents data leakage (preprocessing fit only on training data)
    - Saved model includes preprocessing, so raw data can be passed to predict()
    """
    # --------------------------------------------------
    # STEP 1: Validate Prerequisites
    # --------------------------------------------------
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"❌ Cleaned dataset not found at {DATA_PATH}.\n"
            "   Please run ETL first: python src/v1/etl.py"
        )

    # --------------------------------------------------
    # STEP 2: Load Cleaned Data
    # --------------------------------------------------
    print("📂 Loading cleaned dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df)} rows, {df.shape[1]} columns")

    # --------------------------------------------------
    # STEP 3: Define Target and Features
    # --------------------------------------------------
    # Target variable: insurance charges (what we want to predict)
    target = "charges"
    
    # Feature variables: all factors that might influence insurance costs
    # Note: We use the original 6 features, not the derived ones (age_group, etc.)
    # This is intentional - RandomForest can learn non-linear relationships from raw data
    feature_cols = ["age", "sex", "bmi", "children", "smoker", "region"]

    print(f"🎯 Target: {target}")
    print(f"📊 Features: {', '.join(feature_cols)}")

    # Separate features (X) and target (y)
    X = df[feature_cols].copy()
    y = df[target].copy()

    # --------------------------------------------------
    # STEP 4: Train-Test Split
    # --------------------------------------------------
    # Split data: 80% training, 20% testing
    # random_state=42 ensures reproducibility (same split every time)
    print("✂️  Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training set: {len(X_train)} rows")
    print(f"   Test set: {len(X_test)} rows")

    # --------------------------------------------------
    # STEP 5: Define Feature Types
    # --------------------------------------------------
    # Separate numeric and categorical features for different preprocessing
    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]
    
    print(f"🔢 Numeric features: {', '.join(numeric_features)}")
    print(f"🔤 Categorical features: {', '.join(categorical_features)}")

    # --------------------------------------------------
    # STEP 6: Build Preprocessing Pipeline
    # --------------------------------------------------
    # ColumnTransformer applies different transformations to different columns:
    # - Categorical: OneHotEncoder (converts "male"/"female" → [1,0] or [0,1])
    # - Numeric: Passthrough (no transformation needed for RandomForest)
    print("🔧 Building preprocessing pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            # OneHotEncoder: Creates binary columns for each category
            # handle_unknown="ignore": Safely handles new categories in test/production
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            
            # Passthrough: Keeps numeric features as-is
            # RandomForest doesn't need feature scaling (unlike Linear Regression)
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",  # Drop any columns not specified above
    )
    print("   Configured OneHotEncoder for categorical features")

    # --------------------------------------------------
    # STEP 7: Configure RandomForest Model
    # --------------------------------------------------
    # RandomForestRegressor: Ensemble of 400 decision trees
    # Why 400 trees? Balances performance vs. training time
    # n_jobs=-1: Use all CPU cores for faster training
    print("🌲 Configuring RandomForest model...")
    model = RandomForestRegressor(
        n_estimators=400,    # Number of trees in the forest
        random_state=42,     # For reproducibility
        n_jobs=-1,          # Use all available CPU cores
    )
    print("   Model: RandomForestRegressor with 400 trees")

    # --------------------------------------------------
    # STEP 8: Create Complete Pipeline
    # --------------------------------------------------
    # Pipeline chains preprocessing + model into single object
    # Benefits:
    # 1. Prevents data leakage (preprocessing fit only on training data)
    # 2. Simplifies prediction (raw data → preprocessed → prediction in one step)
    # 3. Easy to save/load entire workflow
    print("🔗 Creating ML pipeline...")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),  # Step 1: Encode categoricals
            ("model", model),               # Step 2: RandomForest prediction
        ]
    )
    print("   Pipeline: preprocessor → RandomForest")

    # --------------------------------------------------
    # STEP 9: Train Model
    # --------------------------------------------------
    # Fit entire pipeline on training data
    # This:
    # 1. Fits OneHotEncoder on training data categorical values
    # 2. Transforms training features
    # 3. Trains RandomForest on transformed features
    print("\n🎓 Training model...")
    print("   (This may take 30-60 seconds with 400 trees)")
    pipeline.fit(X_train, y_train)
    print("   ✅ Training complete!")

    # --------------------------------------------------
    # STEP 10: Evaluate on Test Set
    # --------------------------------------------------
    print("\n📊 Evaluating model performance...")
    y_pred = pipeline.predict(X_test)
    
    # Calculate performance metrics:
    # - MAE: Average absolute prediction error (easier to interpret)
    # - RMSE: Root mean squared error (penalizes large errors more)
    # - R²: Proportion of variance explained (0-1 scale, higher is better)
    mae = mean_absolute_error(y_test, y_pred)
    rmse_score = rmse(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # --------------------------------------------------
    # Performance Report
    # --------------------------------------------------
    print("\n" + "="*50)
    print("📈 MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"MAE (Mean Absolute Error):  ${mae:,.2f}")
    print(f"   → On average, predictions are off by ${mae:,.0f}")
    print(f"\nRMSE (Root Mean Squared Error): ${rmse_score:,.2f}")
    print(f"   → Typical prediction error is ${rmse_score:,.0f}")
    print(f"\nR² Score: {r2:.4f} ({r2*100:.2f}%)")
    print(f"   → Model explains {r2*100:.1f}% of cost variation")
    print("="*50)

    # --------------------------------------------------
    # STEP 11: Save Trained Model
    # --------------------------------------------------
    # Create output directory if it doesn't exist
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save entire pipeline (preprocessing + model) as single file
    # joblib is more efficient than pickle for sklearn models
    print(f"\n💾 Saving model...")
    joblib.dump(pipeline, MODEL_PATH)
    print(f"   ✅ Model saved to: {MODEL_PATH}")
    print(f"   📦 File size: ~{MODEL_PATH.stat().st_size / (1024*1024):.1f} MB")
    
    print("\n🎉 Training pipeline complete!")
    print("   Model is ready for use in Streamlit dashboard.")


if __name__ == "__main__":
    main()
