import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Healthcare Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# Load model and data
@st.cache_resource
def load_model():
    model_path = Path("models/v1/insurance_model.joblib")
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_data
def load_data():
    data_path = Path("data/v1/processed/insurance_clean.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    return None

model = load_model()
df = load_data()

# Title and intro
st.title("🏥 Healthcare Insurance Cost Analysis & Prediction")
st.markdown("""
This dashboard provides insights into healthcare insurance costs and allows you to predict charges based on client attributes.
Based on analysis of 1,337 insurance records.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Data Overview", "🔬 Analysis Insights", "🎯 Cost Predictor"])

# ===== PAGE 1: DATA OVERVIEW =====
if page == "📊 Data Overview":
    st.header("Dataset Overview")
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Average Charge", f"${df['charges'].mean():,.0f}")
        with col3:
            st.metric("Median Charge", f"${df['charges'].median():,.0f}")
        with col4:
            st.metric("Max Charge", f"${df['charges'].max():,.0f}")
        
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Charge Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['charges'], bins=40, color='skyblue', edgecolor='black')
        ax.set_xlabel('Charges ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Insurance Charges')
        st.pyplot(fig)
        
        st.subheader("Charges by Smoker Status")
        fig, ax = plt.subplots(figsize=(8, 5))
        smoker_data = df.groupby('smoker')['charges'].mean().sort_values(ascending=False)
        ax.bar(smoker_data.index, smoker_data.values, color=['coral', 'lightgreen'], edgecolor='black')
        ax.set_ylabel('Average Charges ($)')
        ax.set_title('Average Charges by Smoker Status')
        st.pyplot(fig)
    else:
        st.error("Data not found. Please ensure data/v1/processed/insurance_clean.csv exists.")

# ===== PAGE 2: ANALYSIS INSIGHTS =====
elif page == "🔬 Analysis Insights":
    st.header("Statistical Analysis & Key Findings")
    
    st.subheader("Hypothesis Test Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Hypothesis A: Smoker Impact**
        
        - **Test:** Welch's t-test
        - **Result:** t = 32.74, p < 0.001
        - **Conclusion:** ✅ Smokers have significantly higher charges
        
        Smoker status is the **strongest driver** of insurance costs.
        """)
        
        st.info("""
        **Hypothesis B: Regional Differences**
        
        - **Test:** One-way ANOVA
        - **Result:** F = 2.93, p = 0.033
        - **Conclusion:** ✅ Significant regional differences exist
        
        Regional variation is modest but statistically significant.
        """)
    
    with col2:
        st.info("""
        **Hypothesis C: BMI Impact (Controlled)**
        
        - **Test:** OLS Regression
        - **Result:** Coefficient = $339.25, p < 0.001
        - **Conclusion:** ✅ BMI is a significant predictor
        
        Even after controlling for age, smoker status, and other factors, BMI remains a significant positive predictor.
        """)
        
        st.success("""
        **Machine Learning Model Performance**
        
        - **Algorithm:** Random Forest Regressor
        - **R² Score:** 0.8843 (88.4% variance explained)
        - **MAE:** $2,549
        - **RMSE:** $4,611
        - **Cross-Validation:** Mean R² = 0.8386 ± 0.0333
        
        The model demonstrates strong, reliable predictive performance.
        """)
    
    if df is not None:
        st.subheader("BMI vs Charges")
        fig, ax = plt.subplots(figsize=(10, 5))
        smokers = df[df['smoker'] == 'yes']
        non_smokers = df[df['smoker'] == 'no']
        ax.scatter(non_smokers['bmi'], non_smokers['charges'], alpha=0.5, label='Non-smoker', s=30)
        ax.scatter(smokers['bmi'], smokers['charges'], alpha=0.5, label='Smoker', s=30, color='coral')
        ax.set_xlabel('BMI')
        ax.set_ylabel('Charges ($)')
        ax.set_title('BMI vs Charges (by Smoker Status)')
        ax.legend()
        st.pyplot(fig)

# ===== PAGE 3: COST PREDICTOR =====
elif page == "🎯 Cost Predictor":
    st.header("Predict Insurance Costs")
    
    if model is not None:
        st.markdown("""
        Enter client information below to get an estimated insurance charge prediction.
        This uses the trained Random Forest model (R² = 0.8843).
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", min_value=18, max_value=64, value=30, step=1)
            bmi = st.slider("BMI", min_value=15.0, max_value=55.0, value=25.0, step=0.1)
            children = st.selectbox("Number of Children", options=[0, 1, 2, 3, 4, 5], index=0)
        
        with col2:
            sex = st.selectbox("Sex", options=["male", "female"])
            smoker = st.selectbox("Smoker Status", options=["no", "yes"])
            region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])
        
        if st.button("🔮 Predict Insurance Cost", type="primary", use_container_width=True):
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [sex],
                'bmi': [bmi],
                'children': [children],
                'smoker': [smoker],
                'region': [region]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.success(f"### Estimated Insurance Cost: ${prediction:,.2f}")
            
            # Show comparison
            if df is not None:
                avg_charge = df['charges'].mean()
                diff = prediction - avg_charge
                diff_pct = (diff / avg_charge) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Your Prediction", f"${prediction:,.0f}")
                with col2:
                    st.metric("Dataset Average", f"${avg_charge:,.0f}")
                with col3:
                    st.metric("Difference", f"${diff:,.0f}", f"{diff_pct:+.1f}%")
                
                # Show similar profiles
                st.subheader("Similar Client Profiles")
                similar = df[
                    (df['smoker'] == smoker) &
                    (df['age'].between(age-5, age+5))
                ].head(5)
                
                if not similar.empty:
                    st.dataframe(
                        similar[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']],
                        use_container_width=True
                    )
                else:
                    st.info("No similar profiles found in the dataset.")
    else:
        st.error("Model not found. Please ensure models/v1/insurance_model.joblib exists.")
        st.info("Run the training script: `python src/v1/train.py`")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**About**

This dashboard was built for healthcare insurance cost analysis. 
The model is trained on historical data and provides estimates based on client attributes.

**Disclaimer:** Predictions are estimates based on historical data and should not be used as final pricing.
""")
