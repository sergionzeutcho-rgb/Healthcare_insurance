# Healthcare Insurance Cost Analysis & Prediction

![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

## Project Overview

This project provides a comprehensive analysis of healthcare insurance costs using statistical hypothesis testing and machine learning. It includes an interactive Streamlit dashboard for data exploration and cost prediction based on client attributes.

**Dataset:** [Healthcare Insurance Dataset from Kaggle](https://www.kaggle.com/datasets/willianoliveiragibin/healthcare-insurance) (1,337 records)

**Key Features:**
- Statistical hypothesis testing with comprehensive visualizations (t-tests, ANOVA, OLS regression)
- Machine learning cost prediction (RandomForest with 88.4% R²)
- Interactive 6-page Streamlit dashboard with real-time data filtering
- Dynamic charts that update based on user-selected filters (age, gender, smoker, region, BMI, children)
- Comprehensive exploratory data analysis (EDA) with 10+ visualizations
- Geographic heat map for regional cost analysis
- Feature importance analysis and 5-fold cross-validation
- Model diagnostic plots (residuals, Q-Q plots, actual vs predicted)
- Stakeholder-friendly interface with descriptions and business reflections
- Robust error handling for all filter combinations and edge cases

---

## Table of Contents

1. [Business Requirements](#business-requirements)
2. [Project Objectives](#project-objectives)
3. [Project Hypothesis & Validation](#project-hypothesis--validation)
4. [Dataset Description](#dataset-description)
5. [Project Structure](#project-structure)
6. [Key Findings](#key-findings)
7. [Installation & Setup](#installation--setup)
8. [Usage](#usage)
9. [Technologies Used](#technologies-used)
10. [Methodology](#methodology)
11. [Results & Performance](#results--performance)
12. [Learning Journey & Reflections](#learning-journey--reflections)
13. [Version Control & Project Management](#version-control--project-management)
14. [Future Improvements](#future-improvements)
15. [Credits](#credits)

---

## Business Requirements

### Context

A healthcare insurance company seeks to understand the key factors driving insurance costs and develop a predictive model to estimate charges for new policyholders. The company has collected historical data on 1,337 policyholders, including demographic information (age, gender, region), health indicators (BMI, smoking status), and family structure (number of children). The goal is to provide actionable insights for pricing strategies, risk assessment, and targeted intervention programs.

### Stakeholder Needs

**Primary Stakeholders:**
- **Actuarial Team:** Needs accurate cost predictions for pricing policies
- **Risk Management:** Requires identification of high-risk customer segments
- **Marketing Department:** Wants insights for targeted campaigns (e.g., smoking cessation programs)
- **Executive Leadership:** Seeks data-driven recommendations to reduce overall costs

### Business Requirements

**BR1: Cost Driver Analysis**
- Identify and quantify the impact of each factor (age, BMI, smoking, region, children, gender) on insurance costs
- Provide statistical evidence for which factors significantly affect charges
- Rank factors by importance to prioritize intervention strategies
- **Success Criteria:** Statistical validation of cost drivers with p-values < 0.05 and clear effect sizes

**BR2: Predictive Cost Estimation**
- Develop a machine learning model to predict insurance charges for new customers
- Achieve minimum 85% accuracy (R² score) on unseen data
- Enable stakeholders to estimate costs before policy issuance
- **Success Criteria:** Model R² ≥ 0.85, MAE < $3,000, validated through cross-validation

**BR3: Interactive Data Exploration**
- Create a user-friendly dashboard for non-technical stakeholders
- Enable filtering and segmentation of data by multiple criteria
- Provide real-time visualizations of cost patterns and distributions
- **Success Criteria:** Dashboard accessible without coding knowledge, responsive to user inputs, handles all filter combinations

**BR4: Smoking Impact Quantification**
- Measure the exact cost difference between smokers and non-smokers
- Calculate potential savings from smoking cessation programs
- Provide ROI justification for wellness initiatives
- **Success Criteria:** Quantified dollar impact with statistical confidence intervals

**BR5: Regional Cost Comparison**
- Analyze cost variations across geographic regions
- Identify regions with highest/lowest average costs
- Support regional pricing strategy decisions
- **Success Criteria:** ANOVA test results showing regional differences (if significant)

**BR6: Actionable Recommendations**
- Translate statistical findings into business strategies
- Prioritize initiatives by potential cost reduction impact
- Provide clear implementation guidance for stakeholders
- **Success Criteria:** Ranked list of recommendations with expected ROI

### Expected Deliverables

1. **Statistical Analysis Report:** Hypothesis test results with business interpretations
2. **Predictive Model:** Trained RandomForest model achieving ≥85% R² score
3. **Interactive Dashboard:** Streamlit application with 6 functional pages
4. **Documentation:** Comprehensive README with methodology, findings, and deployment instructions
5. **Business Recommendations:** Prioritized action items with cost/benefit analysis

### User Stories

**US1:** As an **actuary**, I want to **understand which factors most influence insurance costs**, so that **I can adjust pricing models accordingly**.

**US2:** As a **risk manager**, I want to **predict costs for new customers**, so that **I can assess risk levels before policy approval**.

**US3:** As a **marketing manager**, I want to **see the cost impact of smoking**, so that **I can justify investment in smoking cessation programs**.

**US4:** As an **executive**, I want to **explore cost patterns across demographics**, so that **I can make informed strategic decisions**.

**US5:** As a **data analyst**, I want to **filter data by multiple criteria**, so that **I can investigate specific customer segments**.

**US6:** As a **policy administrator**, I want to **download filtered data**, so that **I can create custom reports for stakeholders**.

---

## Project Objectives

1. **Data Collection & Cleaning:** Fetch and process healthcare insurance data
2. **Exploratory Data Analysis:** Identify key drivers of insurance costs
3. **Statistical Analysis:** Test hypotheses about smoker impact, regional differences, and BMI association
4. **Machine Learning:** Build and evaluate a supervised regression model to predict charges
5. **Dashboard Development:** Create an interactive Streamlit application for stakeholders
6. **Documentation:** Provide clear findings, limitations, and next steps

---

## Dataset Description

**Source:** Kaggle - Healthcare Insurance Dataset

**Records:** 1,337 insurance policy holders

**Features:**
- `age`: Age of the policyholder (18-64 years)
- `sex`: Gender (male/female)
- `bmi`: Body Mass Index (15.0-55.0)
- `children`: Number of children/dependents (0-5)
- `smoker`: Smoking status (yes/no)
- `region`: Geographic region (northeast, northwest, southeast, southwest)
- `charges`: Medical insurance costs (target variable, $1,122-$63,770)

**Derived Features:**
- `age_group`: Categorized age ranges
- `bmi_category`: BMI classification (underweight, normal, overweight, obese)
- `is_parent`: Binary indicator for having children

---

## Project Structure

```
Healthcare_insurance/
├── app/
│   └── healthcare_app.py         # Interactive dashboard application
├── data/
│   └── v1/
│       ├── raw/
│       │   └── insurance.csv     # Original dataset
│       └── processed/
│           └── insurance_clean.csv  # Cleaned dataset
├── jupyter_notebooks/
│   └── Healthcare_insurance.ipynb   # Full analysis notebook
├── models/
│   └── v1/
│       └── insurance_model.joblib   # Trained RandomForest model
├── reports/
│   └── v1/
│       └── figures/              # (Optional) Saved visualizations
├── src/
│   └── v1/
│       ├── etl.py                # Data extraction & cleaning
│       └── train.py              # Model training script
├── .gitignore                    # Version control exclusions
├── .python-version               # Python version specification for Heroku
├── .slugignore                   # Heroku slug exclusions
├── Procfile                      # Heroku deployment config
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── setup.sh                      # Heroku setup script
```

---

## Key Findings

### Statistical Analysis

**Hypothesis A: Smoker Impact**
- **Test:** Welch's t-test
- **Result:** t = 32.74, p < 0.001
- **Conclusion:** ✅ Smokers have **significantly higher** mean charges than non-smokers
- Smoker status is the **strongest driver** of insurance costs in this dataset

**Hypothesis B: Regional Differences**
- **Test:** One-way ANOVA
- **Result:** F = 2.93, p = 0.033
- **Conclusion:** ✅ **Statistically significant** differences exist across regions
- Regional variation is modest but present

**Hypothesis C: BMI Association (Controlling for Covariates)**
- **Test:** OLS Regression with controls
- **Result:** BMI coefficient = $339.25, p < 0.001
- **Conclusion:** ✅ BMI remains a **significant positive predictor** even after controlling for age, smoker status, sex, region, and children

### Machine Learning Results

**Model:** RandomForest Regressor (400 trees)

**Performance Metrics:**
- **R² Score:** 0.8843 (88.4% variance explained)
- **Mean Absolute Error (MAE):** $2,549
- **Root Mean Squared Error (RMSE):** $4,611
- **Cross-Validation (5-fold):** Mean R² = 0.8386 ± 0.0333

**Feature Importance (Top 5):**
1. Smoker status (no): 45.0%
2. BMI: 21.3%
3. Smoker status (yes): 14.8%
4. Age: 13.8%
5. Children: 2.3%

---

## Business Recommendations

Based on the comprehensive analysis of 1,337 insurance policies, the following prioritized recommendations are proposed to reduce costs and improve risk management:

### Priority 1: Smoking Cessation Programs (Highest ROI)

**Finding:**
- Smokers cost **$23,616 more annually** than non-smokers ($32,050 vs $8,434)
- Smoking accounts for **60%** of the model's predictive power (45% + 14.8% combined)
- Statistical confidence: **99.9%+** (p < 0.001)

**Recommendation:**
- Launch comprehensive smoking cessation support programs
- Offer premium discounts for verified non-smokers
- Partner with wellness providers for nicotine replacement therapy
- Implement annual smoking status verification

**Expected ROI:**
- **Potential savings: $23,616 per successful quit**
- If just 10% of smokers quit → saves ~$2.36M annually (assuming 100 smokers)
- Program cost typically $500-$1,000 per participant
- **Break-even: 1-2 participants succeeding** per 100 enrolled

**Implementation Timeline:** 3-6 months

---

### Priority 2: BMI Management & Wellness Initiatives (High Impact)

**Finding:**
- Every 1 BMI point increase adds **$339 annually** to costs
- BMI is the **2nd strongest predictor** (21.3% importance)
- Effect is **independent** of smoking status (p < 0.001)

**Recommendation:**
- Offer gym membership subsidies or corporate wellness programs
- Provide nutritional counseling and weight management support
- Create incentive programs for BMI improvement milestones
- Partner with fitness apps for tracking and engagement

**Expected ROI:**
- Average BMI reduction of 2-3 points → saves **$678-$1,017 per person/year**
- Program cost: $200-$500 per participant annually
- **Break-even: Modest weight loss** in majority of participants
- Secondary benefit: May support smoking cessation efforts

**Implementation Timeline:** 6-12 months

---

### Priority 3: Regional Pricing Adjustments (Moderate Impact)

**Finding:**
- **Statistically significant** regional differences (p = 0.033)
- Southeast: $14,735 average (highest)
- Southwest: $12,347 average (lowest)
- Difference: **$2,388 between highest and lowest**

**Recommendation:**
- Adjust regional pricing to reflect actual cost variations
- Investigate root causes of regional differences (provider costs, demographics)
- Consider region-specific wellness programs for high-cost areas
- Monitor regional trends for emerging patterns

**Expected ROI:**
- More accurate pricing → improved profitability margins
- **Modest impact** compared to smoking/BMI initiatives
- Supports competitive positioning in lower-cost regions

**Implementation Timeline:** 3-6 months (requires actuarial review)

---

### Priority 4: Predictive Risk Screening (Operational Efficiency)

**Finding:**
- Model achieves **88.4% accuracy** in predicting costs
- Mean Absolute Error of only **$2,549**
- Identifies high-risk customers before policy issuance

**Recommendation:**
- Integrate predictive model into underwriting workflow
- Flag high-risk applicants (predicted cost > $20,000) for additional review
- Offer tiered pricing based on risk profile
- Use predictions to guide personalized wellness recommendations

**Expected ROI:**
- Better risk assessment → **reduced adverse selection**
- Targeted interventions → lower average costs
- Operational efficiency → faster underwriting decisions

**Implementation Timeline:** 1-3 months (model already trained)

---

### Priority 5: Age-Based Preventive Care (Long-term Investment)

**Finding:**
- Age is the **4th strongest predictor** (13.8% importance)
- Costs increase predictably with age
- Opportunity for early intervention

**Recommendation:**
- Implement age-specific wellness checkups and screenings
- Offer preventive care incentives for younger policyholders
- Create educational programs about long-term health management
- Use age as a factor in personalized wellness recommendations

**Expected ROI:**
- **Long-term savings** through early disease detection
- Builds customer loyalty and engagement
- Reduces likelihood of high-cost claims in future years

**Implementation Timeline:** 12+ months

---

### Summary: Recommended Action Plan

**Immediate Actions (0-3 months):**
1. Deploy predictive model to underwriting team
2. Design smoking cessation program framework
3. Conduct regional pricing analysis

**Short-term Initiatives (3-6 months):**
1. Launch smoking cessation pilot program
2. Implement regional pricing adjustments
3. Establish wellness program partnerships

**Medium-term Goals (6-12 months):**
1. Roll out BMI management programs
2. Scale smoking cessation programs based on pilot results
3. Integrate age-based preventive care recommendations

**Expected Overall Impact:**
- **Primary driver: Smoking cessation** → potential to save $20,000+ per successful participant
- **Secondary driver: BMI management** → $500-$1,000 savings per participant annually
- **Combined effect:** Could reduce average costs by **10-15%** if programs achieve 25% participation with 50% success rate

**Investment Required:**
- Program development: $50,000-$100,000
- Annual operating costs: $200-$500 per participant
- Technology integration: $25,000-$50,000 (one-time)

**Measurement & Monitoring:**
- Track program participation rates monthly
- Monitor cost changes for program participants vs. non-participants
- Conduct annual ROI analysis
- Adjust programs based on effectiveness data

---

## Installation & Setup

### Prerequisites
- Python 3.12.8
- Git
- VS Code (recommended)

### Step 1: Clone the Repository

```bash
git clone <https://github.com/sergionzeutcho-rgb/Healthcare_insurance.git>
cd Healthcare_insurance
```

### Step 2: Create Virtual Environment

**In VS Code:**
1. Open Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
2. Type: `Python: Create Environment`
3. Select `Venv`
4. Choose Python 3.12.8

**Or via terminal:**
```bash
python -m venv .venv
```

### Step 3: Activate Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run ETL Pipeline (Optional)

If you want to regenerate the cleaned data:

```bash
python src/v1/etl.py
```

### Step 6: Train Model (Optional)

If you want to retrain the model:

```bash
python src/v1/train.py
```

---

## Usage

### 1. Jupyter Notebook Analysis

Open the analysis notebook:

```bash
jupyter notebook jupyter_notebooks/healthcare_insurance.ipynb
```

The notebook contains:
- Data quality checks
- Exploratory data analysis (EDA)
- Hypothesis testing (3 hypotheses)
- Machine learning model training
- Feature importance analysis
- Cross-validation
- Comprehensive interpretations

### 2. Streamlit Dashboard

Launch the interactive dashboard:

```bash
python -m streamlit run app/healthcare_app.py
```

Or using the full path to Python in Windows:

```bash
".venv/Scripts/python.exe" -m streamlit run app/healthcare_app.py
```

The dashboard will open at `http://localhost:8501`

**Dashboard Features:**

The dashboard provides a comprehensive, stakeholder-friendly interface with 6 interactive pages:

- **📝 Executive Summary:** Complete overview of all findings, statistical proof, business impact, prioritized recommendations, and final verdict for stakeholders
- **📊 Data Overview:** Dataset metrics, quality assessment, sample data, and detailed feature descriptions
- **📈 Exploratory Analysis:** **Interactive data exploration with dynamic filtering**
  - **🎛️ Interactive Filters (Sidebar):** Age range, gender, smoker status, region, BMI range, and number of children
  - **📊 Dynamic Visualizations:** All charts update in real-time based on filter selections:
    - Charges distribution histogram with adaptive binning
    - Smoker vs non-smoker charge comparison
    - BMI vs charges scatter plot (colored by smoker status)
    - Correlation heatmap for numeric features
    - Regional cost analysis with geographic heat map and bar charts
  - **Smart Error Handling:** Graceful handling of edge cases (single categories, small samples)
  - **Data Quality Warnings:** Alerts when filtered dataset is too small (<30 records)
- **🔬 Statistical Tests:** All 3 hypothesis tests with visualizations and statistical results:
  - Hypothesis A: Smoker impact (Welch's t-test with boxplots)
  - Hypothesis B: Regional differences (ANOVA with bar charts)
  - Hypothesis C: BMI association (OLS regression with controlled variables)
- **🤖 ML Model Performance:** Complete model diagnostics and insights:
  - Performance metrics (R², MAE, RMSE, CV scores)
  - Feature importance chart (top 11 features ranked)
  - Diagnostic plots (actual vs predicted, residual analysis, distribution)
  - Model validation assessment
- **🎯 Cost Predictor:** Interactive prediction tool with input form and similar profile analysis

**Interactive Features:**
- **Real-time filtering:** Explore data segments by demographics, health factors, and location
- **Dynamic charts:** All visualizations respond to filter changes instantly
- **Responsive design:** Adapts chart complexity to dataset size
- **Context-aware messages:** Shows relevant insights based on filtered data
- **Top navigation map:** Quick-access info boxes for all 6 pages
- **Sidebar navigation:** Page selector with emoji icons and filter controls

---

## Technologies Used

**Core:**
- Python 3.12.8
- Jupyter Notebook

**Data Processing:**
- pandas
- numpy

**Statistical Analysis:**
- scipy (statistical tests)
- statsmodels (OLS regression)

**Machine Learning:**
- scikit-learn (RandomForest, Pipeline, ColumnTransformer)
- joblib (model serialization)

**Visualization:**
- matplotlib
- seaborn

**Dashboard:**
- Streamlit

---

## Methodology

### 1. Data Preparation
- ETL pipeline to clean and validate raw data
- Feature engineering (age groups, BMI categories, parent indicator)
- No missing values or duplicates in final dataset

### 2. Exploratory Data Analysis
- Descriptive statistics for all variables
- Distribution analysis (charges are right-skewed)
- Correlation analysis for numeric features
- Group comparisons (smokers vs non-smokers)

### 3. Statistical Hypothesis Testing
- **Welch's t-test** for two-group comparisons (smoker vs non-smoker)
- **One-way ANOVA** for multi-group comparisons (regions)
- **OLS Regression** with controls to test BMI association

### 4. Machine Learning Pipeline
- Train-test split (80/20)
- ColumnTransformer for preprocessing:
  - OneHotEncoding for categorical features (sex, smoker, region)
  - Passthrough for numeric features (age, BMI, children)
- RandomForestRegressor with 400 trees
- Pipeline to prevent data leakage
- 5-fold cross-validation for robustness

### 5. Model Validation
- Residual analysis (actual vs predicted, residual plots)
- Q-Q plot for normality assumptions
- Feature importance extraction
- Cross-validation for generalization performance

---

## Results & Performance

### Model Diagnostics

✅ **Strong Predictive Accuracy:** R² = 0.8843 indicates the model explains 88.4% of charge variance

✅ **Robust Performance:** Cross-validation confirms consistent performance across different data splits (CV R² = 0.8386 ± 0.033)

✅ **Low Prediction Error:** MAE of $2,549 means average prediction is within ~$2,500 of actual cost

✅ **Well-Calibrated:** Residual plots show minimal bias, though some heteroscedasticity (non-constant variance) exists for high-charge cases (expected for insurance data)

### Business Insights

1. **Smoking Cessation Programs** could yield the highest ROI for cost reduction
2. **BMI Management** initiatives are justified based on significant positive association
3. **Regional Pricing** may need minor adjustments, but differences are modest
4. **Predictive Tool** enables stakeholders to estimate costs at policy application time

---

## Future Improvements

### Model Enhancements
- Compare RandomForest with Gradient Boosting (XGBoost, LightGBM) and Linear models
- Hyperparameter tuning via GridSearchCV
- Add SHAP values for instance-level interpretability
- Include interaction terms (e.g., smoker × BMI)

### Data Expansion
- Incorporate medical history variables (chronic conditions, hospitalizations)
- Add temporal data to account for inflation and policy changes
- Expand dataset size for better generalization

### Dashboard Features
- ~~Add interactive filters for data exploration~~ ✅ **COMPLETED**
- ~~Enable dynamic chart updates based on user selections~~ ✅ **COMPLETED**
- Add confidence intervals for predictions
- Enable batch predictions via CSV upload
- Add partial dependence plots for feature effects
- Add cost reduction scenario analysis (e.g., impact of smoking cessation programs)
- Export functionality for filtered data, reports, and charts
- User authentication for personalized predictions
- Save and compare multiple prediction scenarios

---

## Project Hypothesis & Validation

This study tested three specific hypotheses about healthcare insurance costs:

### Hypothesis A: Smoking Impact on Costs

**H₀:** Smokers and non-smokers have equal mean insurance charges (μ_smoker = μ_non-smoker)

**H₁:** Smokers have higher mean insurance charges than non-smokers (μ_smoker > μ_non-smoker)

**Test:** Welch's t-test (robust to unequal variances)

**Results:**
- **Test Statistic:** t = 32.74
- **P-Value:** < 0.001 (highly significant)
- **Effect Size:** Smokers pay $23,615 more on average ($32,050 vs. $8,434)
- **Decision:** ✅ REJECT H₀ - Strong evidence that smoking significantly increases insurance costs

**Business Implication:** Smoking is the single strongest driver of insurance costs, justifying differential pricing and smoking cessation program investments.

---

### Hypothesis B: Regional Cost Differences

**H₀:** All four regions have equal mean insurance charges

**H₁:** At least one region has different mean insurance charges

**Test:** One-way ANOVA

**Results:**
- **Test Statistic:** F = 2.93
- **P-Value:** 0.033 (statistically significant at α = 0.05)
- **Regional Means:** Southeast ($14,735) is highest, Southwest ($12,347) is lowest
- **Effect Size:** Modest ($2,388 difference between highest/lowest)
- **Decision:** ✅ REJECT H₀ - Evidence of regional differences, though effect size is smaller than smoker impact

**Business Implication:** Regional variations exist but are modest; suggests localized healthcare cost factors (e.g., provider density, cost of living).

---

### Hypothesis C: BMI Impact (Controlling for Confounders)

**H₀:** BMI has no association with insurance charges when controlling for age, sex, smoker, children, and region

**H₁:** BMI is positively associated with insurance charges even after controlling for other factors

**Test:** OLS Multiple Regression

**Results:**
- **BMI Coefficient:** $339.25 per BMI unit increase
- **P-Value:** < 0.001 (highly significant)
- **Confidence Interval:** [323.8, 354.7] (tight bounds)
- **Model R²:** 0.753 (controls explain 75.3% of variance)
- **Decision:** ✅ REJECT H₀ - BMI remains a significant predictor independent of other factors

**Business Implication:** BMI's independent effect justifies wellness programs targeting weight management, beyond just smoking cessation.

---

## Learning Journey & Reflections

### Challenges Encountered

**1. Directory Management & Project Structure**
- **Challenge:** Maintaining organized file paths across notebooks, scripts, and dashboard
- **Solution:** Adopted `pathlib` for cross-platform path handling and created v1/ subdirectories for versioning
- **Learning:** Version control through folder structure (data/v1/, models/v1/) enables experimentation without breaking production code

**2. Statistical Library Compatibility**
- **Challenge:** `statsmodels` OLS regression failed with "ValueError: could not convert string to float" when using object dtypes
- **Solution:** Added `.astype(float)` conversion before fitting OLS models
- **Learning:** Type checking is critical when integrating multiple libraries; pandas infers types but doesn't always match downstream requirements

**3. Windows Command-Line Execution**
- **Challenge:** Direct `streamlit run` command failed in Git Bash on Windows
- **Solution:** Used `python -m streamlit run` to invoke Streamlit as a module within the virtual environment
- **Learning:** Module-based execution (`python -m`) is more portable across operating systems and shell environments

**4. Stakeholder Communication**
- **Challenge:** Initial dashboard used technical jargon (p-values, heteroscedasticity) unsuitable for non-technical stakeholders
- **Solution:** Rewrote all descriptions with plain language, concrete examples, and business implications
- **Learning:** Data science deliverables must be audience-appropriate; statistical rigor ≠ technical language

**5. Matplotlib Import Errors & Deprecation Warnings**
- **Challenge:** Importing `cm`, `Rectangle`, `Normalize` directly from `matplotlib` caused AttributeError; `cm.get_cmap()` deprecated in Matplotlib 3.7
- **Solution:** Used explicit submodule imports (`matplotlib.cm`, `matplotlib.patches.Rectangle`, `matplotlib.colors.Normalize`) and migrated to `plt.colormaps.get_cmap()`
- **Learning:** Python package structure matters; always check official documentation for correct import paths and stay current with deprecation warnings

**6. Interactive Filter Edge Cases**
- **Challenge:** Filtering by single smoker status or region caused KeyError when accessing missing categories in pandas groupby results
- **Solution:** Implemented conditional checks before accessing specific index values and provided contextual messages for filtered views
- **Learning:** User interactions create unpredictable data states; defensive programming with conditional checks prevents runtime errors

**7. Dynamic Visualization Challenges**
- **Challenge:** Hard-coded chart colors and normalizations failed when filters reduced categories (e.g., only smokers, single region)
- **Solution:** Adaptive color assignment based on available categories, division-by-zero protection in normalization, and empty dataframe checks
- **Learning:** Dynamic UIs require robust error handling; test edge cases (single category, empty results, extreme filters)

### Skills Developed

**Technical:**
- Statistical hypothesis testing (t-tests, ANOVA, OLS regression)
- Machine learning pipelines with preprocessing (ColumnTransformer, OneHotEncoder)
- Model validation (cross-validation, residual diagnostics, Q-Q plots)
- **Interactive dashboard development with real-time filtering and dynamic visualizations**
- **Error handling for edge cases and graceful degradation**
- Version control best practices (v1/ folder structure, .gitignore)

**Analytical:**
- Distinguishing statistical vs. practical significance
- Interpreting p-values, confidence intervals, and effect sizes in business context
- Balancing model complexity vs. interpretability (RandomForest vs. Linear Regression trade-offs)
- Identifying confounding variables and implementing statistical controls
- **Designing user-friendly data exploration tools with appropriate warnings**

**Communication:**
- Translating technical findings into stakeholder-friendly language
- Creating visualizations with clear titles, labels, and annotations
- Structuring analysis narratives (hypotheses → tests → conclusions → business actions)
- **Providing context-aware messages based on filtered data states**

### Adaptation Process

This project required continuous adaptation of tools and methods:

1. **Statistical Methods:** Started with simple t-tests, evolved to OLS regression with controls to address confounding
2. **Machine Learning:** Initially explored Linear Regression, switched to RandomForest for better handling of non-linear relationships
**Dashboard Design Iterations:**
1. **Initial:** Basic visualizations on single page
2. **Enhancement:** Multi-page navigation with executive summary
3. **Advanced:** Interactive filters with real-time chart updates
4. **Final:** Robust error handling for all filter combinations and edge cases

**Analysis Workflow Evolution:**
1. **Exploratory Phase:** Initial Jupyter notebook (74 cells) with basic EDA
2. **Statistical Phase:** Hypothesis testing and formal validation
3. **ML Phase:** Model training with pipeline and cross-validation
4. **Deployment Phase:** Interactive dashboard with filtering capabilities
5. **Documentation Phase:** Comprehensive README with methodology and reflections

**Key Takeaway:** Data science projects are inherently iterative; each analysis step reveals new questions requiring methodological adjustments, and production deployment requires extensive edge case testing.

---

## Version Control & Project Management

This project follows a structured version control approach:

### Folder Structure Strategy

**v1/ Subdirectories:** All data, models, and source code are organized under `v1/` folders:
```
data/v1/raw/          # Original unmodified datasets
data/v1/processed/    # Cleaned, validated data
models/v1/            # Trained model artifacts
src/v1/               # Analysis scripts
reports/v1/           # (Future) Generated reports
```

**Benefits:**
1. **Experimentation Safety:** Future versions (v2/, v3/) can coexist without overwriting stable v1 outputs
2. **Reproducibility:** Each version captures a complete snapshot of data → model → results
3. **Rollback Capability:** If new methods fail, v1 remains functional
4. **Auditability:** Stakeholders can reference specific version numbers in discussions

### Git Workflow

**Branch Strategy:**
- `main` branch: Stable, tested code only
- Feature branches: `feature/add-ols-regression`, `feature/streamlit-dashboard`
- Development workflow: Create branch → commit changes → test → merge to main

**Commit Best Practices:**
- Descriptive commit messages: "Add BMI regression with age controls" (not "Update notebook")
- Atomic commits: Each commit addresses one logical change
- Regular commits: After each successful test or analysis milestone

**Files Protected by .gitignore:**
```
core.Microsoft*      # Microsoft core dumps
core.mongo*          # MongoDB core dumps
core.python*         # Python core dumps
env.py              # Environment variables
__pycache__/        # Python bytecode cache
*.py[cod]           # Python compiled files (.pyc, .pyo, .pyd)
node_modules/       # Node.js dependencies
.github/            # GitHub configuration
cloudinary_python.txt  # Cloudinary credentials
kaggle.json         # Kaggle API credentials
.venv/              # Virtual environment (large, platform-specific)
.ipynb_checkpoints/ # Jupyter notebook checkpoints
.DS_Store           # macOS metadata
Thumbs.db           # Windows thumbnail cache
*.log               # Log files
*.sqlite, *.db      # Database files
.pytest_cache/      # Pytest cache
.coverage, htmlcov/ # Coverage reports
dist/, build/       # Distribution and build files
*.egg-info/, .eggs/ # Python package metadata
```

This ensures the repository stays clean and only tracks essential project files.

---

## Credits

**Dataset:** [Kaggle - Healthcare Insurance Dataset](https://www.kaggle.com/datasets/willianoliveiragibin/healthcare-insurance)

**Author:** Sergio Kadje (Code Institute Data Analytics Capstone Project 2)

**Institution:** Code Institute - LMS - Teachers

**Date:** February 2026

**AI Assistance:** AI tools (GitHub Copilot) were used to assist with code generation, amendments, and enhancement of the Streamlit dashboard.

---

## License

This project is created for educational purposes as part of the Code Institute Data Analytics program.

---

## Contact

For questions or feedback about this project, please reach out through GitHub

## Deployment Reminders

* Set the `.python-version` Python version to a [Heroku-22](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version that closest matches what you used in this project.
* The project can be deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the **Deploy** tab, select **GitHub** as the deployment method.
3. Select your repository name and click **Search**. Once it is found, click **Connect**.
4. Select the branch you want to deploy, then click **Deploy Branch**.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button **Open App** at the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the `.slugignore` file.
