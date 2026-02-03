# Healthcare Insurance Cost Analysis & Prediction

![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

## Project Overview

This project provides a comprehensive analysis of healthcare insurance costs using statistical hypothesis testing and machine learning. It includes an interactive Streamlit dashboard for data exploration and cost prediction based on client attributes.

**Dataset:** [Healthcare Insurance Dataset from Kaggle](https://www.kaggle.com/datasets/willianoliveiragibin/healthcare-insurance) (1,337 records)

**Key Features:**
- Statistical hypothesis testing with comprehensive visualizations (t-tests, ANOVA, OLS regression)
- Machine learning cost prediction (RandomForest with 88.4% R²)
- Interactive 5-page Streamlit dashboard with navigation map and all analyses integrated
- Comprehensive exploratory data analysis (EDA) with 10+ visualizations
- Feature importance analysis and 5-fold cross-validation
- Model diagnostic plots (residuals, Q-Q plots, actual vs predicted)
- Stakeholder-friendly interface with descriptions and business reflections

---

## Table of Contents

1. [Project Objectives](#project-objectives)
2. [Project Hypothesis & Validation](#project-hypothesis--validation)
3. [Dataset Description](#dataset-description)
4. [Project Structure](#project-structure)
5. [Key Findings](#key-findings)
6. [Installation & Setup](#installation--setup)
7. [Usage](#usage)
8. [Technologies Used](#technologies-used)
9. [Methodology](#methodology)
10. [Results & Performance](#results--performance)
11. [Learning Journey & Reflections](#learning-journey--reflections)
12. [Version Control & Project Management](#version-control--project-management)
13. [Future Improvements](#future-improvements)
14. [Credits](#credits)

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
│   └── healthcare_insurance.ipynb   # Full analysis notebook
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
├── requirements.txt              # Python dependencies
├── setup.sh                      # Heroku setup script
├── Procfile                      # Heroku deployment config
└── README.md                     # Project documentation
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
- **📈 Exploratory Analysis:** 5 interactive visualizations with key findings and business reflections:
  - Charges distribution histogram (right-skewed pattern)
  - Smoker vs non-smoker charge comparison
  - BMI vs charges scatter plot (colored by smoker status)
  - Correlation heatmap for numeric features
  - Regional cost analysis
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

**Navigation Features:**
- Top navigation map with 6 quick-access info boxes
- Sidebar page selector with emoji icons
- Comprehensive descriptions and reflections on every page
- All visualizations from the Jupyter notebook integrated

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

✅ **Well-Calibrated:** Residual plots show minimal bias, though some heteroscedasticity exists for high-charge cases (expected for insurance data)

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
- Add confidence intervals for predictions
- Enable batch predictions via CSV upload
- Add partial dependence plots for feature effects
- Add cost reduction scenario analysis (e.g., impact of smoking cessation)
- Export functionality for reports and charts

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

**5. Matplotlib Import Errors**
- **Challenge:** Importing `cm`, `Rectangle`, `Normalize` directly from `matplotlib` caused AttributeError
- **Solution:** Used explicit submodule imports (`matplotlib.cm`, `matplotlib.patches.Rectangle`, `matplotlib.colors.Normalize`)
- **Learning:** Python package structure matters; always check official documentation for correct import paths

### Skills Developed

**Technical:**
- Statistical hypothesis testing (t-tests, ANOVA, OLS regression)
- Machine learning pipelines with preprocessing (ColumnTransformer, OneHotEncoder)
- Model validation (cross-validation, residual diagnostics, Q-Q plots)
- Interactive dashboard development with Streamlit
- Version control best practices (v1/ folder structure, .gitignore)

**Analytical:**
- Distinguishing statistical vs. practical significance
- Interpreting p-values, confidence intervals, and effect sizes in business context
- Balancing model complexity vs. interpretability (RandomForest vs. Linear Regression trade-offs)
- Identifying confounding variables and implementing statistical controls

**Communication:**
- Translating technical findings into stakeholder-friendly language
- Creating visualizations with clear titles, labels, and annotations
- Structuring analysis narratives (hypotheses → tests → conclusions → business actions)

### Adaptation Process

This project required continuous adaptation of tools and methods:

1. **Statistical Methods:** Started with simple t-tests, evolved to OLS regression with controls to address confounding
2. **Machine Learning:** Initially explored Linear Regression, switched to RandomForest for better handling of non-linear relationships
3. **Dashboard Design:** Iterated from basic charts to comprehensive 5-page navigation with business reflections
4. **Documentation:** Progressed from code comments to formal README with hypothesis validation and methodology justification

**Key Takeaway:** Data science projects are inherently iterative; each analysis step reveals new questions requiring methodological adjustments.

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
