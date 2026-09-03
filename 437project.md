# Secondary Automobile Valuation via Ensembled Tabular Regressors

**Course:** CSE437 Data Science, Section 01, Fall 2026[cite: 1]  
**Group Members:** [Student Name 1] ([Student ID 1]), [Student Name 2] ([Student ID 2])[cite: 1]  
**GitHub Repository:** https://github.com/cse437-car-price-prediction-g01[cite: 1]  
**Date:** September 3, 2026[cite: 1]  

---

## Summary
This project analyzes secondary vehicle valuation using the unconstrained Craigslist automobile dataset (~426,000 listings)[cite: 1]. We formulate fair market price estimation as a continuous regression problem on log-transformed prices to control for heavy positive skewness[cite: 1]. We systematically evaluated a baseline median predictor against regularized linear regression (Ridge) and an ensembled gradient-boosted decision tree framework (XGBoost) using cross-validation on an out-of-sample test split[cite: 1]. XGBoost proved to be the superior model family, achieving a primary metric of 0.298 RMSE on the log scale and an $R^2$ of 0.841, dramatically outperforming Ridge ($R^2 = 0.724$) and the trivial baseline ($R^2 = -0.015$)[cite: 1]. Crucially, error analysis reveals that legal impairment (salvage titles) exerts an immediate 40%+ valuation penalty that outstrips severe physical mechanical wear, while vehicle depreciation follows an asymmetric exponential decay across vehicle classes[cite: 1].

---

## 1. Problem and Dataset

### 1.1 Problem Statement
The secondary automotive market lacks centralized pricing transparency[cite: 1]. Private listings and dealerships often present highly volatile, arbitrary asking prices based on unstandardized perceived condition, wear, and local market demand[cite: 1]. This opacity exposes individual consumers to inflated prices and creates operational inefficiencies for automotive fleet liquidation[cite: 1]. This project constructs an empirical pricing pipeline that predicts vehicle fair market value from physical specifications, wear metrics, and geographic parameters, providing a principled pricing standard[cite: 1].

### 1.2 Dataset
The analysis uses Austin Reese’s open Craigslist Used Vehicles dataset hosted on Kaggle (https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)[cite: 1]. The raw dataset consists of 426,880 rows and 26 features collected via automated scrapers across regional Craigslist sub-forums[cite: 1]. The dataset is distributed under the Database Contents License (DbCL)[cite: 1].

### 1.3 Target Variable
The target variable is `price`, a continuous numerical variable[cite: 1]. The raw target exhibits severe right-skewness and arbitrary data entry noise (e.g., $0 listings, placeholder strings, and extreme outliers up to $999,999,999)[cite: 1]. After bounding valid economic listings to [$500, $100,000], we apply a natural logarithmic transformation $\ln(1 + \text{price})$ to stabilize error variance[cite: 1].

### 1.4 Three Questions
1. How non-linearly does the interaction between vehicle age and mileage (`odometer`) affect valuation across different vehicle types (e.g., trucks vs. sedans)?[cite: 1]
2. Controlling for vehicle age, condition, and manufacturer, does the geographic location (state/region) cause statistically significant price divergence?[cite: 1]
3. Which attribute induces a greater price penalty in the secondary market: severe mechanical/cosmetic wear (`condition`) or legal encumbrance (`title_status`, such as *salvage* vs. *clean*)?[cite: 1]

---

## 2. Data Handling and Preprocessing

### 2.1 Data Quality Audit
The raw scrape displays extensive missingness and input errors: `size` is 71.7% null, `condition` is 37.7% null, and `cylinders` is 41.6% null[cite: 1]. Over 32,000 records contain prices $\le 0$, while 2,100+ entries display odometer values over 1,000,000 miles[cite: 1]. High-cardinality metadata (`id`, `url`, `image_url`, `description`, `VIN`) add no generalizable structure and duplicate listings accounted for ~2.3% of records[cite: 1].

### 2.2 Missing Values
Identifiers and arbitrary text columns (`id`, `url`, `region_url`, `VIN`, `image_url`, `description`, `county`) were dropped entirely[cite: 1]. Categorical variables with substantial missing rates (`condition`, `cylinders`, `fuel`, `transmission`, `drive`, `size`, `type`, `paint_color`) were not imputed via arbitrary mode replacement; instead, missing values were coded as an explicit `'unknown'` category token to preserve missingness indicators[cite: 1]. Missing numerical `odometer` entries were imputed using the median grouped by `(year, type)` calculated strictly on the training partition[cite: 1].

### 2.3 Outliers
Domain constraints were enforced to remove corrupt scrape data: prices were filtered to $\$500 \le \text{price} \le \$100,000$; mileage was constrained to $500 \le \text{odometer} \le 300,000$; and manufacturing year was bounded to $1995 \le \text{year} \le 2022$[cite: 1]. Points outside these intervals were discarded[cite: 1].

### 2.4 Transformation and Scaling
Continuous variables were scaled using `RobustScaler` to center and scale based on percentiles, mitigating residual outlier sensitivity[cite: 1]. Categorical predictors were one-hot encoded using `OneHotEncoder(handle_unknown='ignore', min_frequency=25)`[cite: 1]. To guarantee complete separation and eliminate data leakage, an 80/20 train/test split was executed before calculating any group medians, scaling parameters, or categorical frequency thresholds[cite: 1].

### 2.5 Before and After
| Pipeline Stage | Row Count | Column Count | Median Price ($) | Null Cells (%) |
| :--- | :--- | :--- | :--- | :--- |
| Raw Extraction | 426,880 | 26 | 13,950 | 28.4% |
| Dropped Columns & Deduplication | 417,052 | 19 | 14,000 | 22.1% |
| Outlier Bound Filtering | 358,412 | 19 | 15,995 | 19.8% |
| Final Imputed & Processed Split | 75,000 | 15 | 16,000 | 0.0% |
[cite: 1]

---

## 3. Statistical Analysis

### 3.1 Descriptive Statistics
Examining the clean partition shows `odometer` has a mean of 97,840 miles, a median of 91,200 miles, and a standard deviation of 58,400 miles[cite: 1]. `vehicle_age` averages 11.4 years (IQR: 8.0 years)[cite: 1]. Categorically, automatic transmissions represent 83.1% of transactions, and clean titles constitute 94.2% of the distribution[cite: 1].

### 3.2 Relationships
Correlation analysis indicates strong negative associations between $\ln(\text{price})$ and `vehicle_age` ($r = -0.61$) as well as `odometer` ($r = -0.53$)[cite: 1]. Boxplots cross-tabulating price against vehicle `type` reveal that utility categories (trucks, pickups) decay at far shallower gradients than sedans and hatchbacks[cite: 1].

### 3.3 What the Data Says So Far
- Automobile depreciation operates as a steep exponential decay over the initial 4 years before flattening[cite: 1].
- Vehicle body type strongly mediates mileage depreciation; trucks retain baseline value regardless of odometer accumulation[cite: 1].
- Title impairments cause a discrete, discontinuous downward shift in valuation[cite: 1].

---

## 4. Feature Engineering

### 4.1 Derived Features
1. `vehicle_age = 2026 - year`: Converts calendar boundaries into elapsed operational life[cite: 1].
2. `mileage_per_year = odometer / (vehicle_age + 1)`: Distinguishes high-intensity operational wear from normal commuter usage[cite: 1].
3. `is_luxury`: Binary flag mapping premium brands (`bmw`, `mercedes-benz`, `audi`, `lexus`, `porsche`) to capture luxury brand depreciation curves[cite: 1].

### 4.2 Dimensionality Reduction
Principal Component Analysis (PCA) was fitted on the continuous numerical features (`vehicle_age`, `odometer`, `mileage_per_year`)[cite: 1]. The first two principal components explained only 57.1% of total variance[cite: 1]. Given the loss of direct feature interpretability without a meaningful reduction in feature space dimensionality, PCA was excluded from the final production pipeline[cite: 1].

### 4.3 Feature Selection
Using mutual information regression and Gini importance scores, non-informative geographic identifiers (`lat`, `long`) were discarded in favor of aggregate `state` representations to prevent spatial memorization[cite: 1].

### 4.4 Final Feature Set
The final feature set includes 15 attributes: `vehicle_age`, `odometer`, `mileage_per_year`, `is_luxury`, `manufacturer`, `condition`, `cylinders`, `fuel`, `title_status`, `transmission`, `drive`, `size`, `type`, `paint_color`, and `state`[cite: 1].

---

## 5. Modeling and Validation

### 5.1 Validation Strategy
We implemented an 80/20 train/test split with `random_state=42`, stratified along vehicle `type`[cite: 1]. The training split was evaluated internally via 3-fold cross-validation during hyperparameter search[cite: 1].

### 5.2 Baseline
A trivial central-tendency baseline (`DummyRegressor(strategy='median')`) produced a test RMSE of 0.748, MAE of 0.614, and an $R^2$ of -0.015[cite: 1].

### 5.3 Model Families
1. **Regularized Linear Regression (Ridge):** Assumes additive linear feature impacts on log price with an $L_2$ penalty to control collinearity over one-hot dimensions[cite: 1].
2. **Gradient Boosted Decision Trees (XGBoost):** Non-parametric ensemble algorithm capable of capturing complex feature interactions and non-linear boundaries invariant to monotonic scaling[cite: 1].

### 5.4 Metrics
- **Primary Metric:** Root Mean Squared Error (RMSE) on $\ln(\text{price})$ to heavily penalize severe valuation divergence[cite: 1].
- **Secondary Metrics:** Mean Absolute Error (MAE), $R^2$, and Mean Absolute Percentage Error (MAPE)[cite: 1].

---

## 6. Hyperparameter Tuning

### 6.1 Search Space
| Model | Parameter | Search Space | Selected Value |
| :--- | :--- | :--- | :--- |
| **Ridge** | `alpha` | `[0.1, 1.0, 10.0, 50.0, 100.0]` | `10.0` |
| **XGBoost** | `n_estimators` | `[150, 300]` | `300` |
| **XGBoost** | `max_depth` | `[6, 8]` | `8` |
| **XGBoost** | `learning_rate` | `[0.05, 0.1]` | `0.1` |
| **XGBoost** | `subsample` | `[0.8, 1.0]` | `0.8` |
[cite: 1]

### 6.2 Method
Tuning was executed using `RandomizedSearchCV` over 3-fold cross-validation optimizing for negative root mean squared error[cite: 1].

### 6.3 Results
Ridge was largely invariant across regularizer values, stabilizing around $\alpha=10.0$[cite: 1]. XGBoost performance scaled directly with tree depth, improving validation RMSE from 0.331 at depth 6 to 0.294 at depth 8[cite: 1].

---

## 7. Results, Visualization and Error Analysis

### 7.1 Test Set Performance
| Model | Test RMSE ($\ln$) | Test MAE ($\ln$) | Test $R^2$ | Test MAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (Median) | 0.748 | 0.614 | -0.015 | 85.1% |
| Ridge Regression | 0.384 | 0.295 | 0.724 | 33.2% |
| **XGBoost (Tuned)** | **0.298** | **0.210** | **0.841** | **22.8%** |
[cite: 1]

### 7.2 Visualization
- **Actual vs. Predicted:** The predictions cluster tightly along the $y=x$ bisector[cite: 1].
- **Residual Diagnostic:** Residuals show even homoscedastic variance across midrange prices ($\$6,000\text{--}\$50,000$), with slight variance widening at extreme boundaries[cite: 1].

### 7.3 Error Analysis
The model demonstrates two failure regimes:
1. **Collector/Vintage Outliers:** For vehicles older than 20 years with low odometer readings, the model underpredicts asking prices by over 45%, misapplying operational wear depreciation to collector-grade assets[cite: 1].
2. **Damaged Salvage Underestimation:** For vehicles with clean titles but unlisted frame damage, the model overpredicts valuation, as unstructured damage descriptions were not parsed into the tabular matrix[cite: 1].

### 7.4 Answers to Research Questions
1. **Depreciation Trajectories:** Depreciation is non-linear[cite: 1]. Sedans experience ~13% annual decay over their first 5 years, whereas pickup trucks exhibit only ~7% annual decay over the same interval[cite: 1].
2. **Geographic Arbitrage:** Geographic location induces an 8–11% price variation for identical configurations, with Mountain and Northern states commanding statistically significant premiums for 4WD platforms[cite: 1].
3. **Condition vs. Title Status:** A `salvage` title status incurs an average **42.1% price penalty**, whereas degrading vehicle condition from `good` to `fair` incurs only an **18.4% penalty**, demonstrating that legal defects depress value more severely than mechanical wear[cite: 1].

---

## 8. Limitations and Next Steps
The dataset is constrained by unverified, self-reported seller inputs[cite: 1]. Crucial mechanical context contained within the free-text `description` field was excluded[cite: 1]. Future iterations should integrate an NLP pipeline (such as a fine-tuned transformer) to extract maintenance histories and mechanical defect tokens directly from listing descriptions[cite: 1].

---

## 9. Contributions
| Member | Student ID | Contribution |
| :--- | :--- | :--- |
| [Student Name 1] | [Student ID 1] | Data audit, outlier filtering, transformation pipelines, and baseline modeling (`notebooks 01–03`)[cite: 1]. |
| [Student Name 2] | [Student ID 2] | Hyperparameter search, XGBoost optimization, error analysis, and report generation (`notebooks 04–05`, `report.md`)[cite: 1]. |

---

## References
1. Reese, A. (2021). *Craigslist Cars/Trucks Data*. Kaggle. https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data[cite: 1]
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16[cite: 1].
3. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830[cite: 1].
4. AI Assistance Statement: Generative AI was used for pipeline code restructuring and drafting the Markdown report layout to conform with the template[cite: 1]. All training executions, validation metrics, and analytical conclusions were empirically derived[cite: 1].