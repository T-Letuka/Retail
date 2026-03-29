# RetailLens — From Transactions to Intelligence

> A complete data product built on 100,000 real e-commerce transactions —  
> demonstrating the full analyst → scientist → ML engineer → deployment pipeline.

---

## Quick Links

| | |
|---|---|
| 📄 Executive Report (Non-Technical) | [`outputs/RetailLens_Executive_Report.pdf`](outputs/Executive%20Report.pdf) |
| 📓 Act 1 — Data Analysis Notebook | [`notebooks/01_exploratory_analysis.ipynb`](notebooks/01_exploratory_analysis.ipynb) |
| 📓 Act 2 — Statistical Analysis | [`notebooks/02_statistical_analysis.ipynb`](notebooks/02_statistical_analysis.ipynb) *(coming soon)* |
| 📓 Act 3 — Machine Learning | [`notebooks/03_machine_learning.ipynb`](notebooks/03_machine_learning.ipynb) *(coming soon)* |
| 🚀 Live Dashboard | *Deploying after Act 3* |

---

## What Is This Project?

RetailLens is a portfolio project that takes 100,000 real orders from **Olist** —
Brazil's largest department store marketplace and extracts business intelligence,
statistically validated findings, and a deployable machine learning model from the raw data.

The project is structured as **four progressive acts**, each demonstrating a distinct
professional capability. The same dataset, the same codebase, four different roles.



## The Dataset

**Brazilian E-Commerce Public Dataset by Olist**  
Source: [kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)  
Period: September 2016 – October 2018 *(20 complete months used: Jan 2017 – Aug 2018)*  
Volume: ~100,000 orders across 8 relational CSV files  
Licence: CC BY-NC-SA 4.0

Olist connects small Brazilian businesses to larger retail channels. The dataset covers
customers, sellers, products, payments, delivery logistics, and review scores 
making it rich enough to support analysis across every dimension of a retail business.

---

## Act 1 — Data Analysis ✅

**Role demonstrated:** Data Analyst  
**Notebook:** [`notebooks/01_exploratory_analysis.ipynb`](notebooks/01_exploratory_analysis.ipynb)  
**Report:** [`outputs/RetailLens_Executive_Report.pdf`](outputs/Executive%20Report.pdf)

### What Was Built
A complete exploratory analysis across five dimensions with a data cleaning pipeline,
six publication-quality charts, RFM customer segmentation, and an executive report
written for non-technical stakeholders.

### Key Findings

**Revenue & Growth**
- Revenue grew **20.1% year-over-year** (2017 vs Jan–Aug 2018)
- November 2017 Black Friday spike of **+55% above October baseline** confirmed in data
- Platform stabilising at R$1.3–1.5M monthly by mid-2018 — growth maturing from
  hypergrowth (+85% MoM in early 2017) to single-digit monthly changes

**Delivery Performance**
- **93.2% of orders arrive on time or early** — strong platform-wide performance
- Mean delivery delay of **−11.9 days** — Olist deliberately under-promises delivery time
  by ~12 days, creating positive surprise for customers
- **Amazonas (AM)** is the only state averaging late (+9 days) — structural logistics
  constraint, not seller underperformance

**Customer Satisfaction**
- Mean review score: **4.09 / 5.00**
- Late deliveries score **2.27** vs on-time deliveries **4.29** — a 2.02 point gap
- Correlation between delay and satisfaction: **−0.27** (moderate, not the sole driver)
- Satisfaction dip Nov 2017–Mar 2018 coincides with Black Friday volume surge —
  operational strain during peak periods temporarily degrades customer experience

**RFM Customer Segmentation**
- **Champions: 8.6% of customers, 17.5% of revenue** — disproportionate revenue concentration
- **Loyal Customers: 33.8% of customers, 43.9% of revenue** — largest revenue segment
- **At Risk** customers represent recoverable revenue — were once valuable, recency dropping
- **Lost segment: 13.2%** — one-time buyers unlikely to return without intervention

### Data Engineering Decisions
The cleaning pipeline joined 8 relational tables into one master analytical dataset
with documented decisions at every step. Notable challenges:

- `customer_id` ≠ `customer_unique_id` — the same person receives a new customer_id
  per order. All customer-level analysis uses `customer_unique_id` (true person identifier)
- Payments table aggregated to order level before joining — split payments created
  multiple rows per order that would have inflated row counts on join
- 6 incomplete months excluded from trend analysis (platform launch period Sep–Dec 2016
  and truncated end months Sep–Oct 2018) — identified by counting active order days per month

---

## Act 2 — Statistical Analysis 🔄 *(In Progress)*

**Role demonstrated:** Data Scientist  
**Notebook:** [`notebooks/02_statistical_analysis.ipynb`](notebooks/02_statistical_analysis.ipynb)

### What Will Be Built
Formal hypothesis testing to validate which Act 1 patterns are statistically real,
and feature engineering for the Act 3 churn prediction model.

### Hypotheses to be Tested

| Hypothesis | Test | Status |
|-----------|------|--------|
| Late delivery → lower review scores | Mann-Whitney U + effect size | ⏳ |
| High-value RFM customers behave differently | Independent t-test + chi-squared | ⏳ |
| Review score varies by product category | One-way ANOVA + Tukey HSD | ⏳ |
| Seasonal revenue patterns are significant | Time-series decomposition + autocorrelation | ⏳ |

### Features to be Engineered
All features for the churn model will be built here with domain justification:
delivery delay, processing time, repeat customer flag, days since last order,
average basket value, category diversity, and average review score given.

---

## Act 3 — Machine Learning ⏳ *(Coming Soon)*

**Role demonstrated:** ML Engineer  
**Notebook:** [`notebooks/03_machine_learning.ipynb`](notebooks/03_machine_learning.ipynb)

### What Will Be Built
A customer churn prediction model identifying which customers are at risk of
not returning to the platform, with full model explainability using SHAP values.

### The Prediction Task
**Churn definition:** A customer whose last purchase was more than 180 days before
the dataset end date and who placed no subsequent orders.

**Why this problem:**  
Acquiring a new customer costs 5–7x more than retaining an existing one.
A model that identifies at-risk customers before they leave enables proactive
retention intervention at a fraction of acquisition cost.

### Planned Model Stack

| Component | Approach |
|-----------|---------|
| Class imbalance | SMOTE oversampling |
| Baseline model | Logistic Regression (interpretable) |
| Primary model | XGBoost (performant) |
| Validation | 5-fold cross-validation, ROC-AUC |
| Explainability | SHAP summary + waterfall plots |
| Evaluation metric | ROC-AUC (not accuracy — meaningless on imbalanced data) |

### Why XGBoost
XGBoost uses gradient boosting — each tree corrects errors from the previous one —
making it consistently stronger than Random Forest on tabular data. It also has better
regularisation controls reducing overfitting risk on a dataset of this size.

---

## Act 4 — Deployment ⏳ *(Coming Soon)*

**Role demonstrated:** Deployed product  
**Stack:** Streamlit + Streamlit Community Cloud (free, public URL)

### What Will Be Built
A four-page interactive dashboard making all findings accessible without
opening a single notebook.

| Page | Contents |
|------|---------|
| Business Overview | KPI cards, revenue trend, category breakdown |
| Customer Segments | Interactive RFM explorer, segment profiles |
| Statistical Findings | Hypothesis test results as plain-English finding cards |
| Churn Predictor | Live ML model — adjust customer characteristics, get churn probability |

The Churn Predictor page will load the serialised XGBoost model and return
real-time predictions with the top 3 SHAP factors explaining each specific prediction.

---

## Repository Structure

```
retaillens/
├── data/
│   ├── raw/                          # Original Olist CSVs (not committed)
│   └── processed/
│       ├── master.csv                # Joined analytical dataset (item level)
│       ├── rfm_base.csv              # Customer-level aggregations
│       └── rfm_scored.csv            # RFM scores + segment labels
├── notebooks/
│   ├── 00_data_cleaning.ipynb        # Full cleaning pipeline (8 tables → master.csv)
│   ├── 01_exploratory_analysis.ipynb # Act 1: 5 analyses, 6 charts, RFM segmentation
│   ├── 02_statistical_analysis.ipynb # Act 2: Hypothesis testing + feature engineering
│   └── 03_machine_learning.ipynb     # Act 3: Churn prediction + SHAP explainability
├── dashboard/
│   └── app.py                        # Act 4: Streamlit dashboard
├── outputs/
│   ├── figures/                      # All saved charts (PNG)
│   │   ├── 01_revenue_trend.png
│   │   ├── 02_customer_geography.png
│   │   ├── 03_delivery_performance.png
│   │   ├── 04_customer_satisfaction.png
│   │   └── 05_rfm_segmentation.png
│   ├── models/                       # Serialised ML models (Act 3)
│   │   ├── churn_model.pkl
│   │   └── scaler.pkl
│   └── RetailLens_Executive_Report.pdf
├── requirements.txt
└── README.md
```

---



## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=1.7.0
shap>=0.42.0
imbalanced-learn>=0.11.0
scipy>=1.11.0
statsmodels>=0.14.0
streamlit>=1.28.0
plotly>=5.17.0
jupyter>=1.0.0
ipykernel>=6.0.0
joblib>=1.3.0
openpyxl>=3.1.0
```

---

## Skills Demonstrated

| Skill | Where |
|-------|-------|
| Multi-table data engineering | `00_data_cleaning.ipynb` |
| Exploratory data analysis | `01_exploratory_analysis.ipynb` |
| Business communication | `outputs/RetailLens_Executive_Report.pdf` |
| RFM customer segmentation | `01_exploratory_analysis.ipynb` — Analysis 5 |
| Statistical hypothesis testing | `02_statistical_analysis.ipynb` *(coming)* |
| Feature engineering | `02_statistical_analysis.ipynb` *(coming)* |
| Classification + class imbalance | `03_machine_learning.ipynb` *(coming)* |
| Model explainability (SHAP) | `03_machine_learning.ipynb` *(coming)* |
| Model serialisation + serving | `03_machine_learning.ipynb` *(coming)* |
| Interactive dashboard deployment | `dashboard/app.py` *(coming)* |

---

## About

Built by **Tisetso Letuka**  
Biomedical Science (BSc) · Pathology Honours (cum laude) · 
Transitioning toward computational biology with a foundation in data science and full-stack development.

*This project is part of a portfolio designed to demonstrate the full data analyst → data scientist → ML engineer pipeline on a single real-world dataset.*