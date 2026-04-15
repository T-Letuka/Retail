# RetailLens — From Transactions to Intelligence

> A complete end-to-end data product built on 100,000 real e-commerce transactions,  
> demonstrating the full **analyst → scientist → ML engineer → deployment** pipeline  
> on a single dataset.

---

## 🚀 Live Dashboard

**[retaildashboardtl.streamlit.app](https://retaildashboardtl.streamlit.app/)**

Open it on your phone. No setup. No code. Just the product.

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

RetailLens takes 100,000 real orders from **Olist**  Brazil's largest department store marketplace  and builds a complete data product from raw transactions to a deployed prediction tool.

The project is structured as **four progressive acts**, each demonstrating a distinct professional capability using the same dataset and codebase.

| Act | Role | Question Answered | Status |
|-----|------|------------------|--------|
| **1** | Data Analyst | What is happening in this business? | ✅ Complete |
| **2** | Data Scientist | Which patterns are statistically real? | ✅ Complete |
| **3** | ML Engineer | Which customers are about to churn? | ✅ Complete |
| **4** | Deployment | Can anyone explore this without opening a notebook? | ✅ Live |

---

## The Dataset

**Brazilian E-Commerce Public Dataset by Olist**
- **Source:** [kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **Period:** September 2016 – October 2018 *(20 complete months used: Jan 2017 – Aug 2018)*
- **Volume:** ~100,000 orders across 8 relational CSV files
- **Licence:** CC BY-NC-SA 4.0

---

## Act 1 : Data Analysis ✅

**Role:** Data Analyst
**Notebook:** [`01_exploratory_analysis.ipynb`](notebooks/01_exploratory_analysis.ipynb)
**Report:** [`RetailLens_Executive_Report.pdf`](outputs/RetailLens_Executive_Report.pdf)

Five analyses across revenue, geography, delivery, satisfaction, and RFM customer segmentation. Six publication-quality charts. An executive report written for non-technical stakeholders.

### Key Findings

| Area | Finding |
|------|---------|
| Revenue | 20.1% YoY growth. November Black Friday spike +55% above October baseline. Platform stabilising at R$1.3–1.5M monthly by mid-2018. |
| Delivery | 93.2% on time or early. Mean delay −11.9 days — deliberate under-promise strategy. Amazonas (AM) is the only state averaging late (+9 days). |
| Satisfaction | Mean review score 4.09/5.00. Late deliveries score 2.27 vs on-time 4.29 — a 2.02 point gap. |
| Segmentation | Champions: 8.6% of customers, 17.5% of revenue. 71.1% of all customers never returned after their first order. |

### Data Engineering Notes

The cleaning pipeline joined 8 relational tables with documented decisions at every step. Notable challenges:

- `customer_id` ≠ `customer_unique_id` — the same person receives a new `customer_id` per order. All customer-level analysis uses `customer_unique_id`
- Payments table aggregated to order level before joining — split payments create multiple rows per order that inflate row counts on join
- 6 incomplete months excluded from trend analysis — identified by counting active order days per month

---

## Act 2 — Statistical Analysis ✅

**Role:** Data Scientist
**Notebook:** [`02_statistical_analysis.ipynb`](notebooks/02_statistical_analysis.ipynb)

Formal hypothesis testing to validate Act 1 patterns, churn label definition, and pre-modelling feature validation.

### Hypotheses Tested

| Hypothesis | Test | Result | Effect |
|-----------|------|--------|--------|
| Late delivery → lower review scores | Mann-Whitney U | ✓ Supported (p < 0.001) | Medium |
| High vs low RFM customers differ behaviourally | Mann-Whitney U + Chi-squared | Partially supported | Large on spend only (r = 0.857) |

### Statistical vs Practical Significance

Hypothesis 2 tested 4 variables between high and low RFM customer groups. All 4 reached statistical significance (p < 0.001) — but effect size analysis showed only total spend was practically meaningful (r = 0.857). Order frequency, category diversity, and payment method had negligible effect sizes.

With 96,000 customers, even negligible real-world differences become statistically detectable. Effect size tells the real story. This distinction is documented throughout and becomes a recurring analytical theme across the project.

### Churn Foundation

| Finding | Value |
|---------|-------|
| Churn rate (180-day threshold) | 71.1% |
| Class imbalance ratio | 2.5:1 (churned:retained) |
| Churn rate spread across RFM segments | 71.7 percentage points |
| RFM segment validation | Cramér's V = 0.5731 - large effect |
| Key pre-modelling insight | recency_days has structural target leakage (r = 0.72) |

---

## Act 3 — Machine Learning ⏳ *(Coming Soon)*

**Role:** ML Engineer
**Notebook:** [`03_Churn_model.ipynb`](notebooks/03_Churn_model.ipynb)

### The Two-Model Strategy

This notebook deliberately trained two versions of the churn model to identify, quantify, and resolve target leakage.

**The leakage:**
The churn label was defined as `recency_days > 180`. Including `recency_days` as a feature meant the model learned the *definition* of churn rather than the *drivers* of churn. Both models achieved AUC 1.000 - a red flag(big one), not a success.

**The resolution:**
`recency_days` was removed. Models retrained on behavioural features only. AUC dropped to 0.625 - the honest signal.

**The leakage quantified:**
The gap between AUC 1.000 and AUC 0.625 is **0.375 AUC points** - the exact magnitude of the structural leakage.

### Model Performance

| Model | Features | CV AUC | Test AUC |
|-------|----------|--------|----------|
| Logistic Regression - with recency | 8 | 1.0000 | 1.0000 |
| XGBoost — with recency | 8 | 1.0000 | 1.0000 |
| Logistic Regression - no recency | 7 | 0.5481 | 0.5482 |
| **XGBoost — no recency** ← deployed | **7** | **0.6810** | **0.6250** |

### SHAP Feature Importance

| Rank | Feature | Mean \|SHAP\| | Predicted by Analysis C? |
|------|---------|-------------|--------------------------|
| 1 | avg_delivery_delay | 0.3164 | ✗ Unexpected — correlation missed non-linearity |
| 2 | avg_review_score | 0.1261 | ✓ Confirmed |
| 3 | total_spend | 0.1225 | ✓ Confirmed |
| 4 | avg_installments | 0.1042 | — |
| 5 | avg_processing_time | 0.0313 | ✓ Confirmed weak |
| 6 | total_orders | 0.0250 | ✓ Confirmed weak |
| 7 | category_diversity | 0.0136 | ✓ Confirmed negligible |

**The unexpected finding:**
`avg_delivery_delay` ranked first in SHAP despite only a 0.06 linear correlation with churn in Analysis C. XGBoost found a non-linear threshold effect - severely late deliveries predict churn at disproportionately high rates, a pattern invisible to linear correlation. This directly connects Hypothesis 1 (delivery hurts satisfaction) to churn outcomes, making delivery performance both a satisfaction lever and a retention lever.

### ML Pipeline

```
churn_features.csv (96,136 customers × 8 features)
        ↓
Stratified train/test split (80/20, random_state=42)
        ↓
StandardScaler — fit on train only, transform both
        ↓
SMOTE — training set only (2.5:1 → 1:1)
        ↓
Logistic Regression baseline + XGBoost
        ↓
5-fold stratified cross-validation
        ↓
SHAP TreeExplainer
        ↓
joblib serialisation → outputs/models/
```

### Production Readiness

In production this model would be redesigned as:
**"Will this customer make a second purchase within 90 days?"**

Using features measured at the time of the **first purchase only** — delivery experience, order value, product category, processing time - removes the leakage entirely and produces a genuinely forward-looking prediction tool.

---

## Act 4 — Deployment ✅

**Live:** [retaildashboardtl.streamlit.app](https://retaildashboardtl.streamlit.app/)

| Page | Contents |
|------|---------|
| 📊 Business Overview | KPI cards, interactive revenue trend, category breakdown, delivery donut |
| 👥 Customer Segments | RFM segment explorer, customer vs revenue share, scatter map |
| 📋 Statistical Findings | H1 and H2 as finding cards, churn risk by segment, leakage documented |
| 🤖 Churn Predictor | Live XGBoost model - 7 sliders, real-time gauge, model card |

---

## Repository Structure

```
retaillens/
├── data/
│   ├── raw/                          ← Olist CSVs (download from Kaggle)
│   └── processed/
│       ├── master.csv                ← Joined analytical dataset (item level)
│       ├── rfm_base.csv              ← Customer-level aggregations
│       ├── rfm_scored.csv            ← RFM scores + segment labels
│       ├── rfm_with_churn.csv        ← RFM + churn label
│       └── churn_features.csv        ← Final ML feature set (96,136 × 9)
├── notebooks/    
│   ├── 01_exploratory_analysis.ipynb ← Act 1
│   ├── 02_statistical_analysis.ipynb ← Act 2
│   └── 03_Churn_model.ipynb          ← Act 3
├── src/
│   ├── app.py                        ← Streamlit dashboard
|   └── data_cleaning.py              ← 8 tables → master.csv
|
├── outputs/
│   ├── figures/                      ← All charts (PNG)
│   ├── models/
│   │   ├── churn_model.pkl           ← XGBoost (no recency) — powers dashboard
│   │   ├── scaler.pkl                ← StandardScaler (fit on training data)
│   │   └── churn_model_lr.pkl        ← Logistic Regression (reference)
│   └── RetailLens_Executive_Report.pdf
├── requirements.txt
└── README.md
```

---

## How To Run Locally

```bash
# Clone
git clone https://github.com/T-Letuka/retail.git
cd retail

# Environment (Windows)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Download Olist CSVs from Kaggle → data/raw/

# Run notebooks in order: 00 → 01 → 02 → 03

# Launch dashboard
streamlit run src/app.py
```

---

## Skills Demonstrated

| Skill | Where |
|-------|-------|
| Multi-table data engineering | `00_data_cleaning.py` |
| Exploratory data analysis + business communication | `01_exploratory_analysis.ipynb` |
| Executive reporting for non-technical stakeholders | `outputs/RetailLens_Executive_Report.pdf` |
| RFM customer segmentation from scratch | `01_exploratory_analysis.ipynb` Analysis 5 |
| Hypothesis testing (Mann-Whitney U, Chi-squared) | `02_statistical_analysis.ipynb` |
| Statistical vs practical significance distinction | `02_statistical_analysis.ipynb` H2 |
| Target leakage identification and quantification | `02_statistical_analysis.ipynb` |
| Class imbalance handling (SMOTE + ROC-AUC) | `03_Churn_model.ipynb` |
| Model comparison with cross-validation | `03_Churn_model.ipynb` |
| SHAP explainability (global + individual) | `03_Churn_model.ipynb` |
| Dashboard deployment (Streamlit Community Cloud) | [Live URL](https://retaildashboardtl.streamlit.app/) |

---

## The Interview Answer

> *"My churn model achieved a perfect AUC of 1.000. Most people would celebrate that. I recognised it as a red flag , the model had learned the definition of churn rather than the drivers of churn. I traced the leakage to a structural overlap between the recency feature and the 180-day churn threshold, removed the feature, retrained, and produced an honest 0.625 AUC behavioural model. The SHAP analysis then revealed something my correlation analysis had missed , delivery delay has a non-linear threshold effect on churn that linear correlation cannot capture. The leakage was not a mistake. It was the most instructive result in the project."*

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0
scikit-learn>=1.3.0
xgboost>=1.7.0
shap>=0.42.0
imbalanced-learn>=0.11.0
scipy>=1.11.0
statsmodels>=0.14.0
streamlit>=1.28.0
joblib>=1.3.0
jupyter>=1.0.0
ipykernel>=6.0.0
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