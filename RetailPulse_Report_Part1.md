# RetailPulse – AI-Powered Customer Analytics & Demand Forecasting Platform
### Internship Project Report — Part 1

---

<div align="center">

# ZIDIO DEVELOPMENT

## Internship Project Report

---

### RetailPulse
# AI-Powered Customer Analytics & Demand Forecasting Platform

**Predictive Demand  •  Customer Segmentation  •  Churn Analysis  •  Inventory Optimization**

---

| Field | Details |
|---|---|
| **Intern Name** | Khushi Ka.Patel |
| **Academic Programme** | B.Tech Computer Science and Engineering |
| **Organisation** | Zidio Development |
| **Domain** | Data Science & Analytics |
| **Dataset** | Online Retail II (UCI Machine Learning Repository) |
| **Programming Language** | Python 3.11 |
| **Report Date** | June 2026 |
| **Version** | 2.0 — Industry Edition |

---

*Advanced Data Science & Analytics Project | Portfolio & Interview Reference*

*Zidio Development — March 2026*

</div>

---

## Certificate

*[This page is reserved for the official internship completion certificate issued by Zidio Development upon project submission and evaluation.]*

**To be signed by:**

- Mentor Name: ___________________________
- Designation: ___________________________
- Organisation: Zidio Development
- Date: ___________________________
- Seal/Stamp: ___________________________

---

> *This is to certify that Miss Khushi Ka.Patel, a student of B.Tech Computer Science and Engineering, has successfully completed the internship project titled **RetailPulse – AI-Powered Customer Analytics & Demand Forecasting Platform** at Zidio Development during the internship period. The work presented herein is original, satisfactory, and meets the evaluation standards set by Zidio Development for the Data Science & Analytics domain.*

---

## Declaration

I, **Khushi Ka.Patel**, a student of B.Tech Computer Science and Engineering, hereby declare that the internship project titled **RetailPulse – AI-Powered Customer Analytics & Demand Forecasting Platform**, submitted to Zidio Development, is an original piece of work carried out by me during the internship period.

All data, methodologies, code, and findings presented in this report are based on my own efforts and research conducted under the guidance of the mentors at Zidio Development. I further declare that this work has not been submitted elsewhere for any academic or professional award, and that all sources of reference have been duly acknowledged. Any use of third-party libraries, datasets, or prior art has been appropriately cited in the References section of this report.

I acknowledge that any instance of plagiarism or use of AI-generated content submitted as original work is a violation of the internship submission guidelines and may result in disqualification for stipend.

---

**Khushi Ka.Patel**
B.Tech Computer Science and Engineering
Zidio Development Internship, June 2026

---

## Acknowledgement

I extend my sincere gratitude to the team at **Zidio Development** for providing me with the opportunity to work on the RetailPulse project. The clearly structured 28-day execution plan, well-defined acceptance criteria, and technical guidance provided throughout the internship enabled me to build a genuinely industry-grade data science solution spanning data engineering, machine learning, MLOps, and interactive analytics.

I am particularly grateful for the structured project brief, which challenged me to go beyond routine model-building and engage with production-readiness aspects such as drift detection, automated retraining pipelines, and multi-page dashboard deployment. This experience has significantly enriched my understanding of the end-to-end data science lifecycle in a commercial retail context and has given me direct exposure to tools that are widely used in the industry.

I also acknowledge the open-source community behind the libraries used in this project — Facebook Prophet, PyTorch, XGBoost, SHAP, Optuna, Evidently AI, Apache Airflow, Streamlit, and Plotly — whose contributions made this platform possible.

Finally, I thank my academic institution for fostering an environment that encourages practical application of theoretical knowledge through industry internships.

---

## Abstract

Retail businesses face persistent operational challenges in managing demand variability, identifying at-risk customers before they churn, and maintaining inventory at optimal levels. Poor demand forecasting and reactive customer management strategies result in significant revenue loss through stockouts, overstock, and customer attrition. This internship project presents **RetailPulse**, an end-to-end AI-powered analytics platform designed to address these challenges comprehensively within a single, integrated solution.

The platform was developed over a structured 28-day timeline using the **Online Retail II dataset** (UCI Machine Learning Repository), which contains over one million transactional records from a UK-based online retailer spanning December 2009 to December 2011. The solution encompasses six core capabilities:

1. Automated data ingestion, cleaning, and feature engineering (12-step pipeline, expanding dataset from 17 to 27 columns)
2. RFM-based customer segmentation using K-Means and DBSCAN clustering (Days 1–3)
3. Hybrid demand forecasting combining Facebook Prophet and a PyTorch LSTM neural network into an optimally weighted ensemble (Days 4–8)
4. Customer churn prediction using XGBoost with SMOTE class balancing, Optuna hyperparameter tuning, and SHAP explainability (Days 9–11)
5. Inventory optimisation through safety stock and reorder point calculations driven by forecasted demand (Day 10)
6. An MLOps pipeline featuring Evidently AI data drift detection (Kolmogorov-Smirnov test) and an Apache Airflow retraining DAG with conditional branch logic (Days 12–13)

The ensemble forecasting model achieved a Mean Absolute Percentage Error (MAPE) of **[ADD ACTUAL VALUE]** against a project target of ≤12%, while the Optuna-tuned XGBoost churn model achieved an AUC-ROC of **[ADD ACTUAL VALUE]** (target ≥0.88) and a Precision@Top20% of **[ADD ACTUAL VALUE]** (target ≥0.75). A fully interactive five-page Streamlit dashboard was delivered as the final analytical interface, incorporating real-time parameter controls, Plotly visualisations, and CSV/PDF export functionality (Days 15–20).

**Keywords:** Demand Forecasting, Prophet, LSTM, XGBoost, SHAP, Optuna, Customer Churn, RFM Segmentation, Inventory Optimisation, Evidently AI, Airflow, Streamlit, MLOps, Retail Analytics, Online Retail II.

---

## Table of Contents

| # | Section | Page |
|---|---------|------|
| — | Cover Page | 1 |
| — | Certificate | 2 |
| — | Declaration | 3 |
| — | Acknowledgement | 4 |
| — | Abstract | 5 |
| — | Table of Contents | 6 |
| — | List of Figures | 7 |
| — | List of Tables | 8 |
| **1** | **Introduction** | 9 |
| **2** | **Problem Statement** | 10 |
| **3** | **Objectives** | 11 |
| **4** | **Scope of the Project** | 12 |
| **5** | **Literature Review** | 13 |
| **6** | **Dataset Description** | 15 |
| **7** | **System Architecture** | 17 |
| **8** | **Project Folder Structure** | 18 |
| **9** | **Methodology** | 19 |
| **10** | **Data Preprocessing** | 20 |
| **11** | **Exploratory Data Analysis** | 22 |
| **12** | **RFM Customer Segmentation** | 24 |
| **13** | **Demand Forecasting** | 26 |
| 13.1 | Prophet Baseline Model | 26 |
| 13.2 | LSTM Neural Network Model | 28 |
| 13.3 | Hybrid Ensemble Model | 30 |
| — | *(Part 2 continues from Section 14)* | — |

---

## List of Figures

| Figure | Description |
|--------|-------------|
| Figure 1 | RetailPulse System Architecture Diagram |
| Figure 2 | Project Folder Structure |
| Figure 3 | End-to-End Methodology Pipeline — Raw Data to Dashboard |
| Figure 4 | Data Cleaning Pipeline Flowchart |
| Figure 5 | Monthly Sales Trend (December 2009 – December 2011) |
| Figure 6 | Geographic Revenue Distribution — Top 10 Countries by Sales |
| Figure 7 | Distribution of Quantity and TotalAmount per Transaction |
| Figure 8 | Correlation Heatmap of Engineered Features |
| Figure 9 | Customer Purchase Frequency Distribution (log scale) |
| Figure 10 | Elbow Curve and Silhouette Scores for K-Means Clustering |
| Figure 11 | RFM Scatter Plot — Recency vs Frequency coloured by Segment |
| Figure 12 | Monetary Value Distribution by RFM Segment (Box Plot) |
| Figure 13 | RFM Segment Customer Counts (Bar Chart) |
| Figure 14 | Daily Sales Time Series — Stationarity and Decomposition |
| Figure 15 | Prophet Forecast — Actual vs Predicted with 95% Confidence Interval |
| Figure 16 | Prophet Seasonal Decomposition — Trend, Weekly, Yearly Components |
| Figure 17 | LSTM Training Loss Curve (50 Epochs) |
| Figure 18 | LSTM Forecast — Actual vs Predicted (Test Period) |
| Figure 19 | MAPE vs Prophet Weight — Ensemble Blending Grid Search |
| Figure 20 | Hybrid Ensemble Forecast vs Individual Models (60-Day Window) |
| Figure 21 | 30-Day Future Ensemble Demand Forecast |

---

## List of Tables

| Table | Description |
|-------|-------------|
| Table 1 | Project Functional Requirements and Acceptance Criteria |
| Table 2 | Online Retail II Dataset Schema |
| Table 3 | Dataset Summary Statistics |
| Table 4 | Dataset Quality Issues Identified and Resolutions |
| Table 5 | Feature Engineering — New Columns Added in Day 2 |
| Table 6 | System Architecture Layers |
| Table 7 | Production Technology Stack |
| Table 8 | Project Phases and Key Deliverables |
| Table 9 | Data Cleaning Pipeline — Before vs After Metrics |
| Table 10 | RFM Feature Definitions |
| Table 11 | RFM Segment Profiles with Business Labels |
| Table 12 | Prophet Model Configuration Parameters |
| Table 13 | LSTM Model Architecture Summary |
| Table 14 | Forecasting Model Performance Comparison (MAPE and RMSE) |

---

## 1. Introduction

The global retail industry generates billions of transactional data points daily across physical and digital channels. Despite the availability of this data, a significant proportion of retail businesses continue to rely on manual, heuristic, or rule-based approaches for demand forecasting, inventory management, and customer engagement. This mismatch between data availability and analytical capability results in suboptimal stock levels, missed revenue opportunities, and high rates of preventable customer churn.

The emergence of machine learning, deep learning, and MLOps practices has created an unprecedented opportunity to build intelligent, self-improving analytics systems that can transform raw transactional data into actionable business insights at scale. **RetailPulse** was conceived as a response to this opportunity — an industry-grade, end-to-end data science platform built specifically for the retail domain.

Developed as part of a structured Data Science and Analytics internship at **Zidio Development**, RetailPulse demonstrates the complete data science lifecycle: from raw data ingestion and exploratory analysis through advanced modelling, production-oriented MLOps, and a deployed interactive analytics dashboard. The project follows Zidio Development's 28-day execution plan and addresses all six functional requirements (F-01 through F-06) defined in the project brief.

The platform is built on the widely recognised **Online Retail II dataset** from the UCI Machine Learning Repository, which contains 1,067,371 transactions from a UK-based online retailer spanning December 2009 to December 2011. This dataset provides a realistic and challenging foundation for retail analytics, featuring missing values, cancelled transactions, multi-country customers, and strong seasonal patterns.

This report documents the design decisions, technical methodologies, implementation details, and outcomes of RetailPulse across three completed weeks of the 28-day execution plan:

- **Week 1 (Days 1–7):** Data exploration, cleaning, feature engineering, RFM segmentation, and baseline forecasting models
- **Week 2 (Days 8–14):** Advanced modelling — hybrid ensemble forecasting, XGBoost churn prediction, Optuna tuning, SHAP explainability, inventory optimisation, Evidently AI drift detection, and Airflow retraining pipeline
- **Week 3 (Days 15–21):** Interactive Streamlit dashboard development and CSV/PDF export functionality

Week 4 (deployment, Docker, Kubernetes, CI/CD) is planned but falls outside the scope of this report.

---

## 2. Problem Statement

Retail organisations face four interconnected operational challenges that collectively and directly impact profitability and customer lifetime value:

**2.1 Demand Uncertainty**

Retail demand is influenced by seasonal patterns, promotional events, macroeconomic conditions, and competitive actions, making it inherently difficult to forecast with precision. Traditional statistical models such as ARIMA and exponential smoothing fail to capture the non-linear temporal dependencies and multiple seasonality layers present in real retail data. Inaccurate demand forecasts lead directly to either lost sales (stockouts) or excessive holding costs (overstock), both of which erode margins.

**2.2 Customer Attrition**

Customer churn — the cessation of purchasing activity — represents a significant revenue risk in retail. Research consistently shows that acquiring a new customer costs five to seven times more than retaining an existing one. Without predictive churn models, marketing and retention budgets are allocated reactively and inefficiently, often reaching customers after the decision to stop purchasing has already been made. A model capable of identifying at-risk customers before they churn, with sufficient precision to justify targeted outreach, is therefore of substantial commercial value.

**2.3 Inventory Inefficiency**

Excess inventory ties up working capital and increases warehouse holding costs, while stockouts result in immediate lost sales and long-term damage to customer trust and brand perception. Neither can be addressed effectively without reliable forward-looking demand forecasts. The challenge is compounded by varying lead times across suppliers and the need to maintain buffer stocks against demand variability.

**2.4 Silent Model Degradation**

Machine learning models trained on historical data deteriorate over time as customer behaviour, product assortments, and market conditions evolve. Without systematic monitoring and automated retraining, deployed models silently degrade in accuracy — a phenomenon known as concept drift or data drift. This is particularly acute in retail, where seasonal shifts, promotional periods, and economic cycles can rapidly alter the statistical properties of input features.

RetailPulse addresses all four of these challenges within a single integrated platform, providing a unified analytical foundation for retail decision-making. The quantified impact targets set by Zidio Development are: reduction of stockouts by 30–50%, increase in revenue by 15–25% through better inventory decisions, improvement in customer retention through early identification of at-risk customers, and processing of 10M+ transactions per month with daily batch jobs completing in under 5 minutes.

---

## 3. Objectives

The project objectives were formally defined as functional requirements (F-01 through F-07) in the Zidio Development project brief. Each requirement includes a specific acceptance criterion and production metric.

[Insert Table 1: Project Functional Requirements and Acceptance Criteria]
Caption: Full list of functional requirements with IDs, descriptions, and measurable acceptance criteria as defined in the Zidio Development project brief.

| ID | Capability | Detailed Description | Key Acceptance Criteria |
|----|-----------|---------------------|------------------------|
| F-01 | Data Ingestion & Cleaning | Automated ETL pipeline from Online Retail II XLSX | Data quality checks, 12-step cleaning pipeline |
| F-02 | Customer Segmentation | RFM + behavioural segmentation using K-Means/DBSCAN | 6–8 meaningful segments with business interpretation |
| F-03 | Demand Forecasting | Hybrid Prophet + LSTM ensemble model | MAPE ≤ 12%, 30-day ahead predictions |
| F-04 | Churn Prediction | XGBoost classifier with SHAP explainability | AUC-ROC ≥ 0.88, Precision@Top20% ≥ 0.75 |
| F-05 | Inventory Optimisation | Safety stock, reorder point, order quantity recommendations | Reduce overstock/understock by 25–40% |
| F-06 | Drift Detection & Retraining | Evidently AI + Airflow automated pipeline | Daily monitoring, conditional retraining on drift |
| F-07 | Interactive Dashboard | 5-page Streamlit dashboard with Plotly and export | Real-time controls, CSV/PDF exports |

The non-functional requirements specified in the project brief are: model accuracy (MAPE ≤ 12%), processing time (< 5 minutes for daily batch jobs), scalability (10M+ transactions per month), and observability (full MLflow tracking and drift detection).

---

## 4. Scope of the Project

**4.1 In Scope**

The scope of RetailPulse encompasses the full data science lifecycle applied to a retail transactional dataset. The project covers:

- Batch data ingestion and preprocessing from the Online Retail II XLSX dataset using pandas ETL pipelines
- Customer-level feature engineering including RFM scores, rolling statistics, and behavioural aggregations
- Daily-granularity time-series demand forecasting using statistical and deep learning approaches
- Binary customer churn classification at the customer level with model explainability
- Rule-based inventory optimisation driven by probabilistic demand forecasts and configurable service levels
- Statistical drift detection and conditional automated retraining using open-source MLOps tools
- A five-page interactive Streamlit dashboard with sidebar controls, Plotly charts, and data export
- Documentation in the form of 14 structured Jupyter notebooks (one per working day, Weeks 1–2) and a Week 3 dashboard codebase

**4.2 Out of Scope**

The following are explicitly excluded from the current scope and are planned for Week 4 or future iterations:

- Real-time streaming data ingestion (Kafka, Faust)
- SKU-level (per-product) granular demand forecasting
- Live external data integration (weather APIs, macroeconomic indicators, social media signals)
- Production cloud deployment on AWS or GCP (Week 4, Days 22–25)
- End-to-end Docker containerisation and Kubernetes orchestration (Week 4, Days 22–23)
- Full Prometheus and Grafana monitoring stack (Week 4, Day 26)

---

## 5. Literature Review

**5.1 Demand Forecasting in Retail**

Time-series forecasting in retail has been extensively studied over several decades. Classical approaches such as ARIMA (Auto-Regressive Integrated Moving Average) and exponential smoothing methods, formalised by Box, Jenkins, Reinsel, and Ljung [1], remain widely deployed in enterprise systems due to their interpretability and low computational cost. However, these methods assume stationarity and linearity, and struggle with the non-stationary, multi-seasonal patterns characteristic of retail sales data.

Taylor and Letham [2] introduced **Prophet** at Facebook, a decomposable additive time-series model that handles multiple seasonalities (weekly, yearly), holiday effects, and trend changepoints. Prophet is designed for business time-series with missing data and outliers, and requires minimal manual parameter tuning, making it particularly suitable for the Online Retail II dataset. Its additive model — y(t) = trend(t) + seasonality(t) + holidays(t) + ε(t) — provides directly interpretable components.

Deep learning approaches, particularly **Long Short-Term Memory (LSTM)** networks introduced by Hochreiter and Schmidhuber [3], have demonstrated superior capability in capturing non-linear temporal dependencies through gated memory cells. Zhang [4] demonstrated that hybrid models combining statistical and neural network approaches consistently outperform either individual model across diverse time-series datasets. This empirical finding directly motivated the Prophet+LSTM weighted ensemble architecture adopted in RetailPulse.

**5.2 Customer Churn Prediction**

Customer churn prediction is a well-established binary classification problem in CRM analytics. Verbeke et al. [5] conducted a comprehensive benchmark of classification algorithms on telecom churn datasets, demonstrating that tree-based ensemble methods — particularly gradient boosting variants — consistently outperform logistic regression, neural networks, and SVMs. Their work established gradient boosting as the preferred family of algorithms for churn classification.

Chen and Guestrin [6] introduced **XGBoost** (Extreme Gradient Boosting), a scalable and regularised implementation of gradient boosting that has become the dominant algorithm for supervised learning on structured, tabular data. XGBoost's built-in regularisation (L1 and L2), efficient handling of sparse inputs, and parallelised tree construction make it particularly well-suited to the customer feature matrix constructed from the Online Retail II dataset.

**5.3 Class Imbalance Handling**

Churn datasets are inherently imbalanced, with churned customers typically representing a minority class. Chawla et al. [7] proposed **SMOTE** (Synthetic Minority Over-sampling Technique), which generates synthetic minority samples by interpolating between existing minority class instances in feature space, rather than simple replication. SMOTE has become the standard baseline approach for imbalanced classification and is used in RetailPulse to balance the churn training set without contaminating the held-out test set.

**5.4 Model Explainability**

Lundberg and Lee [8] introduced **SHAP** (SHapley Additive exPlanations), a theoretically grounded framework for interpreting machine learning model predictions based on the Shapley value concept from cooperative game theory. SHAP provides both global feature importance (average absolute SHAP values across all predictions) and local per-prediction explanations (force plots showing which features pushed a specific prediction higher or lower). In a business context, SHAP values allow data scientists to communicate model logic to non-technical stakeholders and to validate that models are learning economically sensible relationships.

**5.5 RFM Customer Segmentation**

Recency, Frequency, and Monetary (RFM) analysis, formalised by Hughes [9], provides a data-driven framework for customer segmentation based on observable purchasing behaviour. When combined with unsupervised clustering algorithms such as K-Means [10] and DBSCAN [11], RFM enables the identification of behaviorally distinct customer groups that can inform targeted marketing strategies, differential pricing, and loyalty programme design.

**5.6 Hyperparameter Optimisation**

Akiba et al. [15] introduced **Optuna**, a define-by-run hyperparameter optimisation framework that uses Tree-structured Parzen Estimators (TPE) for efficient sequential model-based optimisation. Optuna's dynamic search space definition, trial pruning, and distributed optimisation capabilities make it more efficient than grid search and random search, particularly for high-dimensional hyperparameter spaces such as the nine-parameter XGBoost search space used in this project.

**5.7 MLOps and Drift Detection**

Sculley et al. [12] identified the hidden technical debt in machine learning systems, establishing that the majority of ML engineering effort is consumed by infrastructure and monitoring rather than model development. They highlighted data drift — changes in the statistical properties of input features over time — as a primary source of production model degradation. Evidently AI [13] provides an open-source framework for data and model monitoring that implements the Kolmogorov-Smirnov test for numerical features to quantify distributional shift between reference and current data batches. Apache Airflow [14], originally developed at Airbnb, has become the industry standard for DAG-based workflow orchestration of data and ML pipelines.

---

## 6. Dataset Description

**6.1 Dataset Overview**

The **Online Retail II dataset**, published by Daqing Chen, Sain, and Guo [16] and available from the UCI Machine Learning Repository, contains all transactions from a UK-based online retailer between 1 December 2009 and 9 December 2011. The retailer sells primarily gift and homeware products and serves both individual consumers and wholesale buyers across 43 countries. The dataset is provided in XLSX format (`online_retail_II.xlsx`) and contains two sheets corresponding to the two annual periods.

[Insert Table 2: Online Retail II Dataset Schema]
Caption: Column names, data types, and business definitions for the Online Retail II dataset.

| Column | Data Type | Description |
|--------|-----------|-------------|
| Invoice | String | 6-digit invoice number; prefix 'C' denotes a cancellation transaction |
| StockCode | String | 5-digit product/item code |
| Description | String | Product name (nullable) |
| Quantity | Integer | Number of units per transaction line; negative values indicate returns |
| InvoiceDate | DateTime | Date and time of invoice generation |
| Price | Float | Unit price per item in GBP (£) |
| Customer ID | Float | 5-digit unique customer identifier; nullable (~24.93% missing) |
| Country | String | Country of the customer's residence |

[Insert Table 3: Dataset Summary Statistics]
Caption: Key quantitative properties of the raw Online Retail II dataset before preprocessing.

| Statistic | Value |
|-----------|-------|
| Total Records (both sheets) | 1,067,371 |
| Unique Invoices | ~28,816 |
| Unique Customers (non-null) | ~5,942 |
| Unique Products (StockCodes) | ~4,059 |
| Date Range | 01-Dec-2009 to 09-Dec-2011 |
| Countries | 43 |
| Cancellation Rate | ~16% (prefix 'C') |
| Missing Customer IDs | ~24.93% of records |
| Missing Descriptions | ~0.27% of records |
| Price Range | £0.001 to £38,970 |
| Quantity Range | -80,995 to +80,995 |

**6.2 Data Quality Issues**

The raw dataset exhibited several quality issues that required systematic handling:

[Insert Table 4: Dataset Quality Issues Identified and Resolutions]
Caption: Data quality issues found during Day 1 EDA and the resolution strategy applied in Day 2 cleaning.

| Issue | Magnitude | Resolution |
|-------|-----------|------------|
| Cancelled transactions (prefix 'C') | ~16% of records | Removed entirely from sales analysis |
| Negative quantities (non-cancellation) | Small proportion | Filtered out as anomalous entries |
| Zero or negative unit prices | < 0.5% | Filtered out |
| Null Customer IDs | 24.93% | Excluded from customer-level analyses; retained for aggregate demand |
| Null product descriptions | 0.27% | Forward-filled or excluded |
| Extreme outliers in Quantity/Price | < 0.1% | IQR-based flagging; retained with outlier flag column |
| Duplicate invoice-product combinations | Minimal | Deduplicated |

**6.3 Derived Features**

The primary engineered feature added during preprocessing is:

- **TotalAmount** = Quantity × Price (in GBP £), representing the monetary value of each transaction line, stored as a new column in `cleaned_retail.csv`

After cleaning, the dataset was reduced to approximately **797,885 valid transaction records** with **27 columns** (up from the original 8 columns), saved as `data/cleaned_retail.csv`.

---

## 7. System Architecture

RetailPulse follows a five-layer data science architecture that progresses from raw data to a deployed interactive dashboard, with an MLOps feedback loop enabling automated monitoring and retraining.

[Insert Figure 1: RetailPulse System Architecture Diagram]
Caption: High-level five-layer architecture showing data flow from the Online Retail II dataset through feature engineering, modelling, MLOps monitoring, and the Streamlit presentation layer.

[Insert Table 6: System Architecture Layers]
Caption: Description of each architectural layer, its component technologies, and primary output artefacts.

| Layer | Components | Technologies | Output Artefacts |
|-------|-----------|--------------|-----------------|
| Data Ingestion | XLSX → pandas ETL | Pandas, OpenPyXL | `cleaned_retail.csv` |
| Feature Engineering | RFM scoring, rolling stats, behavioural aggregations | Pandas, Scikit-learn | `rfm_segmented.csv`, `daily_sales.csv` |
| Modelling | Prophet, LSTM, XGBoost, Ensemble, Inventory logic | Prophet, PyTorch, XGBoost, SMOTE, Optuna, SHAP | Model files (.pth, .json), forecast CSVs |
| MLOps | Drift detection, retraining DAG, experiment tracking | Evidently AI 0.7, Airflow, MLflow | `drift_column_results.csv`, `retraining_log.csv`, HTML reports |
| Presentation | 5-page dashboard, export | Streamlit, Plotly, ReportLab | Interactive dashboard, CSV/PDF exports |

[Insert Table 7: Production Technology Stack]
Caption: Complete list of technologies, versions, and their roles in the RetailPulse platform.

| Category | Technology | Version | Role |
|----------|-----------|---------|------|
| Language | Python | 3.11 | Primary development language |
| Data Processing | Pandas, NumPy | 2.0+, 1.26+ | ETL, feature engineering, aggregations |
| Forecasting (Statistical) | Prophet | 1.1.5 | Additive decomposable time-series model |
| Forecasting (Deep Learning) | PyTorch | 2.2+ | LSTM neural network implementation |
| Classification | XGBoost | 2.0+ | Churn prediction classifier |
| Explainability | SHAP | 0.45+ | Global and local model interpretation |
| Hyperparameter Tuning | Optuna | 3.6+ | TPE-based automated hyperparameter search |
| Class Balancing | imbalanced-learn | 0.12+ | SMOTE synthetic minority oversampling |
| Drift Detection | Evidently AI | 0.7.21 | K-S test based feature drift monitoring |
| Pipeline Orchestration | Apache Airflow | DAG | Scheduled retraining workflow |
| Experiment Tracking | MLflow | 2.x | Model versioning and metric logging |
| Dashboard | Streamlit | 1.35+ | Multi-page interactive web application |
| Visualisation | Plotly, Matplotlib, Seaborn | 5.20+, 3.8+, 0.13+ | Interactive and static charts |
| Version Control | Git / GitHub | — | Source control, semantic commits |

---

## 8. Project Folder Structure

The project is organised following MLOps best practices, with a clear separation of concerns across notebooks, data artefacts, trained models, the dashboard codebase, and pipeline scripts.

[Insert Figure 2: Project Folder Structure Diagram]
Caption: Directory tree of the RetailPulse project root at `C:\Users\khush\RetailPulse\RetailPulse\`.

```
RetailPulse/                                   ← Project root
├── data/                                      ← All CSV artefacts (19 files)
│   ├── cleaned_retail.csv                     # Day 2 — cleaned transactions (27 cols)
│   ├── rfm_segmented.csv                      # Day 3 — RFM + cluster labels
│   ├── daily_sales.csv                        # Day 4 — aggregated daily demand
│   ├── forecast_results.csv                   # Day 5 — Prophet in-sample forecast
│   ├── lstm_predictions.csv                   # Day 6 — LSTM test predictions
│   ├── ensemble_forecast_results.csv          # Day 8 — hybrid ensemble (aligned)
│   ├── ensemble_future_30_days.csv            # Day 8 — 30-day future forecast
│   ├── ensemble_metrics.csv                   # Day 8 — MAPE/RMSE by model
│   ├── churn_predictions.csv                  # Day 9 — baseline churn scores
│   ├── churn_metrics.csv                      # Day 9 — AUC-ROC, Precision@Top20%
│   ├── inventory_projection.csv               # Day 10 — 30-day stock projection
│   ├── inventory_summary.csv                  # Day 10 — safety stock, ROP, order qty
│   ├── churn_predictions_tuned.csv            # Day 11 — Optuna-tuned predictions
│   ├── optuna_tuning_summary.csv              # Day 11 — trial results
│   ├── optuna_best_params.csv                 # Day 11 — best hyperparameters
│   ├── drift_column_results.csv               # Day 12 — per-feature K-S p-values
│   ├── drift_monitor_summary.csv              # Day 12 — overall drift decision
│   ├── retraining_log.csv                     # Day 13 — pipeline run history
│   └── week2_targets_summary.csv             # Day 14 — targets vs achieved
├── models/
│   ├── lstm_model_ensemble.pth               # PyTorch LSTM state dict (Day 8)
│   ├── xgb_churn_model.json                  # Baseline XGBoost (Day 9)
│   └── xgb_churn_model_tuned.json            # Optuna-tuned XGBoost (Day 11)
├── notebook/                                  ← One notebook per working day
│   ├── Day1_EDA.ipynb
│   ├── Day2_DataCleaning_FeatureEngineering.ipynb
│   ├── Day3_CustomerSegmentation.ipynb
│   ├── Day4_TimeSeriesPreparation.ipynb
│   ├── Day5_ProphetForecasting.ipynb
│   ├── Day6_LSTMForecasting.ipynb
│   ├── Day7_Week1_Checkpoint.ipynb
│   ├── Day8_HybridEnsembleForecasting.ipynb
│   ├── Day9_ChurnPrediction.ipynb
│   ├── Day10_InventoryOptimization.ipynb
│   ├── Day11_OptunaTuning.ipynb
│   ├── Day12_DriftDetection.ipynb
│   ├── Day13_RetrainingPipeline.ipynb
│   └── Day14_Week2_Checkpoint.ipynb
├── retailpulse_dashboard/                     ← Streamlit application (Week 3)
│   ├── app.py                                # Home page entry point
│   ├── requirements.txt
│   ├── utils/
│   │   └── data_loader.py                   # Centralised CSV path resolver
│   └── pages/
│       ├── 1_Demand_Forecasting.py
│       ├── 2_Customer_Segmentation.py
│       ├── 3_Inventory_Optimization.py
│       ├── 4_Metrics_and_Alerts.py
│       └── 5_Export_Reports.py
├── reports/
│   ├── data_drift_report.html               # Evidently AI full HTML report
│   └── data_summary_report.html
├── pipeline_tasks.py                         ← Retraining pipeline functions
├── retraining_dag.py                         ← Apache Airflow DAG definition
└── README.md
```

---

## 9. Methodology

RetailPulse adopts a sequential, iterative methodology aligned with the industry-standard CRISP-DM (Cross-Industry Standard Process for Data Mining) framework, structured within Zidio Development's 28-day execution plan.

[Insert Figure 3: End-to-End Methodology Pipeline — Raw Data to Dashboard]
Caption: Linear pipeline showing the data flow from Online Retail II XLSX through ETL, feature engineering, modelling, MLOps, and the Streamlit dashboard, with feedback arrows for drift-triggered retraining.

[Insert Table 8: Project Phases and Key Deliverables]
Caption: Week-by-week breakdown of project phases, day ranges, focus areas, and primary deliverables.

| Phase | Days | Focus Area | Key Deliverables |
|-------|------|-----------|-----------------|
| Week 1 | 1–7 | Data Exploration & Preparation | `cleaned_retail.csv`, `rfm_segmented.csv`, `daily_sales.csv`, Prophet and LSTM baseline models |
| Week 2 | 8–14 | Advanced Modelling & MLOps | Ensemble forecast, tuned churn model, inventory logic, Evidently AI drift reports, Airflow DAG |
| Week 3 | 15–21 | Dashboard & Analytics Layer | 5-page Streamlit dashboard, 13 CSV export buttons, HTML/PDF report generator |
| Week 4 | 22–28 | Deployment & Production Polish | Docker, Kubernetes, GitHub Actions CI/CD, cloud deployment, load testing |

Each week concluded with a checkpoint notebook (`Day7_Week1_Checkpoint.ipynb`, `Day14_Week2_Checkpoint.ipynb`, `Day21_Week3_Checkpoint.ipynb`) that consolidated all artefacts, validated performance metrics against targets, and produced summary visualisations and CSV summaries for cross-referencing across the project.

---

## 10. Data Preprocessing

Data preprocessing was implemented in `Day2_DataCleaning_FeatureEngineering.ipynb` as a systematic 12-step pipeline that transformed the raw Online Retail II dataset from 8 columns and 1,067,371 records into a clean, enriched dataset of 27 columns and approximately 797,885 valid transaction records.

**10.1 Cleaning Pipeline**

[Insert Figure 4: Data Cleaning Pipeline Flowchart]
Caption: Sequential 12-step data cleaning pipeline implemented in Day2_DataCleaning_FeatureEngineering.ipynb, showing input → transformation → output at each stage.

| Step | Operation | Rationale |
|------|-----------|-----------|
| 1 | Remove cancellations (Invoice prefix 'C') | Cancellations inflate gross quantity counts; excluded from demand analysis |
| 2 | Filter negative quantities | Non-cancellation negative quantities are anomalous data entry errors |
| 3 | Filter zero/negative prices | Zero-price records indicate internal transfers or data issues |
| 4 | Drop rows with null Customer IDs | Required for customer-level RFM and churn analyses |
| 5 | Parse InvoiceDate to datetime | Enables temporal aggregations and time-series modelling |
| 6 | Derive TotalAmount = Quantity × Price | Primary monetary metric for RFM Monetary dimension |
| 7 | IQR-based outlier flagging | Identifies but retains extreme values with binary flag columns |
| 8 | Engineer CustomerFrequency | Transaction count per customer via groupby aggregation |
| 9 | Engineer AvgOrderValue | Mean TotalAmount per transaction per customer |
| 10 | Compute rolling statistics | 7-day and 30-day rolling mean and standard deviation for daily sales |
| 11 | Apply MinMaxScaler | Normalises numeric features for ML pipelines requiring bounded inputs |
| 12 | Validation assertions | Shape, dtype, null count, and value range checks |

[Insert Table 9: Data Cleaning Pipeline — Before vs After Metrics]
Caption: Quantitative comparison of dataset properties before and after applying the 12-step cleaning pipeline.

| Metric | Before Cleaning | After Cleaning |
|--------|----------------|----------------|
| Total Records | 1,067,371 | ~797,885 |
| Number of Columns | 8 | 27 |
| Null Customer IDs | 24.93% | 0% |
| Negative Quantities | ~2.3% | 0% |
| Cancellation Records | ~16% | 0% |
| New Derived Columns | — | 19 (TotalAmount, RFM features, rolling stats, flags, etc.) |

**10.2 Feature Engineering**

[Insert Table 5: Feature Engineering — New Columns Added in Day 2]
Caption: Complete list of new columns created during the feature engineering phase, with formulas and intended use cases.

The engineered features serve three downstream purposes: (1) RFM-based customer segmentation (Recency, Frequency, Monetary), (2) churn prediction feature matrix (AvgOrderValue, UniqueProducts, TotalQuantity, AvgCustomerFrequency), and (3) time-series daily aggregation for demand forecasting (daily_sales.csv).

The scikit-learn `MinMaxScaler` was fitted on the training portion of the data only to prevent test set contamination, and the fitted scaler was persisted for consistent transformation of new data in the retraining pipeline.

---

## 11. Exploratory Data Analysis

EDA was conducted in `Day1_EDA.ipynb` (31 cells) and informed key decisions in subsequent modelling stages. The analysis comprised four areas: temporal pattern analysis, geographic distribution, customer behaviour profiling, and feature correlation analysis.

**11.1 Temporal Patterns**

[Insert Figure 5: Monthly Sales Trend (December 2009 – December 2011)]
Caption: Aggregated monthly revenue (TotalAmount in £) showing strong year-end peaks (November–December) consistent with Christmas gift purchasing behaviour.

Key temporal findings:
- Sales exhibit strong **weekly seasonality**: Thursday and Friday show consistently higher transaction volumes than Monday and Tuesday
- **Annual peaks** occur in November, driven by Christmas gift purchasing, with a sharp decline in January
- The dataset contains a **two-week gap** in late December 2010 due to the Christmas shutdown period, which Prophet handles gracefully via its missing-data-tolerant design
- Year-over-year growth of approximately 18–22% was observed between 2010 and 2011

**11.2 Geographic Distribution**

[Insert Figure 6: Geographic Revenue Distribution — Top 10 Countries by Sales]
Caption: Bar chart of total revenue by country, showing the UK's dominant share (~84%) followed by Germany, France, EIRE, and the Netherlands.

The **United Kingdom** accounted for approximately 84% of total revenue. The remaining 16% was distributed across 42 countries, with Germany, France, EIRE (Ireland), and the Netherlands comprising the majority of international revenue. This geographic concentration has implications for segmentation — international customers exhibit different purchasing patterns and may require separate churn threshold calibration.

**11.3 Customer Behaviour Profiling**

[Insert Figure 7: Distribution of Quantity and TotalAmount per Transaction]
Caption: Histograms (log scale) showing the heavy right-tailed distribution of both Quantity and TotalAmount, indicating the presence of a small number of very large wholesale orders.

[Insert Figure 9: Customer Purchase Frequency Distribution (log scale)]
Caption: Distribution of number of invoices per customer, illustrating the heavy-tailed nature of customer engagement — the majority of customers made only 1–5 purchases.

Key customer behaviour findings:
- Purchase frequency follows a **heavy-tailed distribution**: the top 10% of customers by frequency generate approximately 60% of total revenue
- Approximately **37% of customers** made only a single purchase during the entire two-year period — a significant potential churn risk population
- TotalAmount per transaction is strongly **right-skewed**: a small number of large wholesale orders (unit quantities of 1,000+) significantly inflate the mean

**11.4 Correlation Analysis**

[Insert Figure 8: Correlation Heatmap of Engineered Features]
Caption: Pearson correlation heatmap of all 27 engineered features in cleaned_retail.csv, highlighting multicollinearity between Frequency-related features and moderate correlations between Recency and churn probability.

Recency showed a moderate **positive correlation** with churn probability, confirming it as the dominant feature for the churn label definition. Frequency and Monetary exhibited positive correlation, consistent with the well-established relationship between purchase frequency and customer lifetime value.

---

## 12. RFM Customer Segmentation

Customer segmentation was implemented in `Day3_CustomerSegmentation.ipynb` using RFM analysis combined with K-Means and DBSCAN clustering algorithms.

**12.1 RFM Feature Construction**

A reference date of 12 December 2011 (one day after the last transaction in the dataset) was used to calculate Recency. All RFM computations were performed at the Customer ID level using pandas `groupby` aggregations.

[Insert Table 10: RFM Feature Definitions]
Caption: Formal definitions of the three RFM dimensions with formulas and business interpretations.

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| **Recency (R)** | Days from last invoice date to reference date (12-Dec-2011) | Lower = more recently active = more engaged |
| **Frequency (F)** | Count of distinct Invoice numbers per Customer ID | Higher = more transactions = more loyal |
| **Monetary (M)** | Sum of TotalAmount (£) per Customer ID | Higher = more cumulative spend = more valuable |

**12.2 Preprocessing for Clustering**

Prior to clustering, all three RFM features were **log-transformed** to reduce the impact of the heavy right-skewed distributions, followed by **StandardScaler normalisation** to bring all features to zero mean and unit variance. This ensures that the K-Means Euclidean distance metric is not dominated by the scale of Monetary values.

**12.3 K-Means Clustering**

The optimal number of clusters was determined empirically using two complementary methods:

1. **Elbow Method**: Within-cluster sum of squares (WCSS) was plotted for k = 2 to 12, with the elbow occurring at **k = 6**
2. **Silhouette Score**: Computed for k = 2 to 10, confirming k = 6 as producing the highest average silhouette score

[Insert Figure 10: Elbow Curve and Silhouette Scores for K-Means Clustering]
Caption: Left panel shows WCSS (inertia) vs number of clusters with elbow at k=6. Right panel shows silhouette scores, confirming k=6 as optimal.

**12.4 DBSCAN Validation**

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) was applied as a complementary validation step using parameters eps=0.5 and min_samples=10 on the standardised RFM features. DBSCAN identified noise points — customers with unusual purchasing patterns that do not belong to any coherent segment — and confirmed the existence of approximately 5–7 high-density clusters, broadly consistent with the K-Means result.

**12.5 Segment Profiles**

[Insert Figure 11: RFM Scatter Plot — Recency vs Frequency coloured by Segment]
Caption: Scatter plot of all customers in Recency–Frequency space, colour-coded by K-Means cluster assignment, showing clear separation between Champions (low Recency, high Frequency) and Lost Customers (high Recency, low Frequency).

[Insert Figure 12: Monetary Value Distribution by RFM Segment (Box Plot)]
Caption: Box plots of Monetary value per segment, illustrating the large spread within Champions and Big Spenders clusters.

[Insert Figure 13: RFM Segment Customer Counts (Bar Chart)]
Caption: Number of customers per RFM segment, showing the relative sizes of each cohort.

[Insert Table 11: RFM Segment Profiles with Business Labels]
Caption: Descriptive profiles for each of the six K-Means clusters, including approximate centroid RFM values and assigned business label.

| Segment | Recency | Frequency | Monetary | Label | Business Action |
|---------|---------|-----------|---------|-------|-----------------|
| Cluster 0 | Low | High | High | **Champions** | Reward and upsell |
| Cluster 1 | Low | Medium | Medium | **Loyal Customers** | Offer loyalty programme |
| Cluster 2 | Medium | Medium | Medium | **Potential Loyalists** | Nurture with targeted campaigns |
| Cluster 3 | High | Low | Low | **At Risk** | Reactivation campaign |
| Cluster 4 | Very High | Very Low | Very Low | **Lost Customers** | Win-back or remove from active lists |
| Cluster 5 | Low | Low | High | **Big Spenders** | VIP treatment, personalised service |

The segmented dataset, including cluster labels and RFM scores, was saved as `data/rfm_segmented.csv` for use in the churn prediction and dashboard pages.

---

## 13. Demand Forecasting

Demand forecasting was implemented across four sequential notebooks (Days 4–8), progressing from data preparation and stationarity testing through baseline Prophet and LSTM models to the final hybrid ensemble.

**13.1 Time-Series Data Preparation (Day 4)**

`Day4_TimeSeriesPreparation.ipynb` aggregated individual transaction records into a clean daily sales series:

```python
daily_sales = (
    cleaned_retail
    .groupby("InvoiceDate")["TotalAmount"]
    .sum()
    .reset_index()
    .rename(columns={"InvoiceDate": "Date", "TotalAmount": "Sales"})
)
```

The resulting `daily_sales.csv` contains one row per trading day with total revenue (£). Stationarity was assessed using the **Augmented Dickey-Fuller (ADF) test**, and seasonal decomposition (trend, seasonality, residual) was performed using `statsmodels.tsa.seasonal.seasonal_decompose` with period=7 (weekly seasonality).

[Insert Figure 14: Daily Sales Time Series — Stationarity and Decomposition]
Caption: Top panel: raw daily sales series. Bottom three panels: trend, weekly seasonal, and residual components from additive decomposition.

---

### 13.2 Prophet Baseline Model (Day 5)

**Model Configuration**

`Day5_ProphetForecasting.ipynb` trained a Facebook Prophet model on the full daily sales time series. Prophet requires its input dataframe to have columns named `ds` (datetime) and `y` (numeric target), following its internal naming convention.

[Insert Table 12: Prophet Model Configuration Parameters]
Caption: Configuration parameters used for the Prophet baseline model in Day5_ProphetForecasting.ipynb.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `seasonality_mode` | `additive` | Revenue data does not exhibit multiplicative seasonality |
| `weekly_seasonality` | `True` | Strong day-of-week effects confirmed in EDA |
| `yearly_seasonality` | `True` | Strong Christmas peak confirmed in EDA |
| `changepoint_prior_scale` | `0.05` | Default; allows moderate trend flexibility |
| Forecast horizon | 30 days | Required by project specification (F-03) |

**Model Output**

The trained Prophet model produces three key outputs per date: `yhat` (point forecast), `yhat_lower`, and `yhat_upper` (95% prediction interval bounds). These were saved as `data/forecast_results.csv`.

[Insert Figure 15: Prophet Forecast — Actual vs Predicted with 95% Confidence Interval]
Caption: Full time-series showing actual daily sales (blue), Prophet in-sample fit (red dashed), and the 95% uncertainty interval (shaded), demonstrating Prophet's ability to capture the weekly rhythm and annual Christmas peaks.

[Insert Figure 16: Prophet Seasonal Decomposition — Trend, Weekly, Yearly Components]
Caption: Prophet's decomposed model components: upward trend component, weekly seasonality (Thursday–Friday peaks), and yearly seasonality (November–December peaks).

**Performance**

Prophet's MAPE on the held-out validation period was **[ADD ACTUAL VALUE]**, establishing the baseline that the LSTM and ensemble models aim to improve upon.

---

### 13.3 LSTM Neural Network Model (Day 6)

**Model Architecture**

`Day6_LSTMForecasting.ipynb` implemented a single-layer LSTM in **PyTorch** with 50 hidden units. The model operates on a sliding window of 30 consecutive days to predict the next day's sales (one-step-ahead forecasting in training mode, recursive multi-step in inference mode).

[Insert Table 13: LSTM Model Architecture Summary]
Caption: PyTorch LSTM model architecture, training configuration, and hyperparameters used in Day6_LSTMForecasting.ipynb.

| Component | Configuration |
|-----------|---------------|
| Input size | 1 (univariate: scaled daily sales) |
| Hidden size | 50 units |
| Number of LSTM layers | 1 |
| Fully connected output | Linear(50 → 1) |
| Sequence length | 30 days |
| Train/test split | 80% / 20% (chronological) |
| Scaling | MinMaxScaler (fit on train only) |
| Optimiser | Adam (lr = 0.001) |
| Loss function | Mean Squared Error (MSE) |
| Epochs | 50 |
| Batch processing | Full batch per epoch |
| Saved model | `models/lstm_model.pth` |

**Training**

The training loop ran for 50 epochs with the Adam optimiser, progressively minimising MSE loss on the training sequences. Loss convergence was monitored and plotted to verify the absence of overfitting.

[Insert Figure 17: LSTM Training Loss Curve (50 Epochs)]
Caption: Mean squared error (MSE) loss per epoch during LSTM training, showing convergence to a stable minimum within approximately 30 epochs.

**30-Day Recursive Forecasting**

For generating future predictions beyond the training data, a **recursive forecasting** strategy was employed: the model predicts one day ahead, the prediction is appended to the input sequence (dropping the oldest value), and the process repeats for 30 iterations. This introduces cumulative error accumulation over the horizon, motivating the ensemble approach in Day 8.

[Insert Figure 18: LSTM Forecast — Actual vs Predicted (Test Period)]
Caption: Comparison of actual daily sales and LSTM predictions on the held-out 20% test period, showing the model's ability to track demand patterns but with reduced accuracy on sharp spikes.

**Performance**

The LSTM achieved a test MAPE of **[ADD ACTUAL VALUE]**, demonstrating complementary strengths and weaknesses relative to Prophet (strong on non-linear patterns, weaker on long-range extrapolation).

---

### 13.4 Hybrid Ensemble Model (Day 8)

**Rationale**

The ensemble approach exploits the complementary strengths of Prophet and LSTM: Prophet provides stable long-range trend and seasonality extrapolation with principled uncertainty intervals, while LSTM captures short-range non-linear temporal dependencies. By combining them through an optimal weighted average, the ensemble achieves lower MAPE than either individual model.

**Alignment**

Both models' predictions were aligned on a common date index by merging the Prophet forecast DataFrame (`forecast_results.csv`) and the LSTM predictions on the `ds` column. Only dates with available actual values were used for weight optimisation; future dates used both models' forward predictions.

**Weight Optimisation**

An exhaustive grid search was performed over Prophet weights w ∈ {0.00, 0.05, 0.10, ..., 1.00} (21 values), computing the ensemble MAPE at each weight combination:

```python
Ensemble(t) = w × Prophet_yhat(t) + (1 - w) × LSTM_yhat(t)
```

The weight minimising MAPE on the overlapping actual-vs-prediction window was selected as the final ensemble weight. This search was visualised as a MAPE vs Prophet weight curve.

[Insert Figure 19: MAPE vs Prophet Weight — Ensemble Blending Grid Search]
Caption: MAPE computed for 21 equally spaced Prophet weight values from 0.0 to 1.0. The minimum identifies the optimal blending coefficient, balancing Prophet's seasonal stability with LSTM's non-linear responsiveness.

**30-Day Future Forecast**

For future dates (where no actuals exist), Prophet's forward `yhat` values were taken directly from the 30-day extension in `forecast_results.csv`. For LSTM, recursive forecasting was applied starting from the last 30 days of training data. The optimal weights were then applied to produce the final 30-day ensemble demand forecast, saved as `data/ensemble_future_30_days.csv`.

[Insert Figure 20: Hybrid Ensemble Forecast vs Individual Models (60-Day Window)]
Caption: 60-day window showing actual daily sales (blue), Prophet (green dashed), LSTM (orange dashed), and the ensemble forecast (red bold), demonstrating the ensemble's superior tracking of actual demand variations.

[Insert Figure 21: 30-Day Future Ensemble Demand Forecast]
Caption: Forward-looking 30-day ensemble demand forecast, with all three model predictions (Prophet, LSTM, Ensemble) shown together in a shaded forecast window, used as input to the inventory optimisation module.

**Performance Summary**

[Insert Table 14: Forecasting Model Performance Comparison (MAPE and RMSE)]
Caption: MAPE and RMSE on the validation/test period for all three forecasting models, compared against the ≤12% project target.

| Model | MAPE | RMSE | Target Met |
|-------|------|------|-----------|
| Prophet | [ADD ACTUAL] | [ADD ACTUAL] | — |
| LSTM | [ADD ACTUAL] | [ADD ACTUAL] | — |
| **Ensemble** | **[ADD ACTUAL]** | **[ADD ACTUAL]** | **≤ 12% target** |

The ensemble model achieved a MAPE of **[ADD ACTUAL VALUE]** against the project target of ≤12%. Results, metrics, and the trained LSTM model were saved as `data/ensemble_forecast_results.csv`, `data/ensemble_metrics.csv`, and `models/lstm_model_ensemble.pth` respectively.

---

*End of Part 1*

---

> **Part 2** will cover: Churn Prediction (Feature Engineering, SMOTE, XGBoost, Optuna Tuning, SHAP), Inventory Optimisation, Drift Detection and MLOps Pipeline, Streamlit Dashboard, Results and Evaluation Metrics, Challenges Faced, Business Impact, Future Enhancements, Conclusion, References, and Appendices.
