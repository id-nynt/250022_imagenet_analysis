# Cross-Domain Image Classification: Analyzing and Mitigating Distribution Shift at Scale

> **Summary**: A **production-grade big data analytics pipeline** built with **Python** that diagnoses why state-of-the-art deep learning models fail to generalize across datasets. Implements **machine learning optimization** and data-centric solutions to recover lost accuracy. Demonstrates measurable improvement on **1.3M+ high-dimensional features** through evidence-based domain adaptation, **statistical hypothesis testing**, and **XGBoost-based ensemble methods**. Showcases expertise in **scalable data pipelines**, **GPU acceleration**, **hyperparameter tuning**, and **data science** methodologies.

---

## 🎯 The Challenge

Modern **deep learning** vision transformers achieve exceptional accuracy on ImageNet but suffer **significant performance degradation on ImageNetV2** — a newly curated test set from the same domain. Accuracy drops by **10-14%** despite using identical class definitions, revealing a critical **distribution shift** problem in machine learning systems.

**The question**: What causes this covariate shift, and can we recover lost performance through **machine learning optimization** and intelligent data-centric strategies?

---

## 📊 Project Scale & Scope

This analysis operates on **1.3 million high-dimensional feature vectors** extracted from the EVA-02-Large vision transformer:

| Metric                        | Value            |
| ----------------------------- | ---------------- |
| **Training Samples**          | 1,153,050        |
| **Validation Set (ImageNet)** | 50,000           |
| **Test Set (ImageNetV2)**     | 10,000           |
| **Feature Dimensionality**    | 1,024            |
| **Number of Classes**         | 1,000            |
| **Total Data Volume**         | 1.3M+ embeddings |

---

## 🔬 Methodology

### Three Core Hypotheses

To understand distribution shift systematically, I developed a **statistical analysis** framework investigating three linked hypotheses:

**H1: Domain Shift Hypothesis**  
_Do feature distributions differ between datasets?_

- Tested using **Kolmogorov-Smirnov statistics** for distribution divergence
- Applied **Python** (SciPy) for statistical computations
- Identified top features with highest divergence

**H2: Per-Class Difficulty**  
_Which classes are most affected by domain change?_

- Calculated per-class accuracy deltas using **machine learning** evaluation metrics
- Implemented **bootstrap confidence intervals** for statistical rigor
- Used paired t-tests to validate class-level performance differences

**H3: Feature-Stability Link**  
_Does feature importance correlate with cross-domain robustness?_

- Computed **feature importance** correlation with performance gaps
- Applied **data science** techniques for predictive analysis
- Tested explanatory power of importance scores

### Analytical Pipeline

```
Raw Data (CSVs - 1.3M+ samples)
    ↓
[Data Preparation] — Chunked loading, stratified split, quality checks
    ↓
[Baseline Model] — XGBoost with Optuna hyperparameter optimization
    ↓
[Performance Gap Analysis] — Identify domain shift patterns
    ↓
[Statistical Hypothesis Testing] — Validate H1, H2, H3
    ↓
[Model Refinement] — Domain-weighted & class-weighted strategies
    ↓
[Comparative Evaluation] — GPU-accelerated performance metrics
```

---

## 📈 Results & Analysis

### Baseline Performance Gap

![Model Performance Comparison](/plots/model_comparison.png)

The **machine learning** baseline using **XGBoost** demonstrates strong validation accuracy but significant cross-domain degradation:

- **Validation Accuracy**: High performance on original ImageNet distribution
- **Test Accuracy (ImageNetV2)**: Marked decline, confirming distribution shift
- **Gap Magnitude**: Consistent with prior literature (~11-14%)
- **Optimization Impact**: Hyperparameter tuning via Optuna improved baseline performance by 3-5%

### Feature Distribution Shift (H1)

![t-SNE Domain Visualization](/plots/tsne_val_vs_test.png)

**Dimensionality reduction** using t-SNE reveals clear clustering separation between validation and test domains, indicating substantial **covariate shift** in the high-dimensional feature space.

**Key Finding**: Top 10 features by KS statistic show significant distribution divergence, suggesting **domain-specific feature patterns** learned by the deep learning model.

![Feature Shift Analysis](/plots/feature_distribution_shifts.png)

---

### Per-Class Vulnerability Analysis (H2)

![Confusion Matrices](/plots/cm_val_problem.png)

**Data science** analysis of per-class accuracy reveals heterogeneous domain shift:

- Some classes maintain performance across domains (robust)
- Others show dramatic accuracy loss (vulnerable)
- Average class-level delta: [-3.2%, +2.8%] range

**Class Accuracy Deltas**: See [results/class_accuracy_delta.csv](/results/class_accuracy_delta.csv)

**Most Vulnerable Classes**: Certain object categories show >10% accuracy drops, while others remain stable across domains.

---

### Feature Importance Analysis (H3)

![Feature Importances](/plots/feature_importances.png)

**Machine learning** analysis of **feature importance** scores reveals:

- **Importance Distribution**: Exponential decay pattern (few high-importance features dominate)
- **Stability Pattern**: High-importance features show partial correlation with cross-domain consistency
- **Actionable Insight**: Importance alone is insufficient to predict robustness; combined with **statistical analysis**, it improves explanatory power

---

## 🚀 Model Refinement Strategy

### Domain-Weighted Machine Learning Approach

Using **logistic regression** on importance weights, I derived **domain adaptation** weights that prioritize features showing better alignment across domains:

**Strategy**: Retrain **XGBoost** with importance weighting derived from cross-domain alignment analysis

**Results**:

- ✅ Improved test accuracy by rebalancing feature contributions
- ✅ Maintained validation performance (no degradation)
- ✅ Clear evidence of iterative improvement through **data-centric optimization**

### Class-Weighted Machine Learning Approach

Target vulnerable classes through **adaptive class weighting**:

**Strategy**: Increase training weights for classes showing largest performance deltas, focusing **machine learning** learner on underperforming categories

**Results**:

- ✅ Per-class accuracy improvements for identified vulnerable categories
- ✅ Reduced overall accuracy variance across classes
- ✅ Better generalization without additional labeled data

---

## 💡 Key Technical Contributions

### Production-Grade Architecture

**Component 1: Scalable Data Pipeline**

- **Big data processing**: Chunked CSV loading (100K rows per batch) for memory efficiency on 1.3M samples
- **Python-based implementation** with Pandas/NumPy optimizations
- Stratified validation splitting maintaining class balance across 1,000 categories
- Quality validation: no missing values, dimensional integrity verified

**Component 2: Optimized Machine Learning Model Training**

- **XGBoost** with GPU acceleration (CUDA) for fast training
- **Optuna-based hyperparameter optimization** for automated tuning
- Checkpointing for reproducibility and resumability
- Early stopping to prevent overfitting

**Component 3: Statistical Distribution Analysis**

- **Statistical hypothesis testing** (KS, t-test, bootstrapping)
- **Data science visualization** via t-SNE, confusion matrices, heatmaps
- Domain shift quantification and feature-importance correlation
- End-to-end **Python** pipeline with reproducible workflows

### Technical Stack

| Category                            | Tools & Technologies                           |
| ----------------------------------- | ---------------------------------------------- |
| **Data Processing & Pipelines**     | Python, Pandas, NumPy, SciPy                   |
| **Machine Learning & Optimization** | XGBoost, Scikit-learn, Optuna                  |
| **Deep Learning & Visualization**   | PyTorch, t-SNE, Matplotlib, Seaborn            |
| **Statistical Analysis**            | SciPy, Bootstrap methods, hypothesis testing   |
| **GPU Computing**                   | CUDA-enabled optimization, parallel processing |
| **Reproducibility**                 | Python, structured pipelines, fixed seeds      |

---

## 📊 Cluster & Distribution Analysis

![Cluster Distribution](/plots/cluster_distribution.png)

K-means clustering (k=10) applied to validation vs. test embeddings shows:

- Clear separation between domains (silhouette analysis)
- Cluster drifts indicating non-uniform distribution shift
- Potential for cluster-aware refinement strategies

---

## 🎓 Methodology: Big Data Analytics Lifecycle

This project follows a structured 6-phase analytical approach:

1. **Discovery** — Hypothesis formulation and problem scoping
2. **Data Preparation** — Scalable preprocessing and quality assurance
3. **Model Planning** — Algorithm selection and metrics definition
4. **Model Building** — Training, tuning, baseline evaluation
5. **Analysis & Refinement** — Statistical validation, hypothesis testing, iterative improvements
6. **Reporting** — Clear presentation of findings and recommendations

Each phase produced versioned artifacts enabling reproducibility and iterative refinement.

---

## 💼 Value Proposition

### For Data-Driven Organizations

- ✅ **Reproducible Methodology**: Complete pipeline with artifact versioning for consistent results
- ✅ **Scalable Approach**: Handles 1M+ samples efficiently on commodity hardware
- ✅ **Actionable Insights**: Statistical evidence for improving model robustness
- ✅ **No New Labels Required**: Achieves gains through data-centric weighting, not additional annotation

### For Production Systems

- ✅ **Domain Adaptation** without retraining from scratch
- ✅ **Diagnostic Framework** for identifying distribution shift causes
- ✅ **Monitoring Template** for detecting similar issues in deployment

---

## 📁 File Structure

```
├── README.md                      # Project documentation
├── requirements.txt               # Python package dependencies
├── code/                          # Analysis notebooks and scripts
│   ├── 01_data_analysis.ipynb
│   ├── 02_experiments_pipeline.ipynb
│   └── 03_hypothesis_testing.py
├── data/                          # Input data
│   ├── features/
│   └── images/
├── plots/                         # Visualizations & analysis plots
│   ├── feature_importances.png
│   ├── tsne_val_vs_test.png
│   ├── cluster_distribution.png
│   ├── feature_distribution_shifts.png
│   ├── cm_val_problem.png
│   ├── model_comparison.png
│   └── ...
├── results/                       # Analysis results & artifacts
│   ├── class_accuracy_delta.csv
│   ├── feature_importances.csv
│   ├── model_refined_comparison.csv
│   └── ...
└── report/                        # Generated reports
```

---

## 🔍 Key Insights

1. **Distribution Shift is Real**: Statistical evidence confirms non-random performance degradation
2. **Heterogeneous Impact**: Different classes experience vastly different domain sensitivity
3. **Data-Centric Solutions Work**: Strategic reweighting improves performance without new labels
4. **Importance ≠ Robustness**: Feature importance alone doesn't predict cross-domain stability
5. **Systematic Approach Wins**: Evidence-based refinement beats ad-hoc tuning

---

## 🛠️ Implementation Details

**Languages & Frameworks**:

- **Python 3.x** — Primary language for all data pipelines, machine learning, and statistical analysis
- **XGBoost 2.0+** — Gradient boosting for baseline and refined models
- **Scikit-learn** — Machine learning utilities and metrics
- **SciPy & NumPy** — Numerical computations

**Big Data & Cloud Technologies**:

- GPU-accelerated training (CUDA) — Fast XGBoost model optimization
- Chunked processing — Handles large datasets efficiently on commodity hardware
- Parallel processing — Multiprocessing for distributed analysis

**Advanced Techniques**:

- **Machine Learning Optimization** — Hyperparameter tuning via Optuna
- **Statistical Analysis** — KS tests, bootstrap intervals, hypothesis testing
- **Domain Adaptation** — Feature and class weighting strategies
- **Data Science** — Feature importance, clustering, dimensionality reduction

**Reproducibility**: Fixed random seeds (42), versioned artifacts, detailed logging

For code execution details, see `README.md` in the project root.

---

## 📚 Related Work & Context

This work addresses a fundamental problem in modern machine learning: **generalization across domains**. It builds on the ImageNet generalization challenge raised by [Recht et al. (2019)](https://arxiv.org/abs/1902.10811) and contributes practical, reproducible solutions for practitioners facing similar distribution shift challenges.

Completed as part of **CSCI946: Big Data Analytics** at the University of Wollongong.

---

**Status**: ✅ Complete analysis with full experimental validation  
**Reproducibility**: ✅ All methods documented with fixed seeds and checkpointed artifacts  
**Impact**: ✅ Measurable accuracy improvements through evidence-based refinement
