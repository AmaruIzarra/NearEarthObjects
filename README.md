# NearEarthObjects

A machine learning project classifying Near-Earth Asteroids as potentially hazardous, using 338,199 observations from NASA close-approach records spanning 1910 to 2024.

**Dataset:** [NASA Nearest Earth Objects (1910-2024)](https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024) (Kaggle)  
**Course:** Machine Learning, WATSpeed — University of Waterloo

---

## The Problem

NASA flags asteroids as Potentially Hazardous based on two thresholds: a Minimum Orbit Intersection Distance under 0.05 AU and an absolute magnitude brighter than 22. Neither value is directly in the dataset. Rather than treating this as a dead end, the project frames the model as a triage tool — one that learns which observable properties most strongly correlate with a hazardous designation, not one that claims to reproduce NASA's exact formula.

The dataset is also heavily imbalanced: 87.2% non-hazardous vs 12.8% hazardous. A model that always predicts "non-hazardous" gets 87.4% accuracy, which is useless in a safety context. Every modelling decision in this project accounts for that.

---

## Features

The raw dataset had 9 columns. After cleaning, `estimated_diameter_min` and `estimated_diameter_max` were consolidated into a single mean — they had a perfect 1.00 correlation, so keeping both was redundant. Three physics-motivated features were then engineered:

| Feature | Description |
|---|---|
| `absolute_magnitude` | Intrinsic brightness; lower means brighter/larger |
| `estimated_diameter_mean` | Average of min and max diameter estimates (km) |
| `relative_velocity` | Velocity relative to Earth (km/h) |
| `miss_distance` | Closest approach distance (km) |
| `kinetic_energy` | Engineered: diameter squared times velocity squared |
| `size_to_distance_ratio` | Engineered: diameter divided by miss distance |
| `magnitude_velocity` | Engineered: absolute magnitude times relative velocity |

---

## Models and Results

| Model | Accuracy | Precision (Haz.) | Recall (Haz.) | F1 (Haz.) | Macro F1 |
|---|---|---|---|---|---|
| Dummy Baseline | 87.4% | -- | 0.00 | 0.00 | 0.47 |
| Logistic Regression | 87.4% | 0.49 | 0.00 | 0.01 | 0.47 |
| AdaBoost | 87.4% | 0.00 | 0.00 | 0.00 | 0.47 |
| XGBoost | 89.3% | 0.70 | 0.27 | 0.39 | 0.66 |
| Random Forest (balanced) | 91.1% | 0.71 | 0.50 | 0.59 | 0.77 |
| Linear SVM (balanced) | 72.2% | 0.30 | 0.93 | 0.46 | 0.64 |
| RBF SVM (20k subsample) | 70.0% | 0.30 | 0.99 | 0.46 | 0.63 |
| LightGBM (balanced) | 74.3% | 0.33 | 0.98 | 0.49 | 0.66 |

ROC-AUC: Random Forest 0.942, XGBoost 0.915, LightGBM 0.906, RBF SVM 0.863

Models trained without class balancing (Logistic Regression, AdaBoost, XGBoost) completely failed to learn the minority class despite looking accurate, which is exactly why accuracy is a misleading metric when classes are imbalanced.

Random Forest had the best overall balance, with the highest macro F1 and AUC-ROC. LightGBM had the highest hazardous-class recall (0.98), making it the best fit for a triage scenario where missing a real threat is the worst possible outcome. The SVM variants maximized sensitivity at the cost of precision, which could still be useful when there is capacity to filter false alarms downstream.

---

## Key Findings

Absolute magnitude was the strongest individual predictor in both Random Forest and LightGBM (correlation of -0.34 with the target), consistent with its role in NASA's actual PHA definition.

Miss distance was consistently the weakest predictor across every stage of the analysis, correlation, boxplots, PCA loadings, and feature importance. How close an asteroid passes on a single observation tells you very little about whether it is classified as hazardous over its full orbital lifetime.

The engineered kinetic energy feature ranked among the top predictors in both models, which validated the approach of incorporating physics intuition into feature design.

PCA on the full feature set captured 78.6% of variance in two components (PC1: 54.4%, PC2: 24.2%), with PC1 representing size and PC2 capturing orbital dynamics. The 2D projection showed meaningful but partial class separation, confirming that the models have something real to learn from, but also that perfect classification is not achievable without MOID.

---

## Risk Score

On top of LightGBM's output probabilities, a composite 0 to 100 risk score was built by combining hazard probability (60%) with normalized asteroid diameter (40%). The score produces strong class separation overall, but the top-ranked asteroids revealed a limitation: very large non-hazardous asteroids can receive inflated scores because their diameter overrides the probability signal. Future work should explore nonlinear severity weighting to fix this.

---

## Setup

```bash
git clone https://github.com/AmaruIzarra/NearEarthObjects.git
cd NearEarthObjects
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm kagglehub
jupyter notebook neo_classifier.ipynb
```

The dataset downloads automatically via kagglehub on first run.

---

## Contributors

| Name | Contribution |
|---|---|
| Amaru Izarra-Jacome | Data cleaning, EDA, PCA, feature engineering, model comparison, visualizations |
| Nikhilkumar Patel | Feature engineering, Logistic Regression |
| Silvio Silva Junior | Dummy baseline classifier |
| Hugo Velasco | Random Forest, XGBoost, AdaBoost, SVM |
| Aten Cheung | LightGBM |
| Robert Allan Bolista | Risk score |
