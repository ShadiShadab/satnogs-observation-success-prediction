# SatNOGS Observation Success Prediction

Predicting which satellite observations will succeed on the SatNOGS open ground-station network, using machine learning on real network data.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-2ea44f)
![License](https://img.shields.io/badge/license-MIT-blue)

Capstone project, **MIT Emerging Talent Program, Computer & Data Science** (December 2025). Individual project by [Shadi Shadabshoar](https://www.linkedin.com/in/shadi-shadabshoar/).

---

## At a glance

| | |
|---|---|
| **Problem** | About half of all SatNOGS observations fail (49.5% success in the study data). Each failed observation wastes ground-station time that volunteers contribute. |
| **Question** | Can observation and station characteristics predict whether an observation will succeed? |
| **Data** | SatNOGS Network database: 12.5M+ raw observations, filtered to **578,010 observations (2021–2025)** with 24 features |
| **Approach** | Data cleaning → EDA → feature engineering → baseline and advanced models → interpretation → Streamlit dashboard |
| **Best model** | Tuned Random Forest |
| **Test results** | **ROC-AUC 0.946 · accuracy 88.3%** · precision 86.4% · recall 90.5% · F1 0.884 (86,702 held-out observations, random split) |
| **Robustness check** | Trained on 2021–2024 and tested on 2025, ROC-AUC is **0.66–0.76**. See [Robustness check](#robustness-check). |

---

## Why it matters

SatNOGS is a global, volunteer-run network of satellite ground stations operated by the [Libre Space Foundation](https://libre.space/). Observers schedule passes on stations around the world, and stations record and upload the data. A model that flags likely failures before scheduling could help observers and station operators make better use of limited station time.

---

## What I did

All work in this repository is my own:

- **Domain study:** SatNOGS architecture, observation lifecycle, ground-station parameters and satellite pass geometry
- **Data engineering:** loaded the SatNOGS database dump (MariaDB) and profiled the schema. I filtered 12.5M+ records to a clean four-year dataset and removed invalid durations and missing values.
- **Leakage control:** found and removed fields that reveal the outcome (`status`, `waterfall_status`, `vetted_status`, `archive_url` and others; see `metadata.json`)
- **Feature engineering:** 24 temporal, geographic and geometric features (for example, cyclic hour and month encodings, altitude transforms, absolute latitude)
- **Modelling:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM and an MLP, with hyperparameter tuning and validation/test separation
- **Interpretation:** permutation importance, partial dependence and ICE plots, error-pattern analysis, and four operator-facing business rules
- **Communication:** Streamlit dashboard, final report, presentation and a blog-style write-up

---

## Results

![Final results summary](4_data_analysis/ml_results/advanced/final_summary_visualization.png)

| Model | ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| **Random Forest (tuned)** | **0.947** | **88.4%** | **0.885** |
| LightGBM | 0.935 | 86.7% | 0.870 |
| XGBoost (optimised) | 0.934 | 86.5% | 0.868 |

Validation-set comparison from [`best_model_info.json`](4_data_analysis/ml_results/advanced/best_model_info.json). The final test-set scores for the tuned Random Forest are in the table above. They come from [`final_test_metrics.json`](4_data_analysis/ml_results/advanced/final_test_metrics.json).

**How to read the "49.5% → 88.3%" chart (bottom left):** it compares the model's test accuracy with the dataset's base success rate. It is a **simulated scheduling scenario on historical data**, not a measured improvement in live network operations.

### Key findings

- Station location and altitude are among the strongest predictors. Stations between about 40° and 53° latitude performed best in this dataset.
- Observation year has a strong effect, which reflects changes in the network over time.
- Satellite pass geometry (maximum elevation, duration) matters less than station characteristics.

---

## Robustness check

Added in November 2026 during a portfolio review. The original results are unchanged. Details are in [`4_data_analysis/ml_results/robustness/`](4_data_analysis/ml_results/robustness/README.md).

| Check | ROC-AUC |
|---|---|
| Saved LightGBM model reproduces its reported test score | 0.934 |
| Random split, without the `archived` field | 0.932 |
| Random split, without station-location features | 0.835 |
| **Time split** (train 2021–2024 → test 2025), without `archived` | **0.756** |

What I learned from this:

1. **`archived` is a post-observation field.** SatNOGS archives recordings to the [Internet Archive](https://archive.org/details/satnogs) after upload, so the field is not known at scheduling time. Removing it costs only about 0.02 ROC-AUC on the random split, so the headline result does not depend on it.
2. **A random split overestimates forward-looking performance.** When the model has to predict a future year it has not seen, performance drops considerably. A production version should be validated on a time-based split.
3. **Station coordinates act as station identity.** The model partly learns which stations are reliable. That is useful for known stations, but it will not carry over to new ones.

---

## Interactive dashboard

The Streamlit dashboard presents model performance, feature analysis, interpretation, business rules and a rule-based success estimator.

Note: the estimator applies rules derived from the model's findings. It does **not** run the trained model. The tuned Random Forest (267 MB) is not stored in the repository.

[Dashboard screen recording](5_communication_strategy/dashboard/dashboard_prototype/14.12.2025_22.48.57_REC.mp4)

---

## Reproduce

Requirements: Python 3.10+ and Git. A SatNOGS database dump is only needed to rebuild the dataset from scratch.

```bash
git clone https://github.com/ShadiShadab/satnogs-observation-success-prediction.git
cd satnogs-observation-success-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Run the dashboard** (uses the saved results in `4_data_analysis/ml_results/`):

```bash
streamlit run 5_communication_strategy/dashboard/dashboard_prototype/app.py
```

**Re-run the modelling notebooks or the robustness check:** extract `4_data_analysis/ml_results/advanced/fixed_data.rar` (train/validation/test splits, about 145 MB unpacked) into the same folder with any RAR tool (7-Zip, WinRAR, `unar`). The result should be `4_data_analysis/ml_results/advanced/fixed_data/`. Then:

```bash
python 4_data_analysis/04_robustness_check.py
```

**Run the tests** (results integrity, dashboard smoke test, credential check):

```bash
pytest
```

**Rebuild the dataset from the database (optional):** notebooks in `1_datasets/` and `2_data_preparation/` connect to a local MariaDB/MySQL copy of the SatNOGS database. Replace the `User` / `Password` placeholders with your own local credentials.

---

## Repository structure

```text
├── 0_domain_study/            SatNOGS background research
├── 1_datasets/                Schema exploration, data dictionary, raw samples
├── 2_data_preparation/        Dataset creation and cleaning, feature list
├── 3_data_exploration/        EDA, geographic analysis, feature engineering (+ plots)
├── 4_data_analysis/           Baseline, advanced and interpretation notebooks
│   ├── 04_robustness_check.py Robustness check (added Nov 2026)
│   └── ml_results/            Metrics, plots, saved models, data splits (.rar)
├── 5_communication_strategy/  Streamlit dashboard, blog post, stakeholder material
├── 6_final_presentation/      Final report and slides
├── collaboration/             Learning goals, constraints, retrospectives
├── tests/                     Integrity and smoke tests
└── requirements.txt
```

---

## Limitations and future work

- **Evaluation:** random train/test split. Add time-based validation and report both.
- **Feature timing:** remove post-observation or current-state fields (`archived`, current `station_status`) for a true pre-scheduling model.
- **Generalisation:** replace raw station coordinates with history-based station reliability features.
- **Deployment:** serve the actual model in the dashboard instead of the rule-based estimator, and connect it to the live SatNOGS Network API.
- **Data:** the processed CSV built from the database is not included because of its size. The saved splits in `fixed_data.rar` let you reproduce the modelling steps.

---

## License

Code: MIT ([LICENSE](LICENSE)). SatNOGS data: [CC BY-SA 4.0](https://network.satnogs.org/), as published by the SatNOGS Network.

Developed as part of the MIT Emerging Talent Program in Computer and Data Science, 2025.
