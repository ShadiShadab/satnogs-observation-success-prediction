# SatNOGS Success Predictor Dashboard

A Streamlit dashboard that presents the results of the SatNOGS observation-success capstone project to ground-station operators and observers.

## Sections

| Section | Content |
|---|---|
| Performance Overview | Test-set metrics for the tuned Random Forest, plus a simulated scheduling scenario |
| Feature Analysis | Feature importance and permutation importance |
| Model Insights | Interpretation plots: model comparison, error patterns, prediction confidence |
| Business Rules | Four operator-facing rules derived from the model's findings |
| Interactive Demo | A **rule-based** estimator built from those rules |

### Data sources

The dashboard reads saved results from `4_data_analysis/ml_results/` (metrics JSON, feature-importance CSVs and plots). Paths are resolved relative to `app.py`, so the dashboard works from any working directory. If a file is missing, it falls back to the values recorded in the final report.

### Important limitation

The Interactive Demo does **not** run the trained model. The tuned Random Forest is about 267 MB and is not stored in the repository. The demo adds rule-based adjustments to the 49.5% base rate, so its percentages are illustrative, not calibrated probabilities. Serving the real model is listed as future work in the main README.

## Run

From the repository root:

```bash
pip install -r 5_communication_strategy/dashboard/dashboard_prototype/requirements.txt
streamlit run 5_communication_strategy/dashboard/dashboard_prototype/app.py
```

Tested with streamlit 1.66, pandas 3.1, numpy 2.5 and plotly 7.1. A smoke test that opens every section runs in `tests/test_project_integrity.py`.

A screen recording of the dashboard is in this folder: `14.12.2025_22.48.57_REC.mp4`.
