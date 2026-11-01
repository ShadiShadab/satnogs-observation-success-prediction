# Robustness Check (added November 2026)

This check was added during a portfolio review. It does **not** replace or change the original capstone results. It tests how far those results can be relied on.

Script: [`../../04_robustness_check.py`](../../04_robustness_check.py) · Raw output: [`robustness_results.json`](robustness_results.json)
Model used for the checks: LightGBM (fast to retrain). The tuned Random Forest (267 MB) is not stored in the repository.

## Results

| Check | ROC-AUC | Accuracy |
|---|---|---|
| Saved LightGBM model, saved test split (reproduction) | 0.934 | 86.5% |
| Retrained LightGBM, random split, all features | 0.952 | 88.9% |
| Same, **without `archived`** | 0.932 | 86.3% |
| Same, without station-location features | 0.835 | 75.0% |
| **Time split**: train 2021–2024, test 2025, all features | **0.658** | 46.3% |
| Time split, without `archived` | **0.756** | 63.3% |

## What this means

1. **Reproducible.** The saved LightGBM model reproduces its reported test score (0.934 vs 0.935 reported).
2. **`archived` is not known at scheduling time.** SatNOGS sets it after the recordings move to the Internet Archive. On the random split, removing it costs about 0.02 ROC-AUC, so the headline performance does not depend on it.
3. **Station location carries much of the signal.** Station coordinates identify individual stations, so the model partly learns which stations are reliable. That helps when scheduling on known stations. It will not carry over to new stations.
4. **Forward-looking performance is lower.** The original evaluation used a random split, so observations from the same period and station appear in both training and test data. Trained on 2021–2024 and tested on 2025, ROC-AUC falls to 0.66–0.76. Removing `archived` *improves* performance under the time split, which suggests its relationship with success changes over time.

## Recommended next steps

- Retrain without post-observation fields (`archived`, current `station_status`) and use a time-based validation split.
- Add station-history features computed only from past observations (for example, a rolling success rate) instead of raw coordinates.
- Report both random-split and time-split metrics.
