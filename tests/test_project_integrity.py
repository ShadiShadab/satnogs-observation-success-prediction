"""Integrity tests: reported results, dashboard inputs and repository hygiene.

These tests check that the numbers quoted in the README match the saved result
files, and that the dashboard can find and run on the files it needs.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "4_data_analysis" / "ml_results"
APP = ROOT / "5_communication_strategy" / "dashboard" / "dashboard_prototype" / "app.py"


@pytest.fixture(scope="module")
def metrics():
    return json.loads((RESULTS / "advanced" / "final_test_metrics.json").read_text())


def test_confusion_matrix_matches_reported_metrics(metrics):
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    total = tn + fp + fn + tp
    assert total == 86702  # test split size in metadata.json
    assert (tp + tn) / total == pytest.approx(metrics["test_accuracy"], abs=1e-6)
    assert tp / (tp + fp) == pytest.approx(metrics["test_precision"], abs=1e-6)
    assert tp / (tp + fn) == pytest.approx(metrics["test_recall"], abs=1e-6)


def test_readme_headline_metrics_match_saved_results(metrics):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{metrics['test_roc_auc']:.3f}" in readme  # 0.946
    assert f"{metrics['test_accuracy'] * 100:.1f}%" in readme  # 88.3%


@pytest.mark.parametrize(
    "relative_path",
    [
        "advanced/final_test_metrics.json",
        "advanced/feature_importance_final.csv",
        "interpretation/permutation_importance.csv",
        "advanced/final_summary_visualization.png",
        "interpretation/permutation_importance_plot.png",
        "interpretation/error_patterns_visualization.png",
        "interpretation/prediction_confidence_histogram.png",
    ],
)
def test_dashboard_input_files_exist(relative_path):
    assert (RESULTS / relative_path).is_file()


def test_dashboard_runs_every_section():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    for section in app.radio[0].options:
        app.radio[0].set_value(section).run()
        assert not app.exception, section


def test_no_hardcoded_credentials_in_tracked_files():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    pattern = re.compile(r"DB_PASSWORD\s*=\s*\\?[\"'](?!Password\\?[\"'])[^\"'\\]+")
    for name in tracked:
        if name.endswith((".py", ".ipynb", ".md", ".txt", ".json")):
            text = (ROOT / name).read_text(encoding="utf-8", errors="ignore")
            assert not pattern.search(text), name
