"""Checks for final-mask plotting data and development-only selection labels."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "plt" / "plot_final_masked_models.py"
    spec = importlib.util.spec_from_file_location("final_masked_figures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot = load_module()


def prediction_rows(values):
    return [{"smiles": str(i), "source_rti": float(i + 1), "predicted_rti": float(value),
             "residual_source_minus_prediction": float(i + 1 - value)} for i, value in enumerate(values)]


def test_masked_columns_are_not_present_in_heatmap():
    payload = {"partition": "test", "smiles": ["a", "b"], "feature_names": list(plot.FEATURES),
               "feature_values": np.zeros((2, 8)).tolist(), "shap_values": np.ones((2, 8)).tolist()}
    model = {"feature_code": "GPT", "physchem_mask": [1, 0, 1, 0], "toxicity_mask": [1, 0, 1, 0]}
    result = plot.retained_shap(payload, model)
    assert result["feature_names"] == ["Monoiso_Mass", "LogP", "IC50", "72h-EC50"]
    assert np.asarray(result["shap_values"]).shape == (2, 4)


def test_gp_does_not_export_toxicity_columns():
    payload = {"partition": "test", "smiles": ["a"], "feature_names": list(plot.FEATURES[:4]),
               "feature_values": [[0, 0, 0, 0]], "shap_values": [[1, 2, 3, 4]]}
    model = {"feature_code": "GP", "physchem_mask": [1, 1, 1, 0], "toxicity_mask": [1, 1, 1, 1]}
    result = plot.retained_shap(payload, model)
    assert result["feature_names"] == ["Monoiso_Mass", "LogKow", "LogP"]


def test_gp_accepts_absent_toxicity_processor():
    payload = {"partition": "test", "smiles": ["a"], "feature_names": list(plot.FEATURES[:4]),
               "feature_values": [[0, 0, 0, 0]], "shap_values": [[1, 2, 3, 4]]}
    model = {"feature_code": "GP", "physchem_mask": [1, 0, 1, 0], "toxicity_mask": None}
    result = plot.retained_shap(payload, model)
    assert result["feature_names"] == ["Monoiso_Mass", "LogP"]


def test_raw_feature_columns_stay_aligned_and_shap_exports_rti_units():
    payload = {"partition": "test", "smiles": ["a"], "feature_names": list(plot.FEATURES[:4]),
               "feature_values": [[0.1, 0.2, 0.3, 0.4]], "shap_values": [[-1, 2, 3, 4]],
               "raw_feature_values": [[101, 202, 303, 404]], "property_std": 20,
               "shap_unit": "standardized_target"}
    model = {"feature_code": "GP", "physchem_mask": [1, 0, 1, 0], "toxicity_mask": None}
    result = plot.retained_shap(payload, model)
    assert result["raw_feature_values"] == [[101, 303]]
    rows = list(plot.shap_long_rows(result, "pos", "selected_mask"))
    assert [row["raw_feature_value"] for row in rows] == [101, 303]
    assert [row["standardized_feature_value"] for row in rows] == [0.1, 0.3]
    assert [row["shap_rti"] for row in rows] == [-20, 60]


def test_raw_feature_shape_mismatch_fails():
    payload = {"partition": "test", "smiles": ["a"], "feature_names": list(plot.FEATURES[:4]),
               "feature_values": [[0, 0, 0, 0]], "shap_values": [[1, 2, 3, 4]],
               "raw_feature_values": [[10, 20, 30]]}
    model = {"feature_code": "GP", "physchem_mask": [1, 0, 1, 0], "toxicity_mask": None}
    with pytest.raises(ValueError, match="Raw feature evidence"):
        plot.retained_shap(payload, model)


def test_development_shap_is_rejected_by_test_plot():
    with pytest.raises(ValueError, match="test SHAP"):
        plot.retained_shap({"partition": "development"}, {})


def test_categories_use_same_samples_and_normalize_within_row():
    predictions = prediction_rows([1.1, 2.2, 2.9])
    tags = {row["smiles"]: [category for category, _ in plot.CATEGORIES] for row in predictions}
    shap = {"smiles": ["2", "0", "1"], "feature_names": ["LogKow", "LogP", "Koc"],
            "shap_values": [[-4, 8, 12], [1, -2, 3], [2, 4, -6]]}
    metrics, means, members = plot.category_evidence(predictions, shap, tags)
    assert len(metrics) == 10
    assert all(row["n"] == 3 for row in metrics)
    assert len(members) == 30  # nonexclusive categories retain every membership
    assert [row["normalized_mean_absolute_shap"] for row in means[:3]] == pytest.approx([0, 0.5, 1])
    assert np.allclose([row["mean_absolute_shap"] for row in means[:3]], [7 / 3, 14 / 3, 7])


def test_constant_category_profile_is_explicit_zero():
    predictions = prediction_rows([1.1, 2.2])
    tags = {row["smiles"]: [category for category, _ in plot.CATEGORIES] for row in predictions}
    shap = {"smiles": ["0", "1"], "feature_names": ["LogP"], "shap_values": [[2], [-4]]}
    _, means, _ = plot.category_evidence(predictions, shap, tags)
    assert all(row["constant_row"] for row in means)
    assert all(row["normalized_mean_absolute_shap"] == 0 for row in means)


def test_category_shap_cannot_use_different_cohort():
    with pytest.raises(ValueError, match="same held-out"):
        plot.category_evidence(prediction_rows([1.1, 2.2]),
                               {"smiles": ["0", "third"]}, {})


def make_steps(tmp_path):
    steps = []
    for step, (predictions, dev_mae) in enumerate((([1.1, 2.1, 3.1], 2), ([1.3, 2.3, 3.3], 1), ([1.01, 2.01, 3.01], 3))):
        rows = prediction_rows(predictions)
        filename = tmp_path / ("step_{}.csv".format(step))
        plot.write_rows(filename, rows)
        steps.append({"step": step, "accepted": step < 2, "selected": step == 1,
                      "development_metrics": {"mae": dev_mae}, "metrics": plot.metric_values(rows),
                      "prediction_file": str(filename), "removed_features": ["Koc"] * step})
    return steps


def test_selection_remains_development_based_when_test_ranking_reverses(tmp_path):
    steps = make_steps(tmp_path)
    selected, _ = plot.validate_steps(steps, tmp_path)
    assert selected == 1
    assert steps[2]["metrics"]["mae"] < steps[1]["metrics"]["mae"]


def test_metric_mismatch_fails(tmp_path):
    steps = make_steps(tmp_path)
    steps[1]["metrics"]["mae"] += 0.1
    with pytest.raises(ValueError, match="metric does not match"):
        plot.validate_steps(steps, tmp_path)


def test_wrong_development_acceptance_fails(tmp_path):
    steps = make_steps(tmp_path)
    steps[2]["development_metrics"]["mae"] = 0.5
    with pytest.raises(ValueError, match="stopping rule"):
        plot.validate_steps(steps, tmp_path)


def test_mask_bars_have_no_error_bars():
    steps = [{"step": 0, "accepted": True, "removed_features": [],
              "metrics": {"mae": 40, "rmse": 55, "r2": 0.9, "pcc": 0.95}}]
    fig, ax = plot.plt.subplots()
    try:
        plot.draw_mask_bars(ax, steps, "ViSRTI-GP", 0, "(b)", "pos")
        assert all(container.errorbar is None for axis in fig.axes for container in axis.containers)
        assert "test set" in ax.get_xlabel()
    finally:
        plot.plt.close(fig)


def test_shap_rank_comparison_preserves_development_removal_order():
    development = {"partition": "development", "feature_names": ["LogP", "Koc"],
                   "shap_values": [[4, 1], [-2, -1]], "removal_order": [1, 0]}
    test = {"partition": "test", "feature_names": ["Koc", "LogP"],
            "shap_values": [[6, 1], [-8, -1]]}
    rows = plot.shap_rank_comparison(development, test, "pos")
    assert rows[0]["feature"] == "LogP"
    assert rows[0]["development_mean_absolute_shap"] == 3
    assert rows[0]["test_mean_absolute_shap"] == 1
    assert rows[0]["development_rank_high_to_low"] == 1
    assert rows[0]["test_rank_high_to_low"] == 2
    assert rows[0]["frozen_development_removal_order_low_to_high"] == 2
    assert rows[1]["frozen_development_removal_order_low_to_high"] == 1


def test_shap_rank_comparison_requires_same_features():
    development = {"partition": "development", "feature_names": ["LogP"]}
    test = {"partition": "test", "feature_names": ["Koc"]}
    with pytest.raises(ValueError, match="different feature"):
        plot.shap_rank_comparison(development, test, "pos")
