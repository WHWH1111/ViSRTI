# Final masked-model analyses

This is the continuation of the GP-positive/GPT-negative masking experiment.
Do not restart `scripts/run_gp_gpt_masking.py`. The final-model controller waits
for that experiment to finish before it opens held-out responses.

## One selected configuration

`outputs/final_masked_models_seed1234/selection_manifest.json` records the
development-MAE stopping decision, exact masks, checkpoints and training settings.
All new Figures 3, 4, S5 and scaffold runs read this selection. GP/GPT were
requested candidates after earlier test results were inspected; GPT is not the
winner of the earlier full development-set comparison, which favored GPTC.

The positive candidate has four P descriptors and no T branch; the negative
candidate has four P plus four T descriptors and a 128-dimensional T output.
Masks retain input dimensions and suppress descriptor values in the model.

## Execution

```powershell
python scripts/data-val/evaluate_final_masked_models.py --preflight-development
python scripts/run_final_model_updates.py
```

Run only one controller; an operating-system file lock prevents duplicates.
Inspect the recorded process before using `--resume` after a failure. It does
not repeat completed masking training or scaffold folds. When missing calibration
inputs are completed, resume runs only that unfinished stage, its figure export
and delivery assembly. Partial calibration exports can be recomputed only for
the same input file and frozen selection; completed predictions are reused.

The stages are development-mask freeze, fixed-test predictions and exact
descriptor SHAP, fixed-test figures, calibration-input checks and (when complete)
Figure 6a-b, ten scaffold training runs, then delivery assembly. Missing calibration
inputs do not block scaffold training. The controller uses the active Python
environment; this project targets `pytorch38-gpu`.

## Figures and data

- Figure 2 and the accepted architecture comparison remain unchanged.
- Figure 3 shows full fixed-test metrics for each already-selected masking step,
  without bootstrap or error bars. Test values never change the retained mask.
  Its displayed test SHAP is descriptive; masking order came from development
  SHAP, and a test-derived ranking must not replace that order.
- Figure S5 uses the selected checkpoint's own training history and predictions.
- Figure 4 uses the same held-out compounds for class metrics and class SHAP.
  The ten frozen categories overlap. Normalize the class mean absolute SHAP
  within each row across retained descriptors; this does not establish causality.
- Figure 5 uses out-of-fold scaffold predictions. Report mean/SD across folds
  separately from metrics pooled across all out-of-fold predictions. This is
  fixed-configuration structural robustness, not nested feature-selection CV.
- Figure 6 requires author-confirmed structures, descriptors and mode-specific
  calibrant lists. Predict the 21 standards and two cases with the selected model;
  refit RT against newly predicted RTI. BaP/BeP never enter calibration fitting.
  `scripts/plt/plot_calibration_results.py` preserves the original axes (measured
  RT horizontally, predicted RTI vertically) using the parametric line
  `RT = intercept + slope * RTI`. It does not replace this with an inverse OLS fit.
  Any calibration intervals use actual residuals, not fabricated bootstrap data.

Each SHAP explanation keeps that sample's own molecular graph, applies stored
training standardization once and uses 50 training-background compounds. Background
selection is not bootstrap resampling of reported performance.

## Delivery boundary

`修改部分内容/07_最终模型后续分析/` holds the new figures, Chinese-named source
data, final models, code, changes and the outstanding calibration input sheet.
Older outputs remain historical evidence, not interchangeable final-model data.
Word documents are not edited in this workflow. Numerical checks and actual
preview inspection are required before marking figures reviewed; missing
calibration inputs must remain visible rather than being replaced by old predictions.
