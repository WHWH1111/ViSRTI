# Final-model evaluation

The released selection is [`results/final_masked_models_seed1234/selection_manifest.json`](../results/final_masked_models_seed1234/selection_manifest.json). It identifies the positive GP model with Koc and LogP masked and the negative GPT model without masking. The matching weights and training parameters are in `checkpoints/`. Figure 3, Figure 4, Figure S5 and the five-fold scaffold evaluation use these configurations.

## Reproduce the analysis

Create the environment in `README.md`, then run the feature-fusion and development-set masking steps described in [Development-set SHAP and feature masking](gp_gpt_masking_analysis.md). From the repository root, run:

```powershell
python scripts/run_final_model_updates.py
```

The controller reads the completed masking run from `outputs/gp_gpt_masking_dev_seed1234/`. It freezes that run's chosen masks in `outputs/final_masked_models_seed1234/selection_manifest.json`, computes fixed-test predictions and SHAP, exports figures, and trains five scaffold folds per mode. New calculations go to `outputs/final_masked_models_seed1234/` and `outputs/final_masked_scaffold_seed1234/`; figures and Chinese-named source tables go to `outputs/final_model_delivery_seed1234/`. The committed `results/` files record the reported run and remain separate from new output.

The controller permits one active process. If an interrupted run has no active process, `python scripts/run_final_model_updates.py --resume` continues from its recorded completed stages. The run state and logs are under `outputs/final_masked_models_seed1234/`.

## Reported results

| Output | Calculation |
| --- | --- |
| Figure 3 | Four metrics from each full fixed-test set after each evaluated masking step; no bootstrap or error bars. Development MAE, not these test metrics, determines the retained mask. |
| Figure 4 | Ten predefined, overlapping chemical categories on the same fixed-test predictions; class metrics and row-wise 0–1 normalized mean absolute SHAP. |
| Figure S5 | Training history and prediction scatter from the selected checkpoint for each mode. |
| Figure 5 | Five scaffold folds per mode using the frozen configuration; fold mean/SD and metrics pooled over valid out-of-fold predictions are separate statistics. |

The five-fold runs test structural robustness of the selected configuration; they do not repeat feature selection within each fold. Two negative-mode compounds failed graph processing, leaving 5,931 valid out-of-fold predictions from 5,933 source rows. Category SHAP shows relative model reliance within each category, not a causal effect.

## Experimental RT–RTI calibration

The repository does not include the confirmed experimental inputs needed to regenerate Figure 6. `scripts/data-val/predict_calibration_samples.py` can predict RTI for the 21 standards and two BaP/BeP case compounds once their structures, descriptors, measured RT and mode-specific calibration membership are supplied. Calibration fits measured RT against predicted RTI using the 21 standards; the two case compounds stay out of that fit. `scripts/plt/plot_calibration_results.py` displays measured RT on the horizontal axis and RTI on the vertical axis. Any confidence or prediction intervals use the fitted residuals.

GP and GPT were selected as masking candidates after exploratory test results had been inspected. Development MAE governs the subsequent masking steps, while fixed-test results describe the resulting models. See the [masking method](gp_gpt_masking_analysis.md) for the selection history.
