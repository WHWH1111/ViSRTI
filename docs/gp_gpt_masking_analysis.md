# Development-set SHAP and feature masking

This analysis starts with the GP positive-mode and GPT negative-mode checkpoints from the fixed random-split feature-fusion run. The published selection is recorded in [`selection_manifest.json`](../results/final_masked_models_seed1234/selection_manifest.json): positive GP excludes Koc and LogP; negative GPT retains all four physicochemical and four toxicity descriptors. The negative-mode toxicity encoder output has 128 dimensions. G denotes the molecular graph, not a set of four named descriptors.

## Selection rule

The GP/GPT pair was specified after exploratory test results had been inspected. The masking procedure selects features *within* GP and GPT; it does not establish GPT as the winner of an untouched comparison across all fusion levels.

The development partitions contain 561 positive-mode and 534 negative-mode compounds. For each compound, SHAP holds its molecular graph fixed and attributes the auxiliary descriptors. It uses 50 training compounds as background (seed 1234) and applies the checkpoint's feature standardization once. Descriptors are ranked by unmasked development-set mean absolute SHAP. The script removes them cumulatively in that order, retrains each candidate, and stops when development MAE first increases. R², RMSE, and PCC are reported but do not change the stopping rule. SHAP describes model attribution, not causal effects.

Each retraining step uses `train_visnet_v2.py --development-only`, the recorded train/development assignments, L1 loss, seed 1234, up to 150 epochs, and development-MAE early stopping with patience 20. Test responses are evaluated only after the mask is fixed. The negative-mode pre-filter training pool contains 4,805 descriptor rows; 4,804 molecules processed successfully for fitting. The effective partition file keeps that exclusion consistent across steps.

## Run and outputs

From an activated `pytorch38-gpu` environment at the repository root:

```powershell
pwsh -File scripts/run_feature_fusion_reproduction.ps1
python scripts/run_gp_gpt_masking.py
```

The first command produces the GP/GPT source checkpoints under `outputs/feature_fusion_random_seed1234/`. The second writes SHAP values, each masking step's parameters, weights and development predictions to `outputs/gp_gpt_masking_dev_seed1234/`. Its development-set plots and source tables go to `outputs/gp_gpt_masking_figures_seed1234/`. Both commands train models and require a suitable GPU.

The development plots may show bootstrap variability around development metrics. Published Figure 3 instead uses each full fixed-test set once, without bootstrap or error bars. Its source data are in `results/figure_source_data/`; the reported final checkpoints are in `checkpoints/`.
