# GP/GPT development-set SHAP and masking

This workflow analyzes the user-requested positive-mode GP and negative-mode GPT
checkpoints from `outputs/feature_fusion_random_seed1234/`. It does not change the
manuscript, Figure 2, scaffold experiments, or subsequent category analyses.

## Fixed analysis rules

- Reuse the recorded training and development assignments. The development sets
  contain 561 positive-mode and 534 negative-mode compounds.
- Explain each compound with its own molecular graph, held fixed while the
  descriptors vary. Apply the checkpoint's feature standardization exactly once.
- Use 50 training compounds as the SHAP background, with seed 1234. Enumerate all
  descriptor coalitions; report both standardized-target and RTI-unit values.
- Rank descriptors by unmasked development-set mean absolute SHAP. Remove them
  cumulatively in that fixed ascending order, retraining after each removal.
- Preserve the original training settings: up to 150 epochs, seed 1234,
  development split seed 42, L1 loss, and development-MAE early stopping with
  patience 20. Stop at the first increase in development MAE and retain the
  preceding configuration. The other three metrics are reported, not substituted
  for the stopping criterion after results are available.
- Every candidate uses `train_visnet_v2.py --development-only`. Test responses
  are not loaded or evaluated. Confirm identical training/development assignments,
  standardization, and development response values before accepting a result.
- Preserve the original pre-filter descriptor pool for standardization. Negative
  mode used 4,805 training rows to calculate those statistics, but only 4,804
  successfully processed molecules for fitting. The frozen effective partition
  file prevents the excluded molecule from being retried or reintroduced.

## Entry points

```powershell
conda run -n pytorch38-gpu python scripts/run_gp_gpt_masking.py
conda run -n pytorch38-gpu python scripts/plt/plot_gp_gpt_masking.py --run-root outputs/gp_gpt_masking_dev_seed1234 --output-dir 修改部分内容/06_GP_GPT逐步掩码分析
```

The first command refuses to overwrite an existing run. Do not launch it while
the recorded supervisor is active. The second command re-exports completed
evidence without training again. `analyze_shap.py` and the archived GPT supplement
are not sources for this analysis.

## Interpretation and delivery

Positive GP contains four physicochemical descriptors and no toxicity encoder;
negative GPT adds four toxicity descriptors and uses a 128-dimensional toxicity
encoder output. G denotes the molecular graph, not four named descriptors.

Bar heights are metrics calculated on the entire development set. Error bars are
the standard deviation of 1,000 paired bootstrap resamples of that set. They do
not measure variability across training seeds or provide an independent estimate
of generalization after model selection.

GP/GPT are the requested candidates for this analysis. Earlier negative-mode
development results favored GPTC across all four metrics. The previous test
results have already been inspected; do not label the requested choice as an
unbiased development-only search across all configurations. SHAP attribution is
not evidence of causality or universal feature redundancy.

Deliver the completed figure, Chinese-named source CSVs, every masking-stage
configuration, and the selected checkpoints under
`修改部分内容/06_GP_GPT逐步掩码分析/`. A run marked `completed` still requires inspection
of the final rendered figure before visual review is marked as passed.
