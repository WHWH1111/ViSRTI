# ViSRTI

ViSRTI predicts positive- and negative-ionization retention time indices (RTI) from molecular structure and optional descriptors. The training targets are **NORMAN-predicted RTI**, not experimental measurements.

## Start here

The supported environment is Windows, Python 3.8.20, PyTorch 1.13.1 with CUDA 11.6, RDKit 2023.9.6, and the package versions in the single dependency file `environment.yml`. Activate the environment before running commands from the repository root:

```powershell
conda env create -f environment.yml
conda activate pytorch38-gpu
python -c "import torch, rdkit; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), rdkit.__version__)"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests/test_model_smoke.py tests/test_etkdg_randomness.py tests/test_metrics_cli.py tests/test_plot_cli.py -q
```

If the environment already exists, use `conda env update -n pytorch38-gpu -f environment.yml` instead of `conda env create`. See [GPU setup](docs/setup/pytorch_pyg_gpu_installation.md) for the PyTorch Geometric wheel details.

To make a quick prediction with the released final positive-mode checkpoint, run:

```powershell
python predict_visnet_v2.py --model_path checkpoints/pos_GP_mask_Koc_LogP/best.pt --params_path checkpoints/pos_GP_mask_Koc_LogP/training_params.json --input_file data/MMF-3/MMF_GNN_pos.csv --output_dir outputs/predict_pos --max_items 5
```

For negative mode:

```powershell
python predict_visnet_v2.py --model_path checkpoints/neg_GPT/best.pt --params_path checkpoints/neg_GPT/training_params.json --input_file data/MMF-3/MMF_GNN_neg.csv --output_dir outputs/predict_neg --max_items 5
```

Input CSVs need SMILES and the descriptors used by the selected model. The preprocessing code fills missing descriptor values with zero, so check these fields before interpreting predictions for new compounds. The script reports evaluation metrics and an actual-versus-predicted plot only when at least two predictions are available and each has an RTI label; an unlabeled CSV still yields predictions. For the full workflow and each Python file's role, see [代码导览](docs/代码导览.md).

## Data, splits, and results

| Location | Purpose |
| --- | --- |
| `data/MMF-3/MMF_GNN_pos.csv`, `MMF_GNN_neg.csv` | Curated modeling data with SMILES, descriptor columns, and `Pred_RTI_Positive_ESI` or `Pred_RTI_Negative_ESI`. |
| `data/splits/random/` | Fixed random assignments used for the original baseline and feature-fusion comparisons. |
| `data/MMF-3/MMF_GNN_pos/`, `MMF_GNN_neg/` | Separate Bemis–Murcko scaffold holdout assignments. |
| `data/MMF-3/MMF_GNN_pos_5/`, `MMF_GNN_neg_5/` | Five-fold scaffold assignments. |
| `results/baseline_random_seed1234/`, `results/feature_fusion_random_seed1234/` | Figure 2 baseline and G → GP → GPT → GPTC metrics and predictions. |
| `results/final_masked_models_seed1234/` and `checkpoints/` | Final model selection record, weights, parameters, and fixed-test predictions. |
| `results/final_masked_scaffold_seed1234/` | Fold metrics, pooled out-of-fold predictions, category metrics, and Figure 5 source data. |
| `results/figure_source_data/`, `results/figures/` | Figure 3, Figure 4, Figure 5, and Figure S5 plotting data and vector figures. |

The split manifests record seed 1234, source-file SHA-256, sample counts, and scaffold overlap checks. Paths are repository-relative. An `original_run` column in a results CSV identifies its source experiment; it is not an input path.

## Model workflow

The four feature levels are G (molecular graph), GP (graph + physicochemical features), GPT (adds toxicity features), and GPTC (adds chromatographic features). `train_visnet_v2.py` trains one configuration; `scripts/run_feature_fusion_reproduction.ps1` runs the fixed random-split G → GP → GPT → GPTC comparison. The final positive model is GP with Koc and LogP masked; the final negative model is unmasked GPT. `scripts/run_gp_gpt_masking.py` ranks auxiliary features with development-set SHAP and stops removal when development MAE first worsens. `scripts/run_final_model_updates.py` produces fixed-test analyses and calls `scripts/run_selected_model_scaffold_cv.ps1` for five-fold scaffold evaluation. The reported selection is in `results/final_masked_models_seed1234/selection_manifest.json`; a new run freezes its own manifest under `outputs/final_masked_models_seed1234/`.

GP and GPT were chosen as masking candidates after earlier test results had been inspected. Development MAE selected masks within those candidates. The fixed-test metrics report the resulting models; they are not an untouched comparison across all fusion candidates. The scaffold runs assess the fixed configurations rather than performing nested model selection.

For an overview of each `.py` file and the execution order, read [代码导览](docs/代码导览.md). `results/` contains the reported run; new executions write to the ignored `outputs/` directory.

The manuscript's repository-path wording is in [Data and Software Availability](docs/Data_and_Software_Availability.md).

The baseline CSVs contain predictions and metrics. Baseline checkpoints and graph caches are not included; baseline inference requires retraining. `scripts/data-val/aggregate_baseline_results.py` needs the external K/M workbooks and Supporting Information to repeat their comparison.

## Scope of experimental data

The Supporting Information describes 21 laboratory calibration standards and the separate BaP/BeP case study. These compounds do not form an independent experimental RTI test set. The experimental workbook is not included; laboratory-specific RT–RTI calibration requires its confirmed inputs.

Project code is distributed under the [MIT license](LICENSE). The NORMAN-derived data carry separate [source attribution and CC BY 4.0 reuse terms](DATA_ATTRIBUTION.md); [third-party code notices](THIRD_PARTY_NOTICES.md) are also included.
