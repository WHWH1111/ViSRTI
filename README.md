# ViSRTI

ViSRTI uses molecular structure and optional descriptors to approximate the **NORMAN-predicted retention time index (RTI)** in positive and negative ionization modes. The target columns in the released CSV files are precomputed predictions, not experimental RTI measurements. The 21 standards used for laboratory RT–RTI calibration and the BaP/BeP case compounds are not an independent experimental RTI test set.

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

For negative mode, use `checkpoints/neg_GPT/` and `MMF_GNN_neg.csv`. Input CSVs need SMILES plus the descriptors active in the chosen model; missing descriptor handling follows the training code and should be checked before interpreting a new prediction. Full commands and a file-by-file guide are in [代码导览](docs/代码导览.md).

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

The split manifests record seed 1234, source-file SHA-256, counts, and scaffold overlap checks. Their `source_file` paths and the released checkpoint parameter paths were made relative for this publication copy; the original source hashes and model-weight hashes are retained. Do not compare the publication JSON files byte-for-byte with the original machine-local manifests.
Any `original_run/` entry in a results CSV is a historical log or experiment identifier, not a file included in this release. The numerical columns and per-sample predictions are unchanged.

## Model workflow

The four feature levels are G (molecular graph), GP (graph + physicochemical features), GPT (adds toxicity features), and GPTC (adds chromatographic features). `train_visnet_v2.py` trains one configuration; `scripts/run_feature_fusion_reproduction.ps1` runs the fixed random-split G → GP → GPT → GPTC comparison. The selected positive candidate is GP with Koc and LogP masked; the selected negative candidate is unmasked GPT. `scripts/run_gp_gpt_masking.py` ranks auxiliary features with SHAP and stops deletion when development-set MAE first worsens. `scripts/run_final_model_updates.py` computes the downstream fixed-test analyses and calls `scripts/run_selected_model_scaffold_cv.ps1` for five-fold structural checks. The latter must use the frozen `selection_manifest.json`; it is **not** a new model-selection search.

The GP/GPT candidate choice was specified after earlier test results had been inspected. Development-set MAE controlled only the subsequent masking decision, so the fixed-test results should not be described as untouched evidence for choosing GP over every alternative. The scaffold runs assess robustness of the fixed configuration and are not nested selection cross-validation.

For an overview of each `.py` file and the order to run them, read [代码导览](docs/代码导览.md). The `results/` directory is a read-only record of the reported run; new executions write to ignored `outputs/` directories. Do not overwrite published results.

The manuscript's repository-path wording is in [Data and Software Availability](docs/Data_and_Software_Availability.md).

The released baseline CSVs allow the reported predictions and metrics to be checked. Baseline checkpoints and temporary graph caches are not shipped; running those models again requires retraining after the G experiment. The historical K/M workbook and Supporting Information comparison in `scripts/data-val/aggregate_baseline_results.py` also requires the original external workbooks, which are not included.

## Scope of experimental data

This repository does not supply an independent experimental RTI benchmark. The 21 calibration standards and BaP/BeP case information are described in the Supporting Information; the 23-compound experimental workbook is not included here. The prediction and calibration code is provided, but reproducing laboratory-specific RT–RTI calibration requires confirmed experimental inputs.

Project code is distributed under the [MIT license](LICENSE). The NORMAN-derived data carry separate [source attribution and CC BY 4.0 reuse terms](DATA_ATTRIBUTION.md); [third-party code notices](THIRD_PARTY_NOTICES.md) are also included.
