# Repository Guidelines

## Project Structure

`core/` handles data preparation, training batches, and scaffold splits. Model definitions are in `models/`; molecule, feature, metric, and plotting helpers are in `utils/`. Run-specific analysis and figure scripts live in `scripts/`, with checks in `tests/`. The released input CSVs and split assignments are in `data/`; frozen predictions, metrics, and figure source data are in `results/`; selected final weights are in `checkpoints/`. Write new runs to the ignored `outputs/` directory. Do not overwrite published results.

## Environment and Commands

There is no build step. On Windows, create and activate the supported environment from the sole dependency file:

```powershell
conda env create -f environment.yml
conda activate pytorch38-gpu
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests -q
python train_visnet_v2.py --help
python predict_visnet_v2.py --help
```

The environment targets Python 3.8, PyTorch 1.13.1, and CUDA 11.6. `README.md` gives a five-sample prediction command; `docs/代码导览.md` explains each Python file and the order of the longer training workflows. Keep the positive and negative modes, feature levels, and split assignments separate.

## Code and Tests

Use four-space indentation, `snake_case` for files and functions, `PascalCase` for classes, and explicit seeds for stochastic work. Match nearby code rather than reformatting unrelated modules. No formatter, linter, or coverage threshold is enforced. Name pytest files `test_<topic>.py` and tests `test_<behavior>()`. Add a focused regression test when changing split logic, feature masks, metric calculations, or restart behavior. State any GPU, checkpoint, or external-data requirement in the test or documentation.

## Results and Reviews

The CSV target columns are precomputed NORMAN RTI predictions, not experimental measurements. Keep model selection tied to the recorded development-set rule; report fixed-test and scaffold-fold results under their separate names. Commit messages use `type(scope): short action`, such as `fix: handle missing calibration inputs`. Pull requests should identify changed commands, tests run, and any effects on figures or published CSVs. Include before-and-after images for plotting changes. Check `DATA_ATTRIBUTION.md` and `THIRD_PARTY_NOTICES.md` before adding external data or code.
