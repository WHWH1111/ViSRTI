# PyTorch 与 PyG GPU 环境

仓库根目录的 `environment.yml` 是依赖版本来源。运行环境为 Windows、Python 3.8.20、PyTorch 1.13.1 和 CUDA 11.6，PyG 扩展与这一组版本配套。

## 创建或更新环境

```powershell
conda env create -f environment.yml
conda activate pytorch38-gpu
```

环境已存在时，先运行 `conda env update -n pytorch38-gpu -f environment.yml`，再激活；不要同时执行创建和更新。`environment.yml` 已固定 NumPy、SciPy、RDKit、PyG 及其 CUDA 扩展的兼容版本，也包含图 2 拼接和基线汇总所需的 Pillow 与 python-docx。

## 验证

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import numpy, scipy, rdkit, torch_geometric; print(numpy.__version__, scipy.__version__, rdkit.__version__, torch_geometric.__version__)"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests/test_model_smoke.py tests/test_etkdg_randomness.py tests/test_metrics_cli.py tests/test_plot_cli.py -q
```

预期 Torch 显示 `1.13.1+cu116`、CUDA 显示 `11.6` 且可用。禁用 pytest 外部插件自动加载可避免共享工作站中与本项目无关的插件影响测试；新建的干净环境通常不需要该变量。

PyG 扩展必须来自 `torch-1.13.0+cu116` wheel 索引。该命名同样适用于 PyTorch 1.13.1，不能替换成 CPU 或其他 CUDA 构建。
