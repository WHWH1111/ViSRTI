import subprocess
import sys
from pathlib import Path


def test_feature_fusion_plot_cli_writes_outputs(tmp_path):
    metrics_file = tmp_path / "metrics.csv"
    metrics_file.write_text(
        "mode,feature_code,mae,rmse,r2,pcc,mae_bootstrap_sd,rmse_bootstrap_sd,r2_bootstrap_sd,pcc_bootstrap_sd\n"
        "pos,G,44,58,0.93,0.97,2,3,0.01,0.01\n"
        "pos,GP,42,55,0.94,0.97,2,3,0.01,0.01\n"
        "pos,GPT,38,51,0.95,0.98,2,3,0.01,0.01\n"
        "pos,GPTC,41,54,0.94,0.97,2,3,0.01,0.01\n"
        "neg,G,51,66,0.88,0.94,2,3,0.01,0.01\n"
        "neg,GP,48,61,0.91,0.96,2,3,0.01,0.01\n"
        "neg,GPT,41,55,0.92,0.96,2,3,0.01,0.01\n"
        "neg,GPTC,44,59,0.91,0.96,2,3,0.01,0.01\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "plot"
    script = Path(__file__).parents[1] / "scripts" / "plt" / "plot_test_metrics_bar_line.py"

    subprocess.run(
        [sys.executable, str(script), "--input", str(metrics_file), "--output-dir", str(output_dir)],
        check=True,
        capture_output=True,
        text=True,
    )

    for suffix in ("svg", "pdf", "png", "tiff"):
        assert (output_dir / ("Figure2ef_feature_fusion." + suffix)).is_file()
