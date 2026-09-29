import subprocess
import sys
from pathlib import Path


def test_metrics_cli_accepts_a_single_tsv(tmp_path):
    prediction_file = tmp_path / "predictions.tsv"
    prediction_file.write_text(
        "SMILES\tCorrect\tPredict\nCCO\t1\t1\nCCN\t2\t2\nCCC\t3\t3\n",
        encoding="utf-8",
    )
    script = Path(__file__).parents[1] / "scripts" / "data-val" / "calculate_metrics.py"

    result = subprocess.run(
        [sys.executable, str(script), str(prediction_file)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "| predictions.tsv | 3 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |" in result.stdout
