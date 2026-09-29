#!/usr/bin/env python3
"""Calculate MAE, MSE, RMSE, R2, and PCC for prediction files."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


COLUMN_PAIRS = (("Correct", "Predict"), ("Actual", "Predicted"))


def read_predictions(path, actual_column=None, predicted_column=None):
    """Read a CSV/TSV file and return finite actual and predicted values."""
    frame = pd.read_csv(path, sep=None, engine="python")
    if actual_column and predicted_column:
        column_pair = (actual_column, predicted_column)
    else:
        column_pair = next(
            (pair for pair in COLUMN_PAIRS if set(pair).issubset(frame.columns)),
            None,
        )
    if column_pair is None or not set(column_pair).issubset(frame.columns):
        expected = " or ".join(f"{actual}/{predicted}" for actual, predicted in COLUMN_PAIRS)
        raise ValueError(f"{path}: expected columns {expected}")

    actual = pd.to_numeric(frame[column_pair[0]], errors="coerce").to_numpy()
    predicted = pd.to_numeric(frame[column_pair[1]], errors="coerce").to_numpy()
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not valid.any():
        raise ValueError(f"{path}: no finite actual/predicted pairs")
    if not valid.all():
        print(f"Warning: {path}: discarded {(~valid).sum()} invalid rows")
    return actual[valid], predicted[valid]


def calculate_metrics(actual, predicted):
    """Return the repository's standard regression metrics."""
    mse = mean_squared_error(actual, predicted)
    enough_variation = len(actual) > 1 and np.std(actual) > 0 and np.std(predicted) > 0
    return {
        "count": len(actual),
        "mae": mean_absolute_error(actual, predicted),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(actual, predicted) if len(actual) > 1 and np.std(actual) > 0 else np.nan,
        "pcc": np.corrcoef(actual, predicted)[0, 1] if enough_variation else np.nan,
    }


def format_markdown(rows):
    header = "| Dataset | Count | MAE | MSE | RMSE | R2 | PCC |\n"
    separator = "|---|---:|---:|---:|---:|---:|---:|\n"
    body = "".join(
        f"| {name} | {metrics['count']} | {metrics['mae']:.4f} | "
        f"{metrics['mse']:.4f} | {metrics['rmse']:.4f} | "
        f"{metrics['r2']:.4f} | {metrics['pcc']:.4f} |\n"
        for name, metrics in rows
    )
    return header + separator + body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="Prediction CSV/TSV files")
    parser.add_argument("--actual-column", help="Actual-value column; requires --predicted-column")
    parser.add_argument("--predicted-column", help="Prediction column; requires --actual-column")
    parser.add_argument("--output", type=Path, help="Optional Markdown output file")
    args = parser.parse_args()

    if bool(args.actual_column) != bool(args.predicted_column):
        parser.error("--actual-column and --predicted-column must be used together")

    rows = []
    all_actual = []
    all_predicted = []
    for path in args.files:
        actual, predicted = read_predictions(path, args.actual_column, args.predicted_column)
        rows.append((path.name, calculate_metrics(actual, predicted)))
        all_actual.append(actual)
        all_predicted.append(predicted)

    if len(rows) > 1:
        rows.append(("Combined", calculate_metrics(np.concatenate(all_actual), np.concatenate(all_predicted))))

    report = format_markdown(rows)
    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Saved metrics to: {args.output}")


if __name__ == "__main__":
    main()
