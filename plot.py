#!/usr/bin/env python3
"""
Plot train and val IoU over epochs for selected classes from a Lightning/TensorBoard metrics CSV.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def find_iou_columns(df, class_name):
    """
    Find train, val, and test IoU column names for a class (case-insensitive match on suffix).
    Returns (train_col, val_col, test_col); any can be None if not found.
    """
    prefix_train = "train/iou_"
    prefix_val = "val/iou_"
    prefix_test = "test/iou_"
    name_lower = class_name.strip().lower()

    train_col = val_col = test_col = None
    for c in df.columns:
        if c.startswith(prefix_train) and c[len(prefix_train):].lower() == name_lower:
            train_col = c
        if c.startswith(prefix_val) and c[len(prefix_val):].lower() == name_lower:
            val_col = c
        if c.startswith(prefix_test) and c[len(prefix_test):].lower() == name_lower:
            test_col = c

    return (train_col, val_col, test_col)


def main():
    parser = argparse.ArgumentParser(
        description="Plot train and val IoU over epochs for selected classes from a metrics CSV."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to metrics CSV file.")
    parser.add_argument("--output_png", type=str, required=True, help="Path for output PNG.")
    parser.add_argument("--train", action="store_true", help="Plot train IoU values.")
    parser.add_argument("--valid", action="store_true", help="Plot validation IoU (val/iou_*) values.")
    parser.add_argument("--test", action="store_true", help="Plot test IoU (test/iou_*) values.")
    parser.add_argument(
        "plot",
        nargs="+",
        help="Class names to plot (e.g. Water wire). Matches IoU columns case-insensitively.",
    )
    args = parser.parse_args()

    if not (args.train or args.valid or args.test):
        print("Error: At least one of --train, --valid, or --test must be present.", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    if not args.plot:
        print("Error: At least one class name required for --plot.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Filter to rows with valid epoch
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"])
    df = df.astype({"epoch": int})

    # Train/val/test are often logged in different rows per epoch; merge by taking last non-null per column
    def last_valid(series):
        valid = pd.to_numeric(series, errors="coerce").dropna()
        return valid.iloc[-1] if len(valid) else float("nan")

    df = df.groupby("epoch", as_index=False).agg(last_valid)
    df = df.sort_values("epoch")
    epochs = df["epoch"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10.colors
    for i, class_name in enumerate(args.plot):
        train_col, val_col, test_col = find_iou_columns(df, class_name)
        if train_col is None and val_col is None and test_col is None:
            print(f"Warning: No IoU columns found for class '{class_name}', skipping.", file=sys.stderr)
            continue
        color = colors[i % len(colors)]
        label_base = class_name.strip()
        if args.train and train_col is not None:
            y = pd.to_numeric(df[train_col], errors="coerce")
            ax.plot(epochs, y, color=color, linestyle="-", label=f"train {label_base}", linewidth=1.5)
        if args.valid and val_col is not None:
            y = pd.to_numeric(df[val_col], errors="coerce")
            ax.plot(epochs, y, color=color, linestyle="--", label=f"val {label_base}", linewidth=1.5)
        if args.test and test_col is not None:
            y = pd.to_numeric(df[test_col], errors="coerce")
            ax.plot(epochs, y, color=color, linestyle=":", label=f"test {label_base}", linewidth=1.5)

    parts = []
    if args.train:
        parts.append("train")
    if args.valid:
        parts.append("validation")
    if args.test:
        parts.append("test")
    title = " and ".join(parts).capitalize() + " IoU over epochs"
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out_path = Path(args.output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
