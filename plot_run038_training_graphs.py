'''
Plots the training graphs for the final run used in the dissertation, run 38
'''
import argparse
import re
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+):\s+"
    r"train_loss=(?P<train_loss>[-+0-9.eE]+)\s+"
    r"train_abs=(?P<train_abs>[-+0-9.eE]+)\s+"
    r"train_rel=(?P<train_rel>[-+0-9.eE]+)\s+"
    r"val_loss=(?P<val_loss>[-+0-9.eE]+)\s+"
    r"val_abs=(?P<val_abs>[-+0-9.eE]+)\s+"
    r"val_rel=(?P<val_rel>[-+0-9.eE]+)"
)


def parse_log(log_path: Path) -> pd.DataFrame:
    rows = []

    with log_path.open("r", errors="replace") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if not m:
                continue

            d = m.groupdict()
            rows.append({
                "epoch": int(d["epoch"]),
                "train_loss": float(d["train_loss"]),
                "train_abs": float(d["train_abs"]),
                "train_rel": float(d["train_rel"]),
                "val_loss": float(d["val_loss"]),
                "val_abs": float(d["val_abs"]),
                "val_rel": float(d["val_rel"]),
            })

    if not rows:
        raise RuntimeError(
            f"No epoch summary lines found in {log_path}. "
            "Expected lines like: "
            "Epoch 17: train_loss=... train_abs=... train_rel=... val_loss=... val_abs=... val_rel=..."
        )

    df = pd.DataFrame(rows)

    # If the log contains duplicated epoch summaries, keep the final one.
    df = df.drop_duplicates(subset=["epoch"], keep="last")
    df = df.sort_values("epoch").reset_index(drop=True)

    return df


def plot_curve(df: pd.DataFrame, train_col: str, val_col: str, ylabel: str, title: str, out_base: Path):
    plt.figure(figsize=(6.5, 4.2))

    plt.plot(df["epoch"], df[train_col], marker="o", label="Train")
    plt.plot(df["epoch"], df[val_col], marker="o", label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(df["epoch"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print("Wrote:", png_path)
    print("Wrote:", pdf_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to run038 training log")
    parser.add_argument("--out_dir", default="run038_training_plots", help="Output directory")
    parser.add_argument("--max_epoch", type=int, default=20, help="Maximum epoch to plot")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = parse_log(log_path)
    df = df[df["epoch"] <= args.max_epoch].copy()

    if df.empty:
        raise RuntimeError(f"No epochs <= {args.max_epoch} found.")

    csv_path = out_dir / "run038_epoch_losses.csv"
    df.to_csv(csv_path, index=False)
    print("Wrote:", csv_path)

    print("\nParsed epoch losses:")
    print(df.to_string(index=False))

    plot_curve(
        df,
        train_col="train_loss",
        val_col="val_loss",
        ylabel="Total loss",
        title="Total loss during EGNN training",
        out_base=out_dir / "run038_total_loss",
    )

    plot_curve(
        df,
        train_col="train_abs",
        val_col="val_abs",
        ylabel="Absolute loss",
        title="Absolute loss during EGNN training",
        out_base=out_dir / "run038_absolute_loss",
    )

    plot_curve(
        df,
        train_col="train_rel",
        val_col="val_rel",
        ylabel="Relative loss",
        title="Relative loss during EGNN training",
        out_base=out_dir / "run038_relative_loss",
    )


if __name__ == "__main__":
    main()
