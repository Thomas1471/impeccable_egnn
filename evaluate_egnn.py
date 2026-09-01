'''
Loads a trained EGNN checkpoint, scores graph shards and reconstructs predicted raw MMPBSA scores from the frame-residual target.
'''
import os
os.environ.setdefault("EGNN_EDGE_CUTOFF", "8.0")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from EGNN import EGNNRegressor
from egnn_data_utils import collate_graphs_plain_torch


def load_graphs(path):
    '''
    Loads graphs in
    '''
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ["graphs", "data_list", "dataset"]:
            if key in obj:
                return obj[key]

    raise TypeError(f"Could not interpret graph file: {path}")


def move_to_device(batch, device):
    if hasattr(batch, "to"):
        return batch.to(device)

    if isinstance(batch, dict):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out

    for name in dir(batch):
        if name.startswith("_"):
            continue
        try:
            v = getattr(batch, name)
        except Exception:
            continue
        if torch.is_tensor(v):
            try:
                setattr(batch, name, v.to(device))
            except Exception:
                pass

    return batch


def get_state_dict(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]

    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]

    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]

    return obj


def clean_state_dict(state):
    return {
        key.removeprefix("module."): value
        for key, value in state.items()
    }


def detect_split_column(df):
    '''
    Find appropriate splitting column, to find sets between train, test and validation
    '''
    if "split" in df.columns:
        return "split"

    for c in ["is_train", "train", "is_training"]:
        if c in df.columns:
            return c

    raise ValueError(
        "Could not find split column. Expected one of: "
        "split, is_train, train, is_training. "
        f"Columns are: {list(df.columns)}"
    )


def filter_split(df, split):
    if split == "all":
        return df.copy()

    col = detect_split_column(df)

    if col == "split":
        s = df[col].astype(str).str.lower()

        if split == "train":
            return df[s.isin(["train", "training"])].copy()

        if split == "val":
            return df[s.isin(["val", "valid", "validation", "dev"])].copy()

        if split == "test":
            return df[s.isin(["test", "testing", "holdout"])].copy()

        raise ValueError(f"Unknown split: {split}")

    vals = df[col]

    if vals.dtype == bool:
        is_train = vals
    else:
        is_train = vals.astype(str).str.lower().isin(
            ["true", "1", "train", "training"]
        )

    if split == "train":
        return df[is_train].copy()

    if split == "val":
        return df[~is_train].copy()

    raise ValueError(
        f"Cannot infer split '{split}' from column '{col}'. "
        "For test-set evaluation, metadata must contain a split column "
        "with values train/val/test."
    )


def detect_frame_col(df, scaling):
    for k in ["frame_col", "frame_column"]:
        if k in scaling and scaling[k] in df.columns:
            return scaling[k]

    candidates = [
        "frame",
        "frame_idx",
        "frame_index",
        "frame_num",
        "pose",
        "pose_idx",
        "pose_index",
        "pose_num",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Could not detect frame/pose column. Columns are: {list(df.columns)}"
    )


def load_scaling(path):
    with open(path) as f:
        scaling = json.load(f)

    mean_keys = ["residual_mean", "target_mean", "mean"]
    std_keys = ["residual_std", "target_std", "std"]

    residual_mean = None
    residual_std = None

    for k in mean_keys:
        if k in scaling:
            residual_mean = float(scaling[k])
            break

    for k in std_keys:
        if k in scaling:
            residual_std = float(scaling[k])
            break

    if residual_mean is None or residual_std is None:
        raise ValueError(
            f"Could not find residual mean/std in scaling file: {scaling}"
        )

    return scaling, residual_mean, residual_std


def pearson(a, b):
    '''
    returns pearson correlation
    '''
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2:
        return np.nan

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    '''
    Returns the spearman correlation
    '''
    a = pd.Series(a).rank(method="average").to_numpy()
    b = pd.Series(b).rank(method="average").to_numpy()
    return pearson(a, b)


def regression_metrics(df, pred_col):
    '''
    Computes regression metrics such as MAE, RMSE, and R^2
    '''
    y = df["mmpbsa"].to_numpy(dtype=float)
    p = df[pred_col].to_numpy(dtype=float)

    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    pr = pearson(y, p)
    sp = spearman(y, p)

    sst = float(np.sum((y - np.mean(y)) ** 2))
    sse = float(np.sum((y - p) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan

    return {
        "rows": int(len(df)),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Pearson": pr,
        "Spearman": sp,
        "true_std": float(np.std(y)),
        "pred_std": float(np.std(p)),
    }


def ranking_metrics(df, score_col):
    '''
    Computes the hitting range for each compound
    '''
    rows = []

    for cid, g in df.groupby("compound_num", sort=False):
        g = g.copy()

        best_true = float(g["mmpbsa"].min())
        selected = g.sort_values(score_col, ascending=True).iloc[0]
        selected_true = float(selected["mmpbsa"])

        selected_rank = int((g["mmpbsa"] < selected_true).sum() + 1)
        regret = selected_true - best_true

        rows.append({
            "compound_num": cid,
            "regret": regret,
            "selected_rank": selected_rank,
            "n_poses": len(g),
            "top1_hit": selected_rank <= 1,
            "top3_hit": selected_rank <= 3,
            "top5_hit": selected_rank <= 5,
            "top10_hit": selected_rank <= 10,
        })

    r = pd.DataFrame(rows)

    return {
        "compounds": int(len(r)),
        "mean_regret": float(r["regret"].mean()),
        "median_regret": float(r["regret"].median()),
        "mean_selected_rank": float(r["selected_rank"].mean()),
        "median_selected_rank": float(r["selected_rank"].median()),
        "top1_hit": float(r["top1_hit"].mean()),
        "top3_hit": float(r["top3_hit"].mean()),
        "top5_hit": float(r["top5_hit"].mean()),
        "top10_hit": float(r["top10_hit"].mean()),
    }


def m90(df, score_col, true_top):
    '''
    Computes the IMPECCABLE -style top-p metrics on 90% of compounds
    '''
    max_poses = int(df.groupby("compound_num").size().max())

    for m in range(1, max_poses + 1):
        hits = []

        for _, g in df.groupby("compound_num", sort=False):
            g = g.copy()

            true_top_ids = set(
                g.sort_values("mmpbsa", ascending=True)
                 .head(min(true_top, len(g)))["eval_row_id"]
                 .tolist()
            )

            pred_top_ids = set(
                g.sort_values(score_col, ascending=True)
                 .head(min(m, len(g)))["eval_row_id"]
                 .tolist()
            )

            hits.append(len(true_top_ids & pred_top_ids) > 0)

        prob = float(np.mean(hits))

        if prob >= 0.90:
            return m

    return None


def within_correlations(df, pred_col):
    '''
    Computes within correlation metrics
    '''
    ps = []
    ss = []

    for _, g in df.groupby("compound_num", sort=False):
        if len(g) < 3:
            continue

        ps.append(pearson(g["mmpbsa"], g[pred_col]))
        ss.append(spearman(g["mmpbsa"], g[pred_col]))

    ps = [x for x in ps if np.isfinite(x)]
    ss = [x for x in ss if np.isfinite(x)]

    return {
        "mean_within_pearson": float(np.mean(ps)) if ps else np.nan,
        "median_within_pearson": float(np.median(ps)) if ps else np.nan,
        "mean_within_spearman": float(np.mean(ss)) if ss else np.nan,
        "median_within_spearman": float(np.median(ss)) if ss else np.nan,
    }


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--sharded_root", required=True)
    ap.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
    )
    ap.add_argument("--out_prefix", required=True)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--in_dim", type=int, default=6)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    sharded_root = Path(args.sharded_root)

    metadata_file = run_dir / "sharded_metadata_with_targets.csv"
    scaling_file = run_dir / "residual_target_scaling.json"

    print("metadata_file:", metadata_file)
    print("scaling_file:", scaling_file)
    print("checkpoint:", args.checkpoint)

    meta = pd.read_csv(metadata_file)
    meta["mmpbsa"] = pd.to_numeric(meta["mmpbsa"], errors="coerce")

    before = len(meta)
    meta = meta[np.isfinite(meta["mmpbsa"].values)].copy()

    print(f"Finite-label rows: {len(meta)} / {before}")

    scaling, residual_mean, residual_std = load_scaling(scaling_file)
    frame_col = detect_frame_col(meta, scaling)

    print("residual_mean:", residual_mean)
    print("residual_std:", residual_std)
    print("frame_col:", frame_col)

    df = filter_split(meta, args.split)
    df = df.copy()
    df["eval_row_id"] = np.arange(len(df))

    if len(df) == 0:
        raise ValueError(f"No rows found for split '{args.split}'.")

    print("split:", args.split)
    print("rows:", len(df))
    print("compounds:", df["compound_num"].nunique())

    required = [
        "source_shard",
        "local_graph_index",
        "target_norm",
        "frame_template_score",
        "mmpbsa",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    device = get_device()
    print("device:", device)

    model = EGNNRegressor(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    state = clean_state_dict(get_state_dict(args.checkpoint))
    model.load_state_dict(state, strict=True)
    model.eval()

    outputs = []

    with torch.no_grad():
        for shard_name in sorted(df["source_shard"].unique()):
            shard_df = df[df["source_shard"] == shard_name].copy()
            shard_df = shard_df.sort_values(
                ["compound_num", "local_graph_index"]
            ).reset_index(drop=True)

            graph_path = None

            if "shard_graph_file" in shard_df.columns:
                graph_path = Path(str(shard_df["shard_graph_file"].iloc[0]))

            if graph_path is None or not graph_path.exists():
                graph_path = sharded_root / shard_name / "graph_metadata_graphs.pt"

            print()
            print("=" * 100)
            print("Loading shard:", shard_name)
            print("graph_path:", graph_path)
            print("rows to score:", len(shard_df))

            graphs = load_graphs(graph_path)

            for start in range(0, len(shard_df), args.batch_size):
                rows = shard_df.iloc[start:start + args.batch_size]

                batch_graphs = []

                for _, row in rows.iterrows():
                    local_idx = int(row["local_graph_index"])
                    g = graphs[local_idx]
                    g.y = torch.tensor(
                        [float(row["target_norm"])],
                        dtype=torch.float32,
                    )
                    batch_graphs.append(g)

                batch = collate_graphs_plain_torch(batch_graphs)
                batch = move_to_device(batch, device)

                pred_norm = model(batch).detach().view(-1).cpu().numpy()

                tmp = rows.copy()
                tmp["pred_norm"] = pred_norm
                outputs.append(tmp)

                if (start // args.batch_size) % 500 == 0:
                    print(
                        f"scored {start + len(rows)}/{len(shard_df)}",
                        flush=True,
                    )

            del graphs

            if device.type == "xpu":
                try:
                    torch.xpu.empty_cache()
                except Exception:
                    pass
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    scored = pd.concat(outputs, ignore_index=True)

    scored["pred_residual"] = (
        scored["pred_norm"] * residual_std + residual_mean
    )
    scored["pred_score"] = (
        scored["frame_template_score"] + scored["pred_residual"]
    )
    scored["lower_frame_score"] = scored[frame_col].astype(float)
    scored["frame_template_baseline_score"] = (
        scored["frame_template_score"].astype(float)
    )

    out_csv = Path(args.out_prefix + "_scores.csv")
    out_txt = Path(args.out_prefix + "_report.txt")
    out_json = Path(args.out_prefix + "_summary.json")

    scored.to_csv(out_csv, index=False)

    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "rows": int(len(scored)),
        "compounds": int(scored["compound_num"].nunique()),
        "methods": {},
    }

    report_lines = []
    report_lines.append(f"checkpoint: {args.checkpoint}")
    report_lines.append(f"split: {args.split}")
    report_lines.append(f"rows: {len(scored)}")
    report_lines.append(f"compounds: {scored['compound_num'].nunique()}")
    report_lines.append("")

    for name, col in [
        ("MODEL", "pred_score"),
        ("DEFRAMED_MODEL", "pred_residual"),
        ("LOWER_FRAME", "lower_frame_score"),
        ("FRAME_TEMPLATE", "frame_template_baseline_score"),
    ]:
        report_lines.append("=" * 100)
        report_lines.append(name)
        report_lines.append("=" * 100)

        method_summary = {}

        rank = ranking_metrics(scored, col)
        method_summary["ranking"] = rank

        report_lines.append("RANKING")
        for k, v in rank.items():
            report_lines.append(f"  {k}: {v}")

        method_summary["m90"] = {}

        report_lines.append("M90")
        for p in [1, 3, 5]:
            m = m90(scored, col, p)
            method_summary["m90"][f"true_top_{p}"] = m
            report_lines.append(f"  m90 true_top_{p}: {m}")

        if col in ["pred_score", "frame_template_baseline_score"]:
            reg = regression_metrics(scored, col)
            method_summary["regression"] = reg

            report_lines.append("REGRESSION")
            for k, v in reg.items():
                report_lines.append(f"  {k}: {v}")

        if col in [
            "pred_score",
            "pred_residual",
            "frame_template_baseline_score",
        ]:
            wc = within_correlations(scored, col)
            method_summary["within_compound_correlation"] = wc

            report_lines.append("WITHIN-COMPOUND CORRELATION")
            for k, v in wc.items():
                report_lines.append(f"  {k}: {v}")

        summary["methods"][name] = method_summary
        report_lines.append("")

    text = "\n".join(report_lines)

    out_txt.write_text(text)

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(text)
    print("Wrote:", out_csv)
    print("Wrote:", out_txt)
    print("Wrote:", out_json)


if __name__ == "__main__":
    main()