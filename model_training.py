'''
Main training file
'''

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from EGNN import EGNNRegressor
from egnn_data_utils import collate_graphs_plain_torch


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_device_cache(device):
    if device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def move_to_device(obj, device):
    if hasattr(obj, "to"):
        return obj.to(device)

    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}

    if torch.is_tensor(obj):
        return obj.to(device)

    if hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if torch.is_tensor(v):
                setattr(obj, k, v.to(device))
        return obj

    return obj


def torch_load_graphs(path):
    '''
    Loads graphs in
    '''
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        for key in ["graphs", "data_list", "dataset"]:
            if key in obj:
                return obj[key]

    return obj


def find_frame_col(df):
    candidates = [
        "frame",
        "frame_idx",
        "frame_index",
        "pose",
        "pose_idx",
        "pose_index",
        "pose_num",
        "frame_num",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        low = col.lower()
        if "frame" in low or "pose" in low:
            if pd.api.types.is_numeric_dtype(df[col]):
                return col

    raise RuntimeError(f"Could not find frame/pose column. Columns: {df.columns.tolist()}")


def discover_shards(sharded_root, max_shards=None):
    '''
    Find shards to use in training
    '''
    root = Path(sharded_root)
    shards = []

    for shard in sorted(root.glob("shard_*")):
        meta = shard / "graph_metadata.csv"
        graphs = shard / "graph_metadata_graphs.pt"
        if meta.exists() and graphs.exists():
            shards.append(shard)

    if max_shards is not None:
        shards = shards[:max_shards]

    if not shards:
        raise RuntimeError(f"No valid shards found under {root}")

    return shards


def build_combined_metadata(shards):
    '''
    Combine both graphs and metadata into one
    '''
    dfs = []

    for shard in shards:
        meta_file = shard / "graph_metadata.csv"
        graph_file = shard / "graph_metadata_graphs.pt"

        df = pd.read_csv(meta_file)
        df["source_shard"] = shard.name
        df["shard_graph_file"] = str(graph_file)
        df["local_graph_index"] = np.arange(len(df), dtype=np.int64)

        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)

    required = ["compound_num", "mmpbsa"]
    for col in required:
        if col not in full.columns:
            raise RuntimeError(f"Missing required metadata column: {col}")

    return full


def make_split_and_targets(df, train_frac, seed):
    '''
    Split up the data into train, test and validation
    '''
    df = df.copy()
    frame_col = find_frame_col(df)

    rng = random.Random(seed)

    # Compound-level split:
    # every pose from a given compound is assigned to exactly one split.
    # With --train_frac 0.70 this gives approximately 70/15/15
    # train/val/test compounds.
    df["split"] = "unassigned"

    compounds = np.array(sorted(df["compound_num"].unique()))
    rng.shuffle(compounds)

    n_compounds = len(compounds)
    n_train = int(train_frac * n_compounds)
    remaining = n_compounds - n_train
    n_val = remaining // 2
   

    train_compounds = set(compounds[:n_train])
    val_compounds = set(compounds[n_train:n_train + n_val])
    test_compounds = set(compounds[n_train + n_val:])

    #Check that splits are independent of each other
    if len(train_compounds & val_compounds):
        raise RuntimeError("Train/val compound overlap detected")
    if len(train_compounds & test_compounds):
        raise RuntimeError("Train/test compound overlap detected")
    if len(val_compounds & test_compounds):
        raise RuntimeError("Val/test compound overlap detected")

    df.loc[df["compound_num"].isin(train_compounds), "split"] = "train"
    df.loc[df["compound_num"].isin(val_compounds), "split"] = "val"
    df.loc[df["compound_num"].isin(test_compounds), "split"] = "test"

    if (df["split"] == "unassigned").any():
        bad = df.loc[df["split"] == "unassigned", "compound_num"].unique()
        raise RuntimeError(f"Unassigned compounds remain: {bad[:20]}")

    #Print Details of Splits
    print("Compound-level split:", flush=True)
    print(f"  total compounds: {n_compounds}", flush=True)
    print(f"  train compounds: {len(train_compounds)}", flush=True)
    print(f"  val compounds: {len(val_compounds)}", flush=True)
    print(f"  test compounds: {len(test_compounds)}", flush=True)
    print("Rows by split:", flush=True)
    print(df["split"].value_counts().sort_index(), flush=True)
    print("Compounds by split:", flush=True)
    print(df.groupby("split")["compound_num"].nunique().sort_index(), flush=True)

    #Find if any poses are not perfectly formed, like if any do not have 100 poses
    pose_counts = df.groupby("compound_num").size()
    bad_pose_counts = pose_counts[pose_counts != 100]
    if len(bad_pose_counts):
        print("WARNING: some compounds do not have exactly 100 poses", flush=True)
        print(bad_pose_counts.head(20), flush=True)
    else:
        print("All compounds have exactly 100 poses.", flush=True)

    train_df = df[df["split"] == "train"].copy()

    frame_template = train_df.groupby(frame_col)["mmpbsa"].mean()
    global_train_mean = float(train_df["mmpbsa"].mean())

    df["frame_template_score"] = df[frame_col].map(frame_template).fillna(global_train_mean)
    df["residual_target"] = df["mmpbsa"] - df["frame_template_score"]

    #Calculate residual used for residual split
    train_residual = df.loc[df["split"] == "train", "residual_target"]
    residual_mean = float(train_residual.mean())
    residual_std = float(train_residual.std(ddof=0))

    if residual_std <= 0:
        raise RuntimeError(f"Bad residual std: {residual_std}")

    df["target_norm"] = (df["residual_target"] - residual_mean) / residual_std

    scaling = {
        "target": "normalised_frame_residual",
        "frame_col": frame_col,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "train_frac": train_frac,
        "seed": seed,
        "frame_template": {str(k): float(v) for k, v in frame_template.items()},
    }

    return df, scaling


def make_batches(meta, batch_size, seed, shuffle):
    '''
    Make batches, grouping by compound number
    '''
    rng = random.Random(seed)
    batches = []

    for _, group in meta.groupby("compound_num"):
        idx = list(group.index)
        if shuffle:
            rng.shuffle(idx)

        for start in range(0, len(idx), batch_size):
            chunk = idx[start:start + batch_size]
            if len(chunk) >= 2:
                batches.append(chunk)

    if shuffle:
        rng.shuffle(batches)

    return batches


def calculate_losses(pred, target, compound_ids, lambda_relative):
    '''
    Function used to calculate the two losses used in this dissertation:
        absolute_loss: The regular mse loss of the prediction and the target
        relative_loss: The average loss based off of the prediction difference between MSE compound-centred predictions and targets,
                   encouraging within-compound pose discrimination.
        total_loss: A combination of the two losses, with the relative_loss being weighted more to emphasise ranking performance
    '''
    pred = pred.view(-1)
    target = target.view(-1)

    abs_loss = F.mse_loss(pred, target)

    rel_losses = []
    for cid in torch.unique(compound_ids):
        mask = compound_ids == cid
        if int(mask.sum().item()) < 2:
            continue

        pred_c = pred[mask] - pred[mask].mean()
        target_c = target[mask] - target[mask].mean()
        rel_losses.append(F.mse_loss(pred_c, target_c))

    if rel_losses:
        rel_loss = torch.stack(rel_losses).mean()
    else:
        rel_loss = torch.zeros((), device=pred.device)

    total = abs_loss + lambda_relative * rel_loss

    return total, abs_loss, rel_loss


def run_one_shard(
    model,
    optimizer,
    shard_name,
    shard_meta,
    batch_size,
    device,
    lambda_relative,
    max_norm,
    train,
    epoch_seed,
    max_batches=None,
    log_every=100,
):
    '''
    Performs one shard, for either training or evaluation
    '''
    graph_file = shard_meta["shard_graph_file"].iloc[0]
    graphs = torch_load_graphs(graph_file)

    batches = make_batches(
        shard_meta,
        batch_size=batch_size,
        seed=epoch_seed,
        shuffle=train,
    )

    if max_batches is not None:
        batches = batches[:max_batches]

    if train:
        model.train()
    else:
        model.eval()

    total_examples = 0
    total_loss_sum = 0.0
    abs_loss_sum = 0.0
    rel_loss_sum = 0.0

    t0 = time.time()

    for b, batch_indices in enumerate(batches, start=1):
        rows = shard_meta.loc[batch_indices]

        graph_batch = []
        for row in rows.itertuples(index=False):
            g = graphs[int(row.local_graph_index)]
            g.y = torch.tensor([float(row.target_norm)], dtype=torch.float32)
            graph_batch.append(g)

        batch = collate_graphs_plain_torch(graph_batch)
        batch = move_to_device(batch, device)

        target = torch.tensor(rows["target_norm"].values, dtype=torch.float32, device=device)
        compound_ids = torch.tensor(rows["compound_num"].values, dtype=torch.long, device=device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            pred = model(batch).view(-1)
            loss, abs_loss, rel_loss = calculate_losses(
                pred=pred,
                target=target,
                compound_ids=compound_ids,
                lambda_relative=lambda_relative,
            )

            if not torch.isfinite(loss):
                print("NONFINITE LOSS DETECTED", flush=True)
                print(f"shard={shard_name} batch={b}/{len(batches)} train={train}", flush=True)
                print(f"loss={loss.detach().cpu()} abs={abs_loss.detach().cpu()} rel={rel_loss.detach().cpu()}", flush=True)
                print("target finite:", bool(torch.isfinite(target).all().detach().cpu()), flush=True)
                print("pred finite:", bool(torch.isfinite(pred).all().detach().cpu()), flush=True)
                print("target min/max:", float(target.min().detach().cpu()), float(target.max().detach().cpu()), flush=True)
                print("pred min/max:", float(pred.min().detach().cpu()), float(pred.max().detach().cpu()), flush=True)
                raise RuntimeError("Stopping because loss became NaN/Inf")

            if train:
                #Performs a training loop
                loss.backward()
                if max_norm is not None and max_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    if not torch.isfinite(grad_norm):
                        print("NONFINITE GRAD NORM DETECTED", flush=True)
                        print(f"shard={shard_name} batch={b}/{len(batches)} grad_norm={grad_norm}", flush=True)
                        raise RuntimeError("Stopping because grad norm became NaN/Inf")
                optimizer.step()

                for name, param in model.named_parameters():
                    if not torch.isfinite(param).all():
                        print("NONFINITE PARAMETER DETECTED AFTER OPTIMIZER STEP", flush=True)
                        print(f"parameter={name}", flush=True)
                        raise RuntimeError("Stopping because model parameters became NaN/Inf")

        n = len(rows)
        total_examples += n
        total_loss_sum += float(loss.detach().cpu()) * n
        abs_loss_sum += float(abs_loss.detach().cpu()) * n
        rel_loss_sum += float(rel_loss.detach().cpu()) * n

        #Prints out losses
        if b % log_every == 0:
            mode = "train" if train else "val"
            print(
                f"{mode} shard={shard_name} batch={b}/{len(batches)} "
                f"loss={total_loss_sum / total_examples:.6f} "
                f"abs={abs_loss_sum / total_examples:.6f} "
                f"rel={rel_loss_sum / total_examples:.6f}",
                flush=True,
            )
    #Deletes current shard, to avoid memory issues
    del graphs
    gc.collect()
    clear_device_cache(device)

    elapsed = time.time() - t0

    if total_examples == 0:
        return {
            "loss": np.nan,
            "absolute": np.nan,
            "relative": np.nan,
            "examples": 0,
            "seconds": elapsed,
        }

    return {
        "loss": total_loss_sum / total_examples,
        "absolute": abs_loss_sum / total_examples,
        "relative": rel_loss_sum / total_examples,
        "examples": total_examples,
        "seconds": elapsed,
    }


def aggregate_metrics(metrics):
    #Prints out losses
    total_examples = sum(m["examples"] for m in metrics)

    if total_examples == 0:
        return {
            "loss": np.nan,
            "absolute": np.nan,
            "relative": np.nan,
            "examples": 0,
            "seconds": sum(m["seconds"] for m in metrics),
        }

    return {
        "loss": sum(m["loss"] * m["examples"] for m in metrics) / total_examples,
        "absolute": sum(m["absolute"] * m["examples"] for m in metrics) / total_examples,
        "relative": sum(m["relative"] * m["examples"] for m in metrics) / total_examples,
        "examples": total_examples,
        "seconds": sum(m["seconds"] for m in metrics),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--sharded_root", type=str, required=True)
    parser.add_argument("--model_store", type=str, default="model_store")

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.8)

    parser.add_argument("--lambda_relative", type=float, default=1.0)
    parser.add_argument("--max_norm", type=float, default=10.0)

    parser.add_argument("--in_dim", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--eval_every", type=int, default=1)

    # Test/debug controls.
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=100)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print("Using device:", device, flush=True)
    print("Args:", args, flush=True)

    run_dir = Path(args.model_store) / f"run_{args.index:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    shards = discover_shards(args.sharded_root, max_shards=args.max_shards)
    print("Found shards:", [s.name for s in shards], flush=True)

    metadata = build_combined_metadata(shards)

    # Drop any compound with at least one non-finite MMPBSA label.
    # For the 10k dataset this removes compound 2263, whose 100 pose labels are NaN.
    #Also removes labels that are extreme nonsensical values, such as compound 5854
    before_rows = len(metadata)
    before_compounds = metadata["compound_num"].nunique()

    metadata["mmpbsa"] = pd.to_numeric(metadata["mmpbsa"], errors="coerce")
    finite_label = np.isfinite(metadata["mmpbsa"].values)
    reasonable_label = metadata["mmpbsa"].abs().values <= 1000.0

    bad_label_mask = ~(finite_label & reasonable_label)
    bad_compounds = set(metadata.loc[bad_label_mask, "compound_num"].unique())
    #Remove bad compounds that are not full - <100 poses
    if bad_compounds:
        bad_rows = metadata.loc[
            metadata["compound_num"].isin(bad_compounds),
            ["source_shard", "local_graph_index", "compound_num", "mmpbsa"],
        ].copy()
        bad_rows.to_csv(run_dir / "dropped_bad_label_rows.csv", index=False)
    if bad_compounds:
        bad_file = run_dir / "dropped_bad_label_compounds.txt"
        with open(bad_file, "w") as f:
            for cid in sorted(bad_compounds):
                f.write(str(cid) + "\n")

        print("WARNING: dropping compounds with non-finite MMPBSA labels", flush=True)
        print("Bad compound count:", len(bad_compounds), flush=True)
        print("Bad compound list saved to:", bad_file, flush=True)

        metadata = metadata[~metadata["compound_num"].isin(bad_compounds)].copy()

    after_rows = len(metadata)
    after_compounds = metadata["compound_num"].nunique()

    print("Label filtering:", flush=True)
    print(f"  rows before={before_rows} after={after_rows} dropped={before_rows - after_rows}", flush=True)
    print(f"  compounds before={before_compounds} after={after_compounds} dropped={before_compounds - after_compounds}", flush=True)

    metadata, scaling = make_split_and_targets(metadata, args.train_frac, args.seed)

    metadata_file = run_dir / "sharded_metadata_with_targets.csv"
    scaling_file = run_dir / "residual_target_scaling.json"

    metadata.to_csv(metadata_file, index=False)
    with open(scaling_file, "w") as f:
        json.dump(scaling, f, indent=2)

    print("Metadata rows:", len(metadata), flush=True)
    print("Compounds:", metadata["compound_num"].nunique(), flush=True)
    print("Train rows:", int((metadata["split"] == "train").sum()), flush=True)
    print("Val rows:", int((metadata["split"] == "val").sum()), flush=True)
    print("Test rows:", int((metadata["split"] == "test").sum()), flush=True)
    print("Train compounds:", int(metadata.loc[metadata["split"] == "train", "compound_num"].nunique()), flush=True)
    print("Val compounds:", int(metadata.loc[metadata["split"] == "val", "compound_num"].nunique()), flush=True)
    print("Test compounds:", int(metadata.loc[metadata["split"] == "test", "compound_num"].nunique()), flush=True)
    print("Saved metadata:", metadata_file, flush=True)
    print("Saved scaling:", scaling_file, flush=True)

    #Defines model and optimiser
    model = EGNNRegressor(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_relative = float("inf")
    log_rows = []

    #Runs epochs
    for epoch in range(1, args.epochs + 1):
        print()
        print("=" * 100, flush=True)
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        print("=" * 100, flush=True)

        epoch_rng = random.Random(args.seed + epoch)
        shard_order = [s.name for s in shards]
        epoch_rng.shuffle(shard_order)

        train_metrics = []

        for shard_name in shard_order:
            shard_meta = metadata[
                (metadata["source_shard"] == shard_name) &
                (metadata["split"] == "train")
            ]

            if len(shard_meta) == 0:
                continue

            m = run_one_shard(
                model=model,
                optimizer=optimizer,
                shard_name=shard_name,
                shard_meta=shard_meta,
                batch_size=args.batch_size,
                device=device,
                lambda_relative=args.lambda_relative,
                max_norm=args.max_norm,
                train=True,
                epoch_seed=args.seed + epoch,
                max_batches=args.max_train_batches,
                log_every=args.log_every,
            )
            train_metrics.append(m)

            latest = {
                "epoch": epoch,
                "completed_train_shard": shard_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }
            torch.save(latest, run_dir / "latest_finite_after_shard.pt")
            torch.save(model.state_dict(), run_dir / "latest_finite_model_only.pt")
            print(f"Saved latest finite checkpoint after epoch={epoch} shard={shard_name}", flush=True)

        train_summary = aggregate_metrics(train_metrics)

        #Performs evaluation to select best epoch, if low_epochs <30, eval every epoch. 
        #If want to save time, increase to 2
        do_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if do_eval:
            val_metrics = []

            for shard_name in [s.name for s in shards]:
                shard_meta = metadata[
                    (metadata["source_shard"] == shard_name) &
                    (metadata["split"] == "val")
                ]

                if len(shard_meta) == 0:
                    continue

                m = run_one_shard(
                    model=model,
                    optimizer=None,
                    shard_name=shard_name,
                    shard_meta=shard_meta,
                    batch_size=args.batch_size,
                    device=device,
                    lambda_relative=args.lambda_relative,
                    max_norm=args.max_norm,
                    train=False,
                    epoch_seed=args.seed + epoch,
                    max_batches=args.max_val_batches,
                    log_every=args.log_every,
                )
                val_metrics.append(m)

            val_summary = aggregate_metrics(val_metrics)
        else:
            val_summary = {
                "loss": np.nan,
                "absolute": np.nan,
                "relative": np.nan,
                "examples": 0,
                "seconds": 0.0,
            }

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_summary['loss']:.6f} "
            f"train_abs={train_summary['absolute']:.6f} "
            f"train_rel={train_summary['relative']:.6f} "
            f"val_loss={val_summary['loss']:.6f} "
            f"val_abs={val_summary['absolute']:.6f} "
            f"val_rel={val_summary['relative']:.6f}",
            flush=True,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_summary["loss"],
            "train_absolute": train_summary["absolute"],
            "train_relative": train_summary["relative"],
            "train_examples": train_summary["examples"],
            "train_seconds": train_summary["seconds"],
            "val_loss": val_summary["loss"],
            "val_absolute": val_summary["absolute"],
            "val_relative": val_summary["relative"],
            "val_examples": val_summary["examples"],
            "val_seconds": val_summary["seconds"],
        }
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(run_dir / "training_log.csv", index=False)

        if do_eval and np.isfinite(val_summary["relative"]) and val_summary["relative"] < best_val_relative:
            best_val_relative = val_summary["relative"]
            torch.save(model.state_dict(), run_dir / "model.pt")
            print(f"Saved new best model: val_relative={best_val_relative:.6f}", flush=True)

    summary = {
        "run": args.index,
        "model_file": "EGNNThomas_batched_chem_edge_cutoff_gated.py",
        "training_script": "model_training_sharded_chem_edge_cut8_gated_normres.py",
        "sharded_root": args.sharded_root,
        "n_shards": len(shards),
        "n_rows": int(len(metadata)),
        "n_compounds": int(metadata["compound_num"].nunique()),
        "train_rows": int((metadata["split"] == "train").sum()),
        "val_rows": int((metadata["split"] == "val").sum()),
        "target": "normalised_frame_residual",
        "best_val_relative": best_val_relative,
        "args": vars(args),
        "scaling_file": str(scaling_file),
        "metadata_file": str(metadata_file),
    }

    with open(run_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Training complete.", flush=True)
    print("Run dir:", run_dir, flush=True)
    print("Best val relative:", best_val_relative, flush=True)


if __name__ == "__main__":
    main()
