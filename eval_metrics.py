'''
Computes regression, regret/rank and IMPECCABLE-style Top-p recovery metrics from the EGNN scores CSV.
'''
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def pick_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f"Could not find {label}. Tried {candidates}. Columns: {list(df.columns)}")


def corr(a, b, method):
    a = pd.Series(a)
    b = pd.Series(b)
    if len(a) < 2:
        return None
    if float(a.std(ddof=0)) == 0.0 or float(b.std(ddof=0)) == 0.0:
        return None
    return float(a.corr(b, method=method))


def regression_metrics(df, true_col, pred_col, compound_col):
    '''
    Computes regression metrics for the csv
    '''
    y = df[true_col].to_numpy(float)
    p = df[pred_col].to_numpy(float)
    err = p - y

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    return {
        "rows": int(len(df)),
        "compounds": int(df[compound_col].nunique()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "pearson": corr(y, p, "pearson"),
        "spearman": corr(y, p, "spearman"),
        "true_std": float(np.std(y, ddof=0)),
        "pred_std": float(np.std(p, ddof=0)),
    }


def selection_metrics(df, true_col, pred_col, compound_col, p_values=(1, 3, 5)):
    '''
    Computes selection metrics, mainly IMPECCABLE's top-p
    '''
    df = df.copy()
    df["_row_id"] = np.arange(len(df))

    max_m = int(df.groupby(compound_col).size().max())
    hit_counts = {p: np.zeros(max_m, dtype=int) for p in p_values}

    regrets = []
    selected_ranks = []
    nposes = []
    n_compounds = 0

    for cid, g in df.groupby(compound_col, sort=False):
        g = g.dropna(subset=[true_col, pred_col]).copy()
        if len(g) == 0:
            continue

        # Lower MMPBSA and lower predicted score are better.
        g_true = g.sort_values(true_col, ascending=True)
        true_ids = list(g_true["_row_id"])
        true_rank = {rid: rank + 1 for rank, rid in enumerate(true_ids)}
        true_best = float(g_true[true_col].iloc[0])

        g_pred = g.sort_values(pred_col, ascending=True)
        pred_ids = list(g_pred["_row_id"])

        selected = g_pred.iloc[0]
        regrets.append(float(selected[true_col] - true_best))
        selected_ranks.append(int(true_rank[selected["_row_id"]]))
        nposes.append(int(len(g)))
        n_compounds += 1

        for p in p_values:
            pp = min(p, len(g))
            true_top_p = set(true_ids[:pp])

            for m in range(1, len(g) + 1):
                if true_top_p.intersection(pred_ids[:m]):
                    hit_counts[p][m - 1:] += 1
                    break

    out = {
        "n_compounds": int(n_compounds),
        "mean_n_poses": float(np.mean(nposes)),
        "min_n_poses": int(np.min(nposes)),
        "max_n_poses": int(np.max(nposes)),
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "mean_selected_rank": float(np.mean(selected_ranks)),
        "median_selected_rank": float(np.median(selected_ranks)),
    }

    for p in p_values:
        frac = hit_counts[p] / n_compounds
        out[f"p{p}_m90"] = int(np.where(frac >= 0.90)[0][0] + 1) if np.any(frac >= 0.90) else None

        for k in [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 50, 75, 100]:
            if k <= max_m:
                out[f"p{p}_hit_at_{k}"] = float(frac[k - 1])

    return out


def evaluate_method(df, true_col, pred_col, compound_col):
    '''
    Evalutes the selected method by looking at the difference in the actual column and the predicted values
    Computes both regresion and selection metrics
    '''
    return {
        "regression": regression_metrics(df, true_col, pred_col, compound_col),
        "selection": selection_metrics(df, true_col, pred_col, compound_col),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    '''
    Parses out the appropriate csv, making a new one containing the stats
    '''
    scores_csv = Path(args.scores_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading:", scores_csv)
    df = pd.read_csv(scores_csv)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    compound_col = pick_col(df, ["compound_num", "compound_id", "compound"], "compound column")
    true_col = pick_col(df, ["mmpbsa", "true_mmpbsa", "raw_mmpbsa", "y_true"], "true MMPBSA column")
    pred_col = pick_col(df, ["pred_score", "pred_mmpbsa", "y_pred"], "EGNN raw prediction column")

    '''
    Creates different residual target columns:
        Residual True: Creates actual residual target
        Residual Prediction: Creates residua prediction column
        Frame Template: The average of each frame
        Lower Frame: The results of always picking the lower frame in ranking
        
    '''
    residual_true_col = pick_col(df, ["residual_target"], "residual target column")
    residual_pred_col = pick_col(df, ["pred_residual"], "residual prediction column")
    frame_template_col = pick_col(df, ["frame_template_baseline_score", "frame_template_score"], "frame-template column")
    lower_frame_col = pick_col(df, ["lower_frame_score"], "lower-frame column")

    if "split" in df.columns:
        print("\nRows by split:")
        print(df["split"].value_counts().sort_index())
        print("\nCompounds by split:")
        print(df.groupby("split")[compound_col].nunique().sort_index())

    print("\nPose counts per compound:")
    print(df.groupby(compound_col).size().describe())

    results = {
        #Computes the metrics for each result type
        "egnn_raw_vs_raw_mmpbsa": evaluate_method(df, true_col, pred_col, compound_col),
        "egnn_residual_vs_residual_target": evaluate_method(df, residual_true_col, residual_pred_col, compound_col),
        "egnn_residual_vs_raw_mmpbsa": evaluate_method(df, true_col, residual_pred_col, compound_col),
        "frame_template_vs_raw_mmpbsa": evaluate_method(df, true_col, frame_template_col, compound_col),
        "lower_frame_vs_raw_mmpbsa": {
            "selection": selection_metrics(df, true_col, lower_frame_col, compound_col)
        },
    }

    out_json = out_dir / "run038_compound_holdout_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    compact_rows = []
    for method, block in results.items():
        row = {"method": method}

        if "regression" in block:
            for k, v in block["regression"].items():
                row[f"reg_{k}"] = v

        if "selection" in block:
            for k, v in block["selection"].items():
                row[f"sel_{k}"] = v

        compact_rows.append(row)

    out_csv = out_dir / "run038_compound_holdout_metrics_compact.csv"
    pd.DataFrame(compact_rows).to_csv(out_csv, index=False)

    print("\nWrote:")
    print(out_json)
    print(out_csv)

    print("\n" + "=" * 100)
    print("RUN038 COMPOUND-LEVEL HOLDOUT TEST SUMMARY")
    print("=" * 100)

    main_block = results["egnn_raw_vs_raw_mmpbsa"]
    r = main_block["regression"]
    s = main_block["selection"]

    #Prints main metrics
    print("\nEGNN raw score vs raw MMPBSA")
    print("Rows:", r["rows"])
    print("Compounds:", r["compounds"])
    print("Top-1/Top-3/Top-5:", s["p1_m90"], s["p3_m90"], s["p5_m90"])
    print("Mean regret:", s["mean_regret"])
    print("Median regret:", s["median_regret"])
    print("Mean selected rank:", s["mean_selected_rank"])
    print("Median selected rank:", s["median_selected_rank"])
    print("MAE:", r["mae"])
    print("RMSE:", r["rmse"])
    print("R2:", r["r2"])
    print("Pearson:", r["pearson"])
    print("Spearman:", r["spearman"])
    print("True std:", r["true_std"])
    print("Pred std:", r["pred_std"])

    print("\nFrame/template baselines")
    for method in ["frame_template_vs_raw_mmpbsa", "lower_frame_vs_raw_mmpbsa"]:
        sel = results[method]["selection"]
        print(method)
        print("  Top-1/Top-3/Top-5:", sel["p1_m90"], sel["p3_m90"], sel["p5_m90"])
        print("  Mean regret:", sel["mean_regret"])
        print("  Mean selected rank:", sel["mean_selected_rank"])

    print("\nDeframed EGNN")
    for method in ["egnn_residual_vs_residual_target", "egnn_residual_vs_raw_mmpbsa"]:
        block = results[method]
        reg = block["regression"]
        sel = block["selection"]
        print(method)
        print("  Top-1/Top-3/Top-5:", sel["p1_m90"], sel["p3_m90"], sel["p5_m90"])
        print("  Mean regret:", sel["mean_regret"])
        print("  Mean rank:", sel["mean_selected_rank"])
        print("  R2:", reg["r2"])
        print("  RMSE:", reg["rmse"])


if __name__ == "__main__":
    main()
