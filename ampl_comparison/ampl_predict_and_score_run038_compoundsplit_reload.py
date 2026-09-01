'''
Reloads the trained AMPL model, predicts on the held-out test split and computes regression and pose-selection metrics.
'''
from pathlib import Path
import json
import numpy as np
import pandas as pd

from atomsci.ddm.pipeline import parameter_parser as parse
from atomsci.ddm.pipeline import model_pipeline as mp

RUN_UUID = "59ed2527-b9f6-4a59-bfff-b4907aca87f7"

DATA_CSV = Path(
    "/lus/flare/projects/CompBioAffin/tomtom147/impeccable_hist_10k/"
    "featurized_data_file_run038_compound_split_ordered.csv"
)

RUN_DIR = Path(
    "/lus/flare/projects/CompBioAffin/tomtom147/ampl_run038_compoundsplit_job/"
    "model_store_ampl_run038_compoundsplit/"
    "featurized_data_file_run038_compound_split_ordered/"
    "NN_computed_descriptors_index_regression"
) / RUN_UUID

OUT_DIR = RUN_DIR / "posthoc_test_predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TOTAL = 993800
N_TRAIN = 695600
N_VALID = 149100
N_TEST = 149100
TEST_START = N_TRAIN + N_VALID

print("RUN_UUID:", RUN_UUID)
print("DATA_CSV:", DATA_CSV)
print("RUN_DIR:", RUN_DIR)
print("OUT_DIR:", OUT_DIR)

assert DATA_CSV.exists(), DATA_CSV
assert RUN_DIR.exists(), RUN_DIR
assert (RUN_DIR / "model_metadata.json").exists()
assert (RUN_DIR / "transformers.pkl").exists()
assert (RUN_DIR / "best_model" / "checkpoint1.pt").exists()

print("Loading TEST split only from ordered CSV...")
#Start from 1 as header is row 0
df = pd.read_csv(DATA_CSV, skiprows=range(1, TEST_START + 1))

print("test rows:", len(df))
print("test compounds:", df["compound_num"].nunique())
print("mean poses per compound:", df.groupby("compound_num").size().mean())

assert len(df) == N_TEST, f"Expected {N_TEST} test rows, got {len(df)}"
assert df["compound_num"].nunique() == 1491, "Expected 1491 test compounds"

pred_params_dict = {
    "dataset_key": str(DATA_CSV),
    "id_col": "uid",
    "smiles_col": "smiles",
    "response_cols": "mmpbsa",
    "result_dir": str(OUT_DIR / "prediction_tmp"),
    "model_type": "NN",
    "prediction_type": "regression",
    "featurizer": "computed_descriptors",
    "descriptor_type": "bin_cont_scores_24",
    "previously_featurized": True,
    "feature_transform_type": "normalization",
    "response_transform_type": "normalization",
    "transformer_key": str(RUN_DIR / "transformers.pkl"),
}

print("Creating prediction pipeline from local reload dir...")
#Create pipe used for predictions
pred_params = parse.wrapper(pred_params_dict)
pipe = mp.create_prediction_pipeline_from_file(
    pred_params,
    reload_dir=str(RUN_DIR),
    model_type="best_model",
)

print("Running predictions on TEST split...")
pred_df = pipe.predict_full_dataset(
    df,
    is_featurized=True,
    contains_responses=True,
    AD_method=None,
)

print("Prediction shape:", pred_df.shape)
print("Prediction columns:")
print(list(pred_df.columns))

#Print csv to specified path
raw_pred_path = OUT_DIR / "ampl_run038_compoundsplit_test_predictions_raw.csv"
pred_df.to_csv(raw_pred_path, index=False)
print("Wrote:", raw_pred_path)

pred_col = None
for c in ["mmpbsa_pred", "pred", "prediction", "mmpbsa"]:
    if c in pred_df.columns and c != "mmpbsa":
        pred_col = c
        break

if pred_col is None:
    candidates = [c for c in pred_df.columns if "pred" in c.lower()]
    if len(candidates) == 1:
        pred_col = candidates[0]

if pred_col is None:
    raise RuntimeError(f"No prediction column found. Columns: {list(pred_df.columns)}")

print("Using prediction column:", pred_col)

df2 = df.copy()
df2["uid"] = df2["uid"].astype(str)
pred_df["uid"] = pred_df["uid"].astype(str)

if set(pred_df["uid"]).issubset(set(df2["uid"])):
    score_df = df2.merge(
        pred_df[["uid", pred_col]],
        on="uid",
        how="inner",
    )
    if len(score_df) != len(pred_df):
        raise RuntimeError(f"Merge row mismatch: merged={len(score_df)} pred={len(pred_df)}")
else:
    print("WARNING: prediction uid values do not match original uid values.")
    print("Falling back to row-order alignment.")
    if len(pred_df) != len(df2):
        raise RuntimeError(f"Cannot row-align: pred={len(pred_df)} original={len(df2)}")
    score_df = df2.copy()
    score_df[pred_col] = pred_df[pred_col].to_numpy()


#Rename and store the scoring column, ensuring all predicted values are finite
score_df = score_df.rename(columns={pred_col: "ampl_pred"})
score_df["true_mmpbsa"] = score_df["mmpbsa"]

score_df = score_df[
    np.isfinite(score_df["true_mmpbsa"]) &
    np.isfinite(score_df["ampl_pred"])
].copy()

print("Scoring rows:", len(score_df))
print("Scoring compounds:", score_df["compound_num"].nunique())

score_path = OUT_DIR / "ampl_run038_compoundsplit_test_scores.csv"
score_df.to_csv(score_path, index=False)
print("Wrote:", score_path)


def regression_metrics(frame, true_col="true_mmpbsa", pred_col="ampl_pred"):
    '''
    Computes regression metrics
    '''
    y_true = frame[true_col].to_numpy(dtype=float)
    y_pred = frame[pred_col].to_numpy(dtype=float)

    err = y_pred - y_true
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return {
        "rows": int(len(y_true)),
        "compounds": int(frame["compound_num"].nunique()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1.0 - ss_res / ss_tot),
        "pearson": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "spearman": float(frame[[true_col, pred_col]].corr(method="spearman").iloc[0, 1]),
        "true_std": float(np.std(y_true, ddof=0)),
        "pred_std": float(np.std(y_pred, ddof=0)),
    }


def selection_metrics(frame, pred_col="ampl_pred", true_col="true_mmpbsa"):
    '''
    Compute selection metrics
    '''
    n_compounds = frame["compound_num"].nunique()

    regrets = []
    selected_ranks = []
    hit_counts = {
        1: np.zeros(100, dtype=int),
        3: np.zeros(100, dtype=int),
        5: np.zeros(100, dtype=int),
    }

    for cid, g in frame.groupby("compound_num", sort=False):
        if len(g) != 100:
            raise RuntimeError(f"Compound {cid} has {len(g)} poses, expected 100")

        g_true = g.sort_values(true_col, ascending=True)
        true_uids = list(g_true["uid"].astype(str))
        true_rank = {uid: i + 1 for i, uid in enumerate(true_uids)}
        true_best = float(g_true[true_col].iloc[0])

        g_pred = g.sort_values(pred_col, ascending=True)
        pred_uids = list(g_pred["uid"].astype(str))

        selected = g_pred.iloc[0]
        selected_uid = str(selected["uid"])

        regrets.append(float(selected[true_col] - true_best))
        selected_ranks.append(int(true_rank[selected_uid]))

        for p in [1, 3, 5]:
            true_top = set(true_uids[:p])
            for m in range(1, 101):
                if true_top.intersection(pred_uids[:m]):
                    hit_counts[p][m - 1:] += 1
                    break

    out = {
        "n_compounds": int(n_compounds),
        "mean_n_poses": float(frame.groupby("compound_num").size().mean()),
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "mean_selected_rank": float(np.mean(selected_ranks)),
        "median_selected_rank": float(np.median(selected_ranks)),
    }

    #Calculate top-p counts for at least 90% of compounds
    for p in [1, 3, 5]:
        frac = hit_counts[p] / n_compounds
        out[f"p{p}_m90"] = int(np.where(frac >= 0.90)[0][0] + 1) if np.any(frac >= 0.90) else None
        out[f"p{p}_hit_at_1"] = float(frac[0])
        out[f"p{p}_hit_at_5"] = float(frac[4])
        out[f"p{p}_hit_at_10"] = float(frac[9])
        out[f"p{p}_hit_at_20"] = float(frac[19])

    return out


reg = regression_metrics(score_df)
sel = selection_metrics(score_df)

summary = {
    "run_uuid": RUN_UUID,
    "run_dir": str(RUN_DIR),
    "data_csv": str(DATA_CSV),
    "split": "test_only_last_149100_rows_of_ordered_compound_split_csv",
    "raw_prediction_file": str(raw_pred_path),
    "score_file": str(score_path),
    "regression": reg,
    "selection_all_100": sel,
}

#Prints and saves summary
summary_path = OUT_DIR / "ampl_run038_compoundsplit_test_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

print("\nREGRESSION")
print(json.dumps(reg, indent=2))

print("\nALL-100 TEST SELECTION")
print(json.dumps(sel, indent=2))

print("\nWrote summary:", summary_path)