
## `ampl_comparison/README.md`

# AMPL Histogram Baseline Comparison

This directory contains the scripts used to reproduce the matched AMPL/IMPECCABLE histogram baseline for comparison with the EGNN.

The AMPL baseline is not part of the EGNN model. It was rerun so that the histogram baseline and EGNN were evaluated on the same compound-level train/validation/test split.

## Files

- `make_ampl_compoundsplit_csv.pbs`  
  Creates the ordered AMPL input CSV. The rows are ordered as train, validation and test so that AMPL index splitting reproduces the EGNN compound-level split.

- `train_ampl_histogram_compound_split.py`  
  Trains the AMPL histogram neural network baseline using the ordered compound-split CSV.

- `run_ampl_run038_compoundsplit_index0.pbs`  
  PBS wrapper used to train the AMPL baseline.

- `ampl_predict_and_score_run038_compoundsplit_reload.py`  
  Reloads the trained AMPL model, predicts on the held-out test split and computes regression and pose-selection metrics.

- `run_ampl_run038_compoundsplit_predict.pbs`  
  PBS wrapper used for AMPL prediction and scoring.

- `training_params.json`  
  Hyperparameters used by the AMPL training script.

## Notes

These scripts depend on the AMPL/atomsci environment and the featurized histogram CSV generated from the wider IMPECCABLE pipeline. The trained AMPL model and generated prediction files are not included.