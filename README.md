# IMPECCABLE EGNN Pose Filtering

This directory contains the EGNN code developed for the dissertation project. The aim of the project was to replace the fixed histogram-based surrogate used in the IMPECCABLE workflow with an atom-level equivariant graph neural network for pose filtering.

The repository is not a standalone version of the full IMPECCABLE pipeline. It contains the code written for graph construction, EGNN training, EGNN evaluation and comparison against the AMPL/IMPECCABLE histogram baseline. The original docking inputs, generated graph shards, trained checkpoints and wider IMPECCABLE pipeline are not included.

This repository will not run by itself, but is an indication of the code used and written by me within the dissertation.
## Main files

### Graph generation

- `feature_generation_general.py`  
  Main graph-generation script. It converts IMPECCABLE docking outputs into atom-level protein-ligand graphs.

- `feature_generation_graph.py`  
  Helper code used during graph construction.

- `create_graphs.pbs`  
  Aurora PBS script used to generate the 10k sharded EGNN graph dataset.

### EGNN model and training

- `EGNN.py`  
  Final EGNN architecture. The model uses atom-level node features, protein-ligand contact edges and gated/contact-edge readout.

- `egnn_data_utils.py`  
  Manual batching utilities for combining graph objects without relying on PyTorch Geometric batching.

- `model_training.py`  
  Final EGNN training script. This performs the compound-level train/validation/test split, constructs the frame-residual target, trains the EGNN and saves the best checkpoint according to validation relative loss.

- `train_run38.pbs`  
  PBS script recreating the final run038 EGNN training job.

### Evaluation

- `evaluate_egnn.py`  
  Loads a trained EGNN checkpoint, scores graph shards and reconstructs predicted raw MMPBSA scores from the frame-residual target.

- `eval_metrics.py`  
  Computes regression, regret/rank and IMPECCABLE-style Top-p recovery metrics from the EGNN scores CSV.

- `eval_run38.pbs`  
  PBS script used to evaluate the final run038 checkpoint on the compound-level test split.

### Plotting

- `plot_run038_training_graphs.py`  
  Parses the run038 training log and produces the training/validation loss curves used in the dissertation.

## External data

The full dataset is not included because it is too large and depends on the wider IMPECCABLE workflow.

The final graph dataset used in the dissertation was generated on Aurora from:

/lus/flare/projects/CompBioAffin/tomtom147/real_labelled_10k_inputs_sharded
and written to:

/lus/flare/projects/CompBioAffin/tomtom147/real_labelled_10k_res6_inter8_sharded

The final EGNN run output was:

model_store/run_038

Important generated files included:

model.pt
sharded_metadata_with_targets.csv
residual_target_scaling.json
training_log.csv
eval_best_test_compound_holdout_scores.csv

These generated files are not included in this repository.


Final run038 settings

The final EGNN experiment used:

index = 38
epochs = 20
batch_size = 8
learning_rate = 3e-4
weight_decay = 0.0
seed = 42
train_frac = 0.7
lambda_relative = 5.0
max_norm = 10.0
hidden_dim = 128
num_layers = 4
dropout = 0.0
edge cutoff = 8.0 Å
residue cutoff = 6.0 Å

The split was compound-level 70/15/15. After label filtering, the dataset contained:

993800 poses
9938 compounds
695600 train rows
149100 validation rows
149100 test rows