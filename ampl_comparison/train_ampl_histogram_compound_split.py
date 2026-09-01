'''
Trains the AMPL histogram neural network baseline using the ordered compound-split CSV.
This script is adapted from the original IMPECCABLE/AMPL training code. The main project-specific change is the use of the same ordered compound-level split as the EGNN run038 experiment, allowing a matched comparison.

'''
import argparse
import json

import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.parameter_parser as parse


def get_model_training_params(args):
    '''
    Loads in model training parameters
    '''
    with open("training_params.json", "r") as f:
        full_parameters = json.load(f)

    return full_parameters[str(args.index)]


def train(args):
    variable_params = get_model_training_params(args)
    bs, ls, lr, wd = variable_params
    #Performs run038 compound-level split:
    # train compounds: 6956 -> 695600 rows
    # val compounds:   1491 -> 149100 rows
    # test compounds:  1491 -> 149100 rows
    # total:           9938 -> 993800 rows
    valid_frac = 149100 / 993800
    test_frac = 149100 / 993800

    params = {
        "dataset_key": "./featurized_data_file_run038_compound_split_ordered.csv",
        "datastore": False,


        "splitter": "index",
        "split_valid_frac": str(valid_frac),
        "split_test_frac": str(test_frac),
        "split_strategy": "train_valid_test",

        "prediction_type": "regression",
        "response_cols": "mmpbsa",
        "id_col": "uid",
        "smiles_col": "smiles",

       
        "result_dir": "./model_store_ampl_run038_compoundsplit",

        "model_type": "NN",
        "featurizer": "computed_descriptors",
        "descriptor_type": "bin_cont_scores_24",
        "previously_featurized": True,

        "max_epochs": 40,
        "weight_decay_penalty_type": wd,
        "learning_rate": lr,
        "layer_sizes": ",".join(str(x) for x in ls),
        "batch_size": bs,
    }

    print("AMPL run038 compound-split params:")
    for k, v in params.items():
        print(f"{k}: {v}")

    pparams = parse.wrapper(params)
    MP = mp.ModelPipeline(pparams)
    MP.train_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=int)
    args = parser.parse_args()

    train(args)
