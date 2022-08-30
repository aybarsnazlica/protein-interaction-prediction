#!/usr/bin/env python
import codecs
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost

from utils import read_queries

PROJECT_DIR = Path().cwd().resolve()
TRAIN_INPUTS_NATIVE = PROJECT_DIR / "work" / "training_data"
TRAIN_INPUTS_RANDROT = PROJECT_DIR / "work" / "training_data_randrot"
REGRESSOR_PATH = PROJECT_DIR / "models" / "dockq_regressor.json"
DATASET_IDS = PROJECT_DIR / "work" / "input_lists" / "train_test_161"


def read_poses(input_file: str, n_pose=70_001):
    targets = []
    reader = codecs.getreader("utf-8")
    archive = tarfile.open(input_file, "r:gz")
    f = reader(archive.extractfile(archive.getmembers()[0]))
    content = f.readlines()
    pose_count = 0
    for line in content:
        if line.startswith("POSE"):
            (
                header,
                index,
                ipose,
                irmsd,
                lig_rmsd,
                rec_prec,
                lig_prec,
                epiper,
                *feat_vec,
            ) = line.strip().split()

            mean_prec = np.mean((float(rec_prec) + float(lig_prec)))
            targets.append(np.array([float(irmsd), float(lig_rmsd), mean_prec]))
            pose_count += 1
            if pose_count > n_pose:
                break

    archive.close()

    return np.stack(targets)


def predict_target(target_feat, regressor_path: str):
    regressor = xgboost.XGBRegressor()
    regressor.load_model(regressor_path)
    target_scores = regressor.predict(target_feat)
    return target_scores


def binarize(target_scores):
    return np.where(target_scores >= 0.23, 1, 0)


def prep_data(queries: list, training_inputs_path: list, regressor_path: str):
    data = []

    for p in training_inputs_path:
        if p == TRAIN_INPUTS_NATIVE:
            query_type = "native"
        elif p == TRAIN_INPUTS_RANDROT:
            query_type = "randrot"

        for query in queries:
            input_file = Path(p) / query / "poses.tar.gz"

            try:
                target_feat = read_poses(input_file)
            except Exception as err:
                print(f"Error: {err} in preparing {query}")
            else:
                target_scores = predict_target(target_feat, regressor_path)
                binary_target = binarize(target_scores)
                number_of_true_models = np.count_nonzero(binary_target)
                data.append([f"{query}_{query_type}", number_of_true_models])

    return data


def main():
    model_data = prep_data(
        list(read_queries(DATASET_IDS)),
        [TRAIN_INPUTS_NATIVE, TRAIN_INPUTS_RANDROT],
        REGRESSOR_PATH,
    )
    df = pd.DataFrame(model_data, columns=["query", "number_of_true_models"])
    df.to_csv(PROJECT_DIR / "work" / "pose_true_false.csv")


if __name__ == "__main__":
    main()
