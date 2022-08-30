#!/usr/bin/env python
import os
from pathlib import Path
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils import prep_data, read_queries

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"

PROJECT_DIR = Path().cwd().resolve()
TRAIN_INPUTS_PATH = PROJECT_DIR / "work" / "training_data"
DATASET_IDS = PROJECT_DIR / "work" / "input_lists" / "train_test_161"
TRAIN_EVALS_PATH = PROJECT_DIR / "reports" / "tf_evals_161.csv"
SVD_MODEL_PATH = PROJECT_DIR / "models" / "svd_model_500"
REGRESSOR_PATH = PROJECT_DIR / "models" / "dockq_regressor.json"
MODEL_DIR_PATH = PROJECT_DIR / "models" / "tf"

# DATASET_IDS = PROJECT_DIR / "work" / "input_lists" / "train_test_5"
# TRAIN_EVALS_PATH = PROJECT_DIR / "reports" / "tf_evals_5.csv"


def prepare_inputs():
    all_ids = list(read_queries(DATASET_IDS))
    train_val_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=0.25, random_state=42
    )
    x_train, y_train = prep_data(
        train_ids, TRAIN_INPUTS_PATH, REGRESSOR_PATH, SVD_MODEL_PATH
    )
    x_val, y_val = prep_data(val_ids, TRAIN_INPUTS_PATH, REGRESSOR_PATH, SVD_MODEL_PATH)
    x_test, y_test = prep_data(
        test_ids, TRAIN_INPUTS_PATH, REGRESSOR_PATH, SVD_MODEL_PATH
    )

    return (x_train, y_train, x_val, y_val, x_test, y_test)


def build_model(normalizer):
    model = tf.keras.Sequential(
        [
            normalizer,
            tf.keras.layers.Dense(units=500, activation="relu"),
            tf.keras.layers.Dense(units=1, activation="sigmoid"),
        ]
    )

    roc_auc = tf.keras.metrics.AUC(name="roc-auc")
    pr_auc = tf.keras.metrics.AUC(name="pr-auc", curve="PR")

    sgd = tf.keras.optimizers.SGD(momentum=0.9, nesterov=True)

    model.compile(
        loss="binary_crossentropy",
        optimizer=sgd,
        metrics=["accuracy", roc_auc, pr_auc],
    )

    return model


def train_model(x_train, y_train, x_val, y_val):
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)
    norm = tf.keras.layers.experimental.preprocessing.Normalization()
    norm.adapt(x_train)

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=20)
    mc = tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR_PATH / "tf_model_161"),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
    )
    tb = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    model = build_model(norm)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=[es, mc, tb],
        verbose=1,
    )

    return history


def eval_model(model, x_test, y_test):
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
    return loss_and_metrics


def main():
    x_train, y_train, x_val, y_val, x_test, y_test = prepare_inputs()
    train_history = train_model(x_train, y_train, x_val, y_val)
    training_results = pd.DataFrame(train_history.history)
    training_results.to_csv(TRAIN_EVALS_PATH)

    trained_model = tf.keras.models.load_model(MODEL_DIR_PATH / "tf_model_161")
    loss_and_metrics = eval_model(trained_model, x_test, y_test)
    print("-------------------------------Test-------------------------------")
    print("Metrics: ", trained_model.metrics_names)
    print("         ", loss_and_metrics)


if __name__ == "__main__":
    main()
