#!/usr/bin/env python
import os
import subprocess
import argparse
from pathlib import Path

PROJECT_DIR = Path().cwd().resolve()
# TRAIN_INPUTS_PATH = PROJECT_DIR / "work" / "training_data"
TRAIN_INPUTS_PATH = PROJECT_DIR / "work" / "piper_scwrl/"
CLUSTER_BIN = "lib/cluster_piper_ft/cluster"

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query")
args = parser.parse_args()


def run_cluster(tag, cluster_out, ref_pdb, column, sortby, sorting_ord):
    n = 100
    cmd = [
        CLUSTER_BIN,
        "--piper",
        "ft.000.00_with_a_score",
        "--rots",
        "fft_rotset",
        "--rec",
        "rec.pdb",
        "--lig",
        "lig.pdb",
        "--ref_pdb",
        ref_pdb,
        "--c",
        column,
        "--o",
        cluster_out,
        "--pdb",
        "1",
        "--irmsd",
        "5",
        "--max",
        str(n),
        "--n",
        "2000",
        "--sortby",
        sortby,
        "--sorder",
        sorting_ord,
    ]

    try:
        proc = subprocess.run(cmd, check=True)
    except Exception as err:
        with open(f"{tag}_cluster_err", "w") as error_handle:
            error_handle.write(err)
    else:
        print(proc.args)
        for i in range(1, n + 1):
            os.rename(f"cluster_{i}.pdb", f"{tag}_cluster.000.0{i - 1}.pdb")


def main():
    query = args.query.strip()
    query_path = TRAIN_INPUTS_PATH / query
    work_dir = query_path / f"{query}_tmp"
    ref = f"../{query}.pdb"
    piper_col = "4"
    a_score_col = "10"
    
    os.chdir(work_dir)
    run_cluster("score_lower", "score_lower_000", ref, piper_col, "score", "lower")
    run_cluster(
        "cluster_lower", "cluster_lower_000", ref, piper_col, "cluster", "lower"
    )
    run_cluster("score_higher", "score_higher_000", ref, a_score_col, "score", "higher")
    run_cluster(
        "cluster_higher", "cluster_higher_000", ref, a_score_col, "cluster", "higher"
    )
    os.chdir(PROJECT_DIR)


if __name__ == "__main__":
    main()
