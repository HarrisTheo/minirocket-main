import argparse
import time
import numpy as np
import pandas as pd

from minirocket import fit, transform
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler


# ==========================================================
# SINGLE RUN (PURE MINIROCKET)
# ==========================================================
def run_minirocket_once(training_data, test_data):

    y_train = training_data[:, 0].astype(int)
    X_train = training_data[:, 1:].astype(np.float32)

    y_test = test_data[:, 0].astype(int)
    X_test = test_data[:, 1:].astype(np.float32)

    # --------------------------------------------------
    # MiniRocket Kernel Fit
    # --------------------------------------------------
    t0 = time.perf_counter()
    parameters = fit(X_train)
    t1 = time.perf_counter()

    # --------------------------------------------------
    # Transform Training Data
    # --------------------------------------------------
    X_train_t = transform(X_train, parameters)
    t2 = time.perf_counter()

    # --------------------------------------------------
    # Scale Features
    # --------------------------------------------------
    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_train_t)
    X_train_t = scaler.transform(X_train_t)

    # --------------------------------------------------
    # Train Classifier (STATIC)
    # --------------------------------------------------
    clf = RidgeClassifierCV(alphas=10 ** np.linspace(-3, 3, 10))
    clf.fit(X_train_t, y_train)
    t3 = time.perf_counter()

    # --------------------------------------------------
    # Transform Test Data
    # --------------------------------------------------
    X_test_t = transform(X_test, parameters)
    X_test_t = scaler.transform(X_test_t)
    t4 = time.perf_counter()

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    acc = clf.score(X_test_t, y_test)
    t5 = time.perf_counter()

    timings = np.array([
        t1 - t0,  # MiniRocket fit
        t2 - t1,  # Transform train
        t3 - t2,  # Classifier training
        t4 - t3,  # Transform test
        t5 - t4   # Evaluation
    ], dtype=np.float64)

    return acc, timings


# ==========================================================
# MULTIPLE RUNS
# ==========================================================
def run_minirocket(training_data, test_data, num_runs=10):

    results = np.zeros(num_runs, dtype=np.float64)
    timings = np.zeros((5, num_runs), dtype=np.float64)

    for i in range(num_runs):
        acc, t = run_minirocket_once(training_data, test_data)
        results[i] = acc
        timings[:, i] = t

    return results, timings


# ==========================================================
# MAIN SCRIPT
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path", required=True)
parser.add_argument("-o", "--output_path", required=True)
parser.add_argument("-n", "--num_runs", type=int, default=10)
args = parser.parse_args()

dataset_names_additional = ("SPY", "ACSF1")

results_additional = pd.DataFrame(
    index=dataset_names_additional,
    columns=[
        "accuracy_mean",
        "accuracy_standard_deviation",
        "time_training_seconds",
        "time_test_seconds",
    ],
    data=0.0,
)

results_additional.index.name = "dataset"

print("RUNNING".center(80, "="))

for dataset_name in dataset_names_additional:

    print(dataset_name.center(80, "-"))

    print("Loading data".ljust(75, "."), end="", flush=True)

    if dataset_name != "PLAID":
        training_data = np.loadtxt(
            f"{args.input_path}/{dataset_name}/{dataset_name}_TRAIN.txt"
        )
        test_data = np.loadtxt(
            f"{args.input_path}/{dataset_name}/{dataset_name}_TEST.txt"
        )
    else:
        training_data = np.loadtxt(
            f"{args.input_path}/{dataset_name}/{dataset_name}_TRAIN.txt",
            delimiter=","
        )
        test_data = np.loadtxt(
            f"{args.input_path}/{dataset_name}/{dataset_name}_TEST.txt",
            delimiter=","
        )

    print("Done.")

    print("Performing runs".ljust(75, "."), end="", flush=True)

    results, timings = run_minirocket(
        training_data,
        test_data,
        num_runs=args.num_runs
    )

    timings_mean = timings.mean(axis=1)

    print("Done.")

    results_additional.loc[dataset_name, "accuracy_mean"] = results.mean()
    results_additional.loc[dataset_name, "accuracy_standard_deviation"] = results.std(ddof=0)

    results_additional.loc[dataset_name, "time_training_seconds"] = (
        timings_mean[0] + timings_mean[1] + timings_mean[2]
    )

    results_additional.loc[dataset_name, "time_test_seconds"] = (
        timings_mean[3] + timings_mean[4]
    )

print("FINISHED".center(80, "="))
results_additional.to_csv(f"{args.output_path}results_minirocket_baseline.csv")