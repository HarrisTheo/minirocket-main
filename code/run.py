import argparse
import time
import os
import numpy as np
import pandas as pd

from minirocket import fit, transform
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# ==========================================================
# AEN PLASTICITY REGULATOR (NO Q-VALUES)
# ==========================================================
class AENRegulator:
    
    def __init__(self):
        self.ego = 0.5
        self.c_drive = 0.5
        self.alpha_ego = 0.15
        self.alpha_c = 0.10

    def update(self, error):
        surprise = abs(error)

        # Ego calibration
        self.ego = (1 - self.alpha_ego) * self.ego + \
                   self.alpha_ego * (1.0 - surprise)

        # Regulatory layer
        target_c = 1.0 - self.ego

        if self.ego > 0.85:
            target_c *= 0.3  # Deep settling when stable

        self.c_drive = (1 - self.alpha_c) * self.c_drive + \
                       self.alpha_c * target_c

    def scale(self):
        # Non-linear modulation
        return 1 + 4 * (self.c_drive ** 2)

# ==========================================================
# TSV LOADER
# ==========================================================
def load_tsv(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep="\t", header=None)
    
    return df.to_numpy()

# ==========================================================
# DATASET DISCOVERY (TSV VERSION)
# ==========================================================
def discover_ucr2018_tsv_datasets(input_path: str):
    dataset_names = []

    for name in sorted(os.listdir(input_path)):
        ds_dir = os.path.join(input_path, name)
        
        if not os.path.isdir(ds_dir):
            continue

        train_path = os.path.join(ds_dir, f"{name}_TRAIN.tsv")
        test_path  = os.path.join(ds_dir, f"{name}_TEST.tsv")

        if os.path.isfile(train_path) and os.path.isfile(test_path):
            dataset_names.append(name)

    return tuple(dataset_names)

# ==========================================================
# SINGLE RUN
# ==========================================================
def run_minirocket_once(training_data, test_data):
    y_train = training_data[:, 0].astype(int)
    X_train = training_data[:, 1:].astype(np.float32)

    y_test = test_data[:, 0].astype(int)
    X_test = test_data[:, 1:].astype(np.float32)

    # --------------------------------------------------
    # MiniRocket Fit
    # --------------------------------------------------
    t0 = time.perf_counter()
    parameters = fit(X_train)
    t1 = time.perf_counter()

    X_train_t = transform(X_train, parameters)
    t2 = time.perf_counter()

    # --------------------------------------------------
    # Feature Scaling
    # --------------------------------------------------
    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_train_t)
    X_train_t = scaler.transform(X_train_t)

    # --------------------------------------------------
    # Online Classifier (instead of Ridge)
    # --------------------------------------------------
    clf = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.01
    )

    clf.fit(X_train_t, y_train)
    t3 = time.perf_counter()

    # --------------------------------------------------
    # Transform Test Data
    # --------------------------------------------------
    X_test_t = transform(X_test, parameters)
    X_test_t = scaler.transform(X_test_t)
    t4 = time.perf_counter()

    # --------------------------------------------------
    # Streaming + AEN Plasticity Regulation
    # --------------------------------------------------
    regulator = AENRegulator()
    preds = []

    base_lr = 0.01

    for i in range(len(X_test_t)):
        x = X_test_t[i:i+1]
        y_true = y_test[i]

        y_pred = clf.predict(x)[0]
        preds.append(y_pred)

        error = int(y_pred != y_true)
        regulator.update(error)

        cap = 0.01 / np.sqrt(i + 1)
        clf.eta0 = min(base_lr * regulator.scale(), cap)

        clf.partial_fit(x, [y_true])

    acc = np.mean(np.array(preds) == y_test)
    t5 = time.perf_counter()

    timings = np.array([
        t1 - t0,  # MiniRocket fit
        t2 - t1,  # Transform train
        t3 - t2,  # Initial classifier fit
        t4 - t3,  # Transform test
        t5 - t4   # Streaming + AEN
    ], dtype=np.float64)

    return acc, timings

# ==========================================================
# MULTIPLE RUNS
# ==========================================================
def run_minirocket(training_data, test_data, num_runs=1):
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
parser.add_argument("-k", "--num_kernels", type=int, default=10_000)  # kept for compatibility
args = parser.parse_args()

dataset_names_additional = (
    "SPY","AllGestureWiimoteX","AllGestureWiimoteY","AllGestureWiimoteZ","BME",
    "Chinatown","Crop","DodgerLoopDay","DodgerLoopGame","DodgerLoopWeekend",
    "EOGHorizontalSignal","EOGVerticalSignal","EthanolLevel","FreezerRegularTrain",
    "FreezerSmallTrain", "GesturePebbleZ1","GesturePebbleZ2","GunPointAgeSpan","GunPointMaleVersusFemale",
    "GunPointOldVersusYoung","HouseTwenty","InsectEPGRegularTrain","InsectEPGSmallTrain",
    "MelbournePedestrian","MixedShapesRegularTrain","MixedShapesSmallTrain","PLAID",
    "PowerCons", "Rock","SemgHandGenderCh2", "SemgHandMovementCh2","SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ","SmoothSubspace"
) #discover_ucr2018_tsv_datasets(args.input_path)

print(f"Found {len(dataset_names_additional)} TSV datasets in: {args.input_path}")

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

    training_data = load_tsv(
        f"{args.input_path}/{dataset_name}/{dataset_name}_TRAIN.tsv"
    )
    test_data = load_tsv(
        f"{args.input_path}/{dataset_name}/{dataset_name}_TEST.tsv"
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

out_file = args.output_path
if os.path.isdir(out_file) or out_file.endswith("/") or out_file.endswith("\\"):
    out_file = os.path.join(out_file, "results_additional_aen_tsv.csv")

results_additional.to_csv(out_file, index=True)
print(f"Saved: {out_file}")