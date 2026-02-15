import argparse
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import LinearSVC
from minirocket import fit, transform
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ==========================================================
# AEN PLASTICITY REGULATOR (NO Q-VALUES)
# ==========================================================
class AENRegulator:
    """
    AEN used purely as plasticity controller.
    It does NOT choose actions.
    It only regulates adaptation pressure.
    """

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

    clf.partial_fit(X_train_t, y_train, classes=np.unique(y_train))
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

        # Update AEN regulator
        regulator.update(error)

        # Modulate learning rate
        clf.eta0 = base_lr * regulator.scale()

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
parser.add_argument("-k", "--num_kernels", type=int, default=10_000)
args = parser.parse_args()

"""
dataset_names_additional = (
    "SPY","ACSF1","AllGestureWiimoteX","AllGestureWiimoteY","AllGestureWiimoteZ","BME",
    "Chinatown","Crop","DodgerLoopDay","DodgerLoopGame","DodgerLoopWeekend",
    "EOGHorizontalSignal","EOGVerticalSignal","EthanolLevel","FreezerRegularTrain",
    "FreezerSmallTrain","Fungi","GestureMidAirD1","GestureMidAirD2","GestureMidAirD3",
    "GesturePebbleZ1","GesturePebbleZ2","GunPointAgeSpan","GunPointMaleVersusFemale",
    "GunPointOldVersusYoung","HouseTwenty","InsectEPGRegularTrain","InsectEPGSmallTrain",
    "MelbournePedestrian","MixedShapesRegularTrain","MixedShapesSmallTrain","PLAID",
    "PickupGestureWiimoteZ","PigAirwayPressure","PigArtPressure","PigCVP","PowerCons",
    "Rock","SemgHandGenderCh2","SemgHandMovementCh2","SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ","SmoothSubspace","UMD"
)
"""

dataset_names_additional = ("SPY","ACSF1")

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
results_additional.to_csv(f"{args.output_path}results_additional6.csv")
