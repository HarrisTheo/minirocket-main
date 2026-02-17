*Stability AEN*

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
# AEN META CONTROLLER
# ==========================================================
class AENController:
    def __init__(self, alpha_e=0.15, alpha_c=0.10, k=6.0):
        # We start with moderate ego and drive to allow 
        # the model to "warm up" during the first few samples.
        self.ego = 0.7
        self.drive = 0.3
        self.alpha_e = alpha_e
        self.alpha_c = alpha_c
        self.k = k

    def update(self, error):
        """
        Calculates the new internal state based on prediction error.
        error: 0 for correct, 1 for incorrect (or continuous loss).
        """
        surprise = abs(error)

        # 1. Fidelity (Ego): Measures how much the model trusts its current state.
        self.ego = (1 - self.alpha_e) * self.ego + \
                   self.alpha_e * (1.0 - surprise)

        # 2. Drive (Adaptation Pressure): The 'urge' to change weights.
        target = 1.0 - self.ego
        self.drive = (1 - self.alpha_c) * self.drive + \
                     self.alpha_c * target

    def plasticity_scale(self):
        """
        Returns the multiplier for the Learning Rate.
        High Drive = High Plasticity (Aggressive Learning).
        """
        return 1 + self.k * (self.drive ** 2)

    def regularization_scale(self):
        """
        Returns the multiplier for the Regularization (Weight Decay).
        High Drive = Low Regularization (Release old constraints).
        """
        # We use your safety floor to prevent numerical collapse
        raw_reg = 1.0 / (1 + 5 * self.drive)
        return max(0.2, raw_reg)



# ==========================================================
# SINGLE RUN
# ==========================================================
def run_minirocket_once(training_data, test_data):

    y_train = training_data[:, 0].astype(int)
    X_train = training_data[:, 1:].astype(np.float32)

    y_test = test_data[:, 0].astype(int)
    X_test = test_data[:, 1:].astype(np.float32)

    # -------------------------------
    # MiniRocket Fit
    # -------------------------------
    t0 = time.perf_counter()
    parameters = fit(X_train)
    t1 = time.perf_counter()

    X_train_t = transform(X_train, parameters)
    t2 = time.perf_counter()

    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_train_t)

    X_train_t = scaler.transform(X_train_t)

    # -------------------------------
    # Online Classifier
    # -------------------------------
    clf = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.01,
        alpha=0.0001
    )

    clf.partial_fit(X_train_t, y_train, classes=np.unique(y_train))
    t3 = time.perf_counter()

    # -------------------------------
    # Transform Test Data
    # -------------------------------
    X_test_t = transform(X_test, parameters)
    X_test_t = scaler.transform(X_test_t)
    t4 = time.perf_counter()

    # -------------------------------
    # Streaming + AEN Control
    # -------------------------------
    aen = AENController()
    base_lr = 0.01
    base_alpha = 0.0001

    preds = []

    for i in range(len(X_test_t)):
        x = X_test_t[i:i+1]
        y_true = y_test[i]

        y_pred = clf.predict(x)[0]
        preds.append(y_pred)

        error = int(y_pred != y_true)

        aen.update(error)

        lr_scale = aen.plasticity_scale()
        reg_scale = aen.regularization_scale()

        clf.eta0 = np.clip(base_lr * lr_scale, 1e-5, 0.1)
        clf.alpha = np.clip(base_alpha * reg_scale, 1e-6, 0.01)

        clf.partial_fit(x, [y_true])

    accuracy = np.mean(np.array(preds) == y_test)
    t5 = time.perf_counter()

    timings = np.array([
        t1 - t0,
        t2 - t1,
        t3 - t2,
        t4 - t3,
        t5 - t4
    ], dtype=np.float64)

    return accuracy, timings

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
results_additional.to_csv(f"{args.output_path}results_additional1.csv")
