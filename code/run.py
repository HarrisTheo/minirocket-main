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

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path", required=True)
parser.add_argument("-o", "--output_path", required=True)
parser.add_argument("-n", "--num_runs", type=int, default=10)
parser.add_argument("-k", "--num_kernels", type=int, default=10_000)
args = parser.parse_args()

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

def run_minirocket_once(training_data, test_data):
    y_train = training_data[:, 0].astype(int)
    X_train = training_data[:, 1:].astype(np.float32)
    X_train = np.diff(X_train, axis=1).astype(np.float32)

    y_test = test_data[:, 0].astype(int)
    X_test = test_data[:, 1:].astype(np.float32)

    t0 = time.perf_counter()
    parameters = fit(X_train)
    t1 = time.perf_counter()

    X_train_t = transform(X_train, parameters)
    t2 = time.perf_counter()

    clf = RidgeClassifierCV(alphas=10 ** np.linspace(-3, 3, 10))

    clf.fit(X_train_t, y_train)
    t3 = time.perf_counter()

    X_test_t = transform(X_test, parameters)
    t4 = time.perf_counter()

    acc = clf.score(X_test_t, y_test)
    t5 = time.perf_counter()

    timings = np.array([t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4], dtype=np.float64)
    return acc, timings

def run_minirocket(training_data, test_data, num_runs = 10):
    results = np.zeros(num_runs, dtype=np.float64)
    timings = np.zeros((5, num_runs), dtype=np.float64)

    for i in range(num_runs):
        acc, t = run_minirocket_once(training_data, test_data)
        results[i] = acc
        timings[:, i] = t

    return results, timings

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
        training_data = np.loadtxt(f"{args.input_path}/{dataset_name}/{dataset_name}_TRAIN.txt")
        test_data = np.loadtxt(f"{args.input_path}/{dataset_name}/{dataset_name}_TEST.txt")
    else:
        training_data = np.loadtxt(f"{args.input_path}/{dataset_name}/{dataset_name}_TRAIN.txt", delimiter=",")
        test_data = np.loadtxt(f"{args.input_path}/{dataset_name}/{dataset_name}_TEST.txt", delimiter=",")

    print("Done.")

    print("Performing runs".ljust(75, "."), end="", flush=True)

    results, timings = run_minirocket(training_data, test_data)
    timings_mean = timings.mean(axis=1)

    print("Done.")

    results_additional.loc[dataset_name, "accuracy_mean"] = results.mean()
    results_additional.loc[dataset_name, "accuracy_standard_deviation"] = results.std(ddof=0)

    results_additional.loc[dataset_name, "time_training_seconds"] = timings_mean[0] + timings_mean[1] + timings_mean[2]
    results_additional.loc[dataset_name, "time_test_seconds"] = timings_mean[3] + timings_mean[4]

print("FINISHED".center(80, "="))
results_additional.to_csv(f"{args.output_path}results_additional.csv")