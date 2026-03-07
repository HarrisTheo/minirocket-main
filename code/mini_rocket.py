import argparse
import time
import numpy as np
import pandas as pd
import os
import re

from minirocket import fit, transform
from sklearn.linear_model import RidgeClassifierCV

def discover_ucr2018_datasets(input_path: str): # Discover all valid UCR-style datasets inside the given input folder
    dataset_names = []  # List that will store the names of valid datasets
    
    for name in sorted(os.listdir(input_path)): # Iterate through all entries in the input directory in sorted order
        ds_dir = os.path.join(input_path, name)
        
        if not os.path.isdir(ds_dir): # Skip the entry if it is not a directory
            continue

        train_path = os.path.join(ds_dir, f"{name}_TRAIN.tsv") # Construct the expected training and test file paths
        test_path = os.path.join(ds_dir, f"{name}_TEST.tsv")

        if os.path.isfile(train_path) and os.path.isfile(test_path): # Keep the dataset only if both TRAIN and TEST files exist
            dataset_names.append(name)

    return tuple(dataset_names) # Return the discovered dataset names as a tuple

# ==========================================================
# SINGLE RUN (PURE MINIROCKET)
# ==========================================================
def run_minirocket_once(training_data, test_data):

    y_train = training_data[:, 0].astype(int) # Split training data into labels and time-series features
    X_train = training_data[:, 1:].astype(np.float32)

    y_test = test_data[:, 0].astype(int) # Split test data into labels and time-series features
    X_test = test_data[:, 1:].astype(np.float32)

    # --------------------------------------------------
    # MiniRocket Kernel Fit
    # --------------------------------------------------
    t0 = time.perf_counter() # Learn the MiniRocket transform parameters from the training set
    parameters = fit(X_train)
    t1 = time.perf_counter()

    # --------------------------------------------------
    # Transform Training Data
    # --------------------------------------------------
    X_train_t = transform(X_train, parameters) # Convert the raw training time series into MiniRocket features
    t2 = time.perf_counter()

    # --------------------------------------------------
    # Train Classifier (STATIC)
    # --------------------------------------------------
    clf = RidgeClassifierCV(alphas=10 ** np.linspace(-3, 3, 10)) # Train a standard static Ridge classifier on the transformed features
    clf.fit(X_train_t, y_train)
    t3 = time.perf_counter()

    # --------------------------------------------------
    # Transform Test Data
    # --------------------------------------------------
    X_test_t = transform(X_test, parameters) # Apply the same MiniRocket transform to the test set
    t4 = time.perf_counter()

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    acc = clf.score(X_test_t, y_test) # Compute classification accuracy on the transformed test set
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

    results = np.zeros(num_runs, dtype=np.float64) # Array to store accuracy from each repeated run
    timings = np.zeros((5, num_runs), dtype=np.float64) # Matrix to store timing breakdown for each run

    for i in range(num_runs):
        acc, t = run_minirocket_once(training_data, test_data) # Repeat the full MiniRocket experiment num_runs times
        results[i] = acc
        timings[:, i] = t

    return results, timings  # Return all accuracies and timing measurements

# ==========================================================
# DATA LOADING HELPERS
# ==========================================================
def load_tsv(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep="\t", header=None)  # Read the TSV file with no header row

    return df.to_numpy() # Convert the DataFrame to a NumPy array

# ==========================================================
# MAIN SCRIPT
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path", required=True)
parser.add_argument("-o", "--output_path", required=True)
parser.add_argument("-n", "--num_runs", type=int, default=10)
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
)

print(f"Found {len(dataset_names_additional)} datasets in {args.input_path}") # Print how many datasets will be processed from the input path

results_additional = pd.DataFrame( # Name the index column so it appears clearly in the saved CSV
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

print("RUNNING".center(80, "=")) # Print a formatted header to indicate that execution is starting

for dataset_name in dataset_names_additional: # Loop through each dataset in the predefined dataset list

    print(dataset_name.center(80, "-")) # Print dataset name as a section header
    print("Loading data".ljust(75, "."), end="", flush=True) # Show loading progress message

    training_path = f"{args.input_path}/{dataset_name}/{dataset_name}_TRAIN.tsv" # Load training and test data from TSV files
    test_path = f"{args.input_path}/{dataset_name}/{dataset_name}_TEST.tsv"

    training_data = load_tsv(training_path)
    test_data = load_tsv(test_path)

    print("Done.")

    print("Performing runs".ljust(75, "."), end="", flush=True) # Show progress message before repeated experiment runs

    results, timings = run_minirocket(  #Run the MiniRocket pipeline multiple times for the current dataset
        training_data,
        test_data,
        num_runs=args.num_runs
    )

    timings_mean = timings.mean(axis=1) # Compute mean timing per stage across all runs

    print("Done.")

    results_additional.loc[dataset_name, "accuracy_mean"] = results.mean() # Store mean accuracy across runs
    results_additional.loc[dataset_name, "accuracy_standard_deviation"] = results.std(ddof=0) # Store standard deviation of accuracy across runs

    results_additional.loc[dataset_name, "time_training_seconds"] = ( # Total training time includes: MiniRocket fit + train transform + classifier training
        timings_mean[0] + timings_mean[1] + timings_mean[2]
    )

    results_additional.loc[dataset_name, "time_test_seconds"] = (
        timings_mean[3] + timings_mean[4]
    )

print("FINISHED".center(80, "=")) # Print completion header
results_additional.to_csv(f"{args.output_path}results_minirocket_baseline_tsv.csv", index=True) # Save the final results DataFrame to CSV