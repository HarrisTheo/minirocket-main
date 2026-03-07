# MiniROCKET & MiniROCKET + AEN — Time Series Classification

This repository contains two experiment scripts for time series classification using the UCR Archive (TSV format):

| Script | Method |
|---|---|
| `mini_rocket.py` | Pure MiniROCKET with a static RidgeClassifierCV |
| `mini_rocket_aen.py` | MiniROCKET + AEN plasticity regulator (online SGD classifier) |

---

## Requirements

Python 3.10+ is recommended. You will also need the `minirocket` module available locally (e.g. from the [ROCKET preprint repo](https://github.com/angus924/minirocket)).

### Dependencies

```
joblib==1.5.3
llvmlite==0.46.0
numba==0.63.1
numpy==2.2.6
pandas==2.3.3
python-dateutil==2.9.0.post0
pytz==2025.2
scikit-learn==1.7.2
scipy==1.15.3
six==1.17.0
threadpoolctl==3.6.0
tzdata==2025.3
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Ensure `minirocket.py` is on your Python path

Place the `minirocket.py` module (from the ROCKET preprint repo) in the same directory as the scripts, or add its location to `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/rocket-master/preprint:$PYTHONPATH
```

---

## Dataset Structure

Both scripts expect UCR datasets in TSV format, organised as follows:

```
UCRArchive_2018_tsv/
├── BME/
│   ├── BME_TRAIN.tsv
│   └── BME_TEST.tsv
├── Chinatown/
│   ├── Chinatown_TRAIN.tsv
│   └── Chinatown_TEST.tsv
└── ...
```

Each TSV file has **no header**. The first column is the class label and the remaining columns are the time series values.

---

## Running the Scripts

### MiniROCKET (baseline)

```bash
ipython mini_rocket.py \
  -i ../../rocket-master/preprint/datasets/UCRArchive_2018_tsv \
  -o ./
```

Output saved to: `./results_minirocket_baseline_tsv.csv`

---

### MiniROCKET + AEN

```bash
ipython mini_rocket_aen.py \
  -i ../../rocket-master/preprint/datasets/UCRArchive_2018_tsv \
  -o ./
```

Output saved to: `./results_additional_aen_tsv.csv`

---

## Arguments

| Flag | Long form | Required | Default | Description |
|---|---|---|---|---|
| `-i` | `--input_path` | ✅ | — | Path to the UCR TSV dataset root folder |
| `-o` | `--output_path` | ✅ | — | Path to the output folder (or full CSV filename) |
| `-n` | `--num_runs` | ❌ | `10` | Number of repeated runs per dataset |
| `-k` | `--num_kernels` | ❌ | `10000` | Number of MiniROCKET kernels *(AEN script only)* |

---

## Output Format

Both scripts produce a CSV with one row per dataset and the following columns:

| Column | Description |
|---|---|
| `dataset` | Dataset name |
| `accuracy_mean` | Mean classification accuracy across runs |
| `accuracy_standard_deviation` | Std deviation of accuracy across runs |
| `time_training_seconds` | Mean total training time (fit + transform + classifier) |
| `time_test_seconds` | Mean total test time (transform + inference) |

---

## How It Works

### `mini_rocket.py`
1. Fits MiniROCKET kernel parameters on the training set.
2. Transforms both train and test sets into feature vectors.
3. Trains a `RidgeClassifierCV` (static, cross-validated regularisation).
4. Evaluates accuracy on the test set.
5. Repeats for `num_runs` runs and reports mean ± std.

### `mini_rocket_aen.py`
1. Fits MiniROCKET kernel parameters and transforms features (same as above).
2. Trains an initial `SGDClassifier` (logistic loss) on the training features.
3. At test time, processes samples **one by one in a streaming fashion**:
   - Predicts the label.
   - Computes the prediction error.
   - Updates the **AEN plasticity regulator**, which scales the learning rate based on recent error stability.
   - Calls `partial_fit` to update the classifier online.
4. This allows the classifier to adapt to distributional shift (e.g. financial regime changes in the SPY dataset).

---

## Datasets Evaluated

Both scripts evaluate the following 34 datasets (including the custom SPY financial dataset):

`SPY`, `AllGestureWiimoteX/Y/Z`, `BME`, `Chinatown`, `Crop`, `DodgerLoopDay/Game/Weekend`, `EOGHorizontalSignal`, `EOGVerticalSignal`, `EthanolLevel`, `FreezerRegularTrain`, `FreezerSmallTrain`, `GesturePebbleZ1/Z2`, `GunPointAgeSpan`, `GunPointMaleVersusFemale`, `GunPointOldVersusYoung`, `HouseTwenty`, `InsectEPGRegularTrain`, `InsectEPGSmallTrain`, `MelbournePedestrian`, `MixedShapesRegularTrain`, `MixedShapesSmallTrain`, `PLAID`, `PowerCons`, `Rock`, `SemgHandGenderCh2`, `SemgHandMovementCh2`, `SemgHandSubjectCh2`, `ShakeGestureWiimoteZ`, `SmoothSubspace`