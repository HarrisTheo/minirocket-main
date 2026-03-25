import argparse
import os
import numpy as np
import pandas as pd

from minirocket import fit, transform
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ==========================================================
# AEN REGULATOR
# ==========================================================
class AENRegulator:
    def __init__(self, base_alpha=0.15, gamma=5.0):
        self.ego        = 0.5
        self.c_drive    = 0.5
        self.base_alpha = base_alpha
        self.gamma      = gamma
        self.streak     = 0
        self.alpha_c    = 0.12

    def update(self, error):
        if error != 0:
            self.streak += 1
        else:
            self.streak = 0
        current_alpha    = min(self.base_alpha + (self.streak * 0.15), 0.75)
        instant_fidelity = np.exp(-abs(error))
        self.ego         = (1 - current_alpha) * self.ego + current_alpha * instant_fidelity
        target_c         = self.gamma / (1.0 + self.ego)
        self.c_drive     = (1 - self.alpha_c) * self.c_drive + self.alpha_c * target_c

    def scale(self):
        baseline = self.gamma / 2.2
        diff     = max(0, self.c_drive - baseline)
        return 1 + 25 * (diff ** 3)


# ==========================================================
# FEATURE QUALITY TRACKER
# ==========================================================
class FeatureQualityTracker:
    def __init__(self, n_features, decay=0.98):
        self.n_features          = n_features
        self.decay               = decay
        self.feature_correctness = np.ones(n_features) * 0.5
        self.feature_coef_mag    = np.zeros(n_features)

    def update(self, coef_abs, top_indices, correct: bool):
        self.feature_correctness *= self.decay
        self.feature_coef_mag     = (self.decay * self.feature_coef_mag
                                     + (1 - self.decay) * coef_abs)
        update_val = 1.0 if correct else 0.0
        self.feature_correctness[top_indices] += (1 - self.decay) * update_val

    def score(self):
        coef_norm = self.feature_coef_mag / (self.feature_coef_mag.max() + 1e-9)
        return coef_norm * self.feature_correctness

    def get_ordered_mask(self, top_k=1000, prune_threshold=0.05):
        scores    = self.score()
        max_score = scores.max() + 1e-9
        useful    = np.where(scores >= prune_threshold * max_score)[0]
        ordered   = useful[np.argsort(scores[useful])[::-1]]
        return ordered[:top_k].astype(np.int32)


# ==========================================================
# EGO80 FEATURE PATTERN
# Accumulate vote scores from high-confidence (ego>0.80) correct
# snapshots only. Returns top-k features by combined score.
# ==========================================================
def extract_ego80_pattern(correct_snapshots, n_features, top_k=150, prune=0.07):
    ego80 = [(indices, w) for indices, w in correct_snapshots if w > 0.80]
    if len(ego80) < 10:
        ego80 = correct_snapshots   # fallback if too few

    vote = np.zeros(n_features, dtype=np.float64)
    for indices, weight in ego80:
        ranks         = np.arange(len(indices), 0, -1, dtype=np.float64)
        vote[indices] += ranks * weight

    max_v   = vote.max() + 1e-9
    useful  = np.where(vote >= prune * max_v)[0]
    ordered = useful[np.argsort(vote[useful])[::-1]]
    pattern = ordered[:top_k].astype(np.int32)

    if len(pattern) < 10:
        pattern = np.arange(min(top_k, n_features), dtype=np.int32)

    return pattern, len(ego80)


# ==========================================================
# ALPHA SEARCH
# ==========================================================
ALPHA_GRID = (1000.0, 5000.0, 10000.0, 50000.0, 100000.0, 200000.0, 500000.0)

def best_ridge_alpha(X, y, alphas=ALPHA_GRID):
    best_a, best_score = alphas[0], -1.0
    for a in alphas:
        score = cross_val_score(RidgeClassifier(alpha=a), X, y,
                                cv=5, scoring="accuracy").mean()
        if score > best_score:
            best_score = score
            best_a     = a
    return best_a, best_score


# ==========================================================
# MAIN PIPELINE
# ==========================================================
def run_aen_spy(training_data, test_data):
    y_train = training_data[:, 0].astype(int)
    X_train = training_data[:, 1:].astype(np.float32)
    y_test  = test_data[:, 0].astype(int)
    X_test  = test_data[:, 1:].astype(np.float32)

    # --- MiniRocket ---
    print("Transforming with MiniRocket...")
    parameters = fit(X_train)
    X_train_t  = transform(X_train, parameters)
    scaler     = StandardScaler(with_mean=False)
    X_train_t  = scaler.fit_transform(X_train_t)
    n_features = X_train_t.shape[1]

    X_test_t = transform(X_test, parameters)
    X_test_t = scaler.transform(X_test_t)

    # -------------------------------------------------------
    # PHASE 1: AEN training — discover ego80 feature pattern
    # -------------------------------------------------------
    print("Phase 1: AEN training — discovering high-confidence feature pattern...")

    clf       = SGDClassifier(loss="log_loss", learning_rate="constant",
                              eta0=0.01, max_iter=1000, warm_start=True)
    regulator = AENRegulator()
    tracker   = FeatureQualityTracker(n_features)
    correct_snapshots = []

    clf.fit(X_train_t, y_train)

    for i in range(len(X_train_t)):
        x_full = X_train_t[i:i+1]
        y_true = y_train[i]

        y_pred  = int(clf.predict(x_full)[0])
        error   = int(y_pred != y_true)
        correct = not bool(error)

        regulator.update(error)

        ordered_mask = tracker.get_ordered_mask(top_k=1000)
        tracker.update(np.abs(clf.coef_[0]), ordered_mask, correct)

        if correct:
            correct_snapshots.append((ordered_mask, regulator.ego))

        clf.eta0 = max(1e-4, 0.01 * regulator.scale())
        clf.partial_fit(x_full, [y_true])

    total_correct = len(correct_snapshots)
    ego80_count   = sum(1 for _, w in correct_snapshots if w > 0.80)
    print(f"  -> {total_correct} correct snapshots  |  {ego80_count} with ego > 0.80")

    # -------------------------------------------------------
    # PHASE 2: Extract ego80 pattern + fit Ridge
    # -------------------------------------------------------
    print("Phase 2: Extracting ego80 feature pattern...")

    pattern, n_ego80 = extract_ego80_pattern(correct_snapshots, n_features,
                                              top_k=150, prune=0.07)
    print(f"  -> Pattern: {len(pattern)} features from {n_ego80} ego>0.80 snapshots")

    X_sub         = X_train_t[:, pattern]
    best_a, cv_acc = best_ridge_alpha(X_sub, y_train)
    print(f"  -> Best alpha={best_a}  CV accuracy={cv_acc:.4f}")

    ridge = RidgeClassifier(alpha=best_a)
    ridge.fit(X_sub, y_train)
    train_acc = float(np.mean(ridge.predict(X_sub) == y_train))
    print(f"  -> Train accuracy={train_acc:.4f}")

    # -------------------------------------------------------
    # PHASE 3: Test
    # -------------------------------------------------------
    print("Phase 3: Testing...")

    X_test_sub = X_test_t[:, pattern]
    preds      = ridge.predict(X_test_sub)
    accuracy   = float(np.mean(preds == y_test))

    # Breakdown by class
    for cls in np.unique(y_test):
        mask     = y_test == cls
        cls_acc  = float(np.mean(preds[mask] == y_test[mask]))
        cls_frac = float(mask.mean())
        print(f"  -> Class {cls}: {cls_acc:.4f} acc  ({cls_frac:.2%} of test)")

    print(f"\n  CV  accuracy : {cv_acc:.4f}")
    print(f"  Train accuracy : {train_acc:.4f}")
    print(f"  Test accuracy  : {accuracy:.4f}")

    return accuracy


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path",  required=True)
    parser.add_argument("-o", "--output_path", required=True)
    args = parser.parse_args()

    dataset    = "SPY"
    train_file = os.path.join(args.input_path, dataset, f"{dataset}_TRAIN.txt")
    test_file  = os.path.join(args.input_path, dataset, f"{dataset}_TEST.txt")

    if os.path.exists(train_file):
        train_data = np.loadtxt(train_file)
        test_data  = np.loadtxt(test_file)

        acc = run_aen_spy(train_data, test_data)

        print(f"\nFINAL SPY ACCURACY: {acc:.4f}")

        os.makedirs(args.output_path, exist_ok=True)
        pd.DataFrame([{
            "dataset":       dataset,
            "accuracy":      acc,
            "method":        "AEN-ego80-Ridge",
            "n_features":    150,
            "alpha":         50000,
        }]).to_csv(
            os.path.join(args.output_path, f"{dataset}_final_results.csv"),
            index=False
        )
