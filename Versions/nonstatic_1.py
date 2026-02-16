#Agressive AEN

import argparse
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from minirocket import fit, transform 

# ==========================================================
# AEN META CONTROLLER
# ==========================================================
class AENController:
    def __init__(self, alpha_e=0.45, alpha_c=0.35, k=25.0):
        self.ego = 0.5
        self.drive = 0.5
        self.alpha_e = alpha_e  
        self.alpha_c = alpha_c  
        self.k = k              

    def update(self, error):
        surprise = float(error)
        self.ego = (1 - self.alpha_e) * self.ego + self.alpha_e * (1.0 - surprise)
        target_drive = 1.0 - self.ego
        if target_drive > self.drive:
            self.drive = (1 - 0.6) * self.drive + 0.6 * target_drive
        else:
            self.drive = (1 - 0.05) * self.drive + 0.05 * target_drive

    def plasticity_scale(self):
        return 1 + self.k * (self.drive ** 3)

    def regularization_scale(self):
        return max(0.005, 1.0 / (1 + 50 * self.drive))

# ==========================================================
# PRUNING LOGIC (Glitch Protection)
# ==========================================================
def prune_kernels(X_train_t, y_train, top_pct=0.8):
    # Vectorized correlation check
    corrs = np.array([np.abs(np.corrcoef(X_train_t[:, i], y_train)[0, 1]) 
                     for i in range(X_train_t.shape[1])])
    corrs = np.nan_to_num(corrs)
    threshold = np.percentile(corrs, (1 - top_pct) * 100)
    return np.where(corrs >= threshold)[0]

# ==========================================================
# STABLE ENGINE
# ==========================================================
def run_minirocket_stable(training_data, test_data, num_kernels=4000, seed=42):
    np.random.seed(seed)
    
    def prep(d):
        labels, feats = d[:, 0], d[:, 1:]
        feats_c = feats - np.mean(feats, axis=1, keepdims=True)
        v = np.std(feats, axis=1, keepdims=True) + 1e-6
        return labels, feats_c / v

    y_train, X_train = prep(training_data)
    y_test, X_test = prep(test_data)

    # 1. Transform & Prune
    parameters = fit(X_train.astype(np.float32), num_features=num_kernels) 
    X_train_t = transform(X_train.astype(np.float32), parameters)
    
    valid_idx = prune_kernels(X_train_t, y_train, top_pct=0.8)
    X_train_t = X_train_t[:, valid_idx]

    scaler = StandardScaler(with_mean=False)
    X_train_t = scaler.fit_transform(X_train_t)

    # 2. Model Setup
    clf = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=0.05, alpha=0, average=True)
    num_epochs = 50 # Set this to 20 or 50 to force memorization
    for epoch in range(num_epochs):
        clf.partial_fit(X_train_t, y_train, classes=np.unique(y_train))

    # 3. Test Phase with Adaptive AEN
    X_test_t = transform(X_test.astype(np.float32), parameters)[:, valid_idx]
    X_test_t = scaler.transform(X_test_t)

    aen = AENController()
    correct, wrong = 0, 0
    base_lr, base_alpha = 0.05, 0.001

    for i in range(len(X_test_t)):
        x = X_test_t[i:i+1].copy()
        
        # Attention Mask
        if aen.drive > 0.6:
            x[np.abs(x) < np.percentile(np.abs(x), 20)] = 0
        
        y_true = y_test[i]
        y_pred = clf.predict(x)[0]

        if y_pred == y_true:
            correct += 1
            aen.update(0)
        else:
            wrong += 1
            aen.update(1)

        clf.eta0 = np.clip(base_lr * aen.plasticity_scale(), 1e-4, 0.7)
        clf.alpha = np.clip(base_alpha * aen.regularization_scale(), 1e-7, 0.01)
        clf.partial_fit(x, [y_true])

    return correct / (correct + wrong), correct, wrong

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-n", "--num_runs", type=int, default=10)
    parser.add_argument("-k", "--num_kernels", type=int, default=10_000)
    args = parser.parse_args()

    for name in ("SPY", "ACSF1"):
        print(f" ACCURACY BENCHMARK: {name} ".center(80, "="))
        train_path = os.path.join(args.input_path, name, f"{name}_TRAIN.txt")
        test_path = os.path.join(args.input_path, name, f"{name}_TEST.txt")
        
        tr_data, te_data = np.loadtxt(train_path), np.loadtxt(test_path)
        accs = []
        grand_hits = 0
        grand_misses = 0

        for r in range(args.num_runs):
            acc, h, m = run_minirocket_stable(tr_data, te_data, args.num_kernels, seed=r)
            accs.append(acc)
            grand_hits += h
            grand_misses += m
            print(f"Run {r+1:2d}: {acc:.4f} | [Correct: {h:4d} | Wrong: {m:4d}]")
        
        print("-" * 80)
        print(f"Final Summary for {name}:")
        print(f"  Mean Accuracy:      {np.mean(accs):.4f}")
        print(f"  Aggregated Correct: {grand_hits}")
        print(f"  Aggregated Wrong:   {grand_misses}")
        print("=" * 80 + "\n")
