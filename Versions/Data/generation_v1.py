# high volatile data generation

import numpy as np
import pandas as pd
import yfinance as yf
import os


# ==========================================================
# CONFIG
# ==========================================================
ticker = "SPY"
start_date = "2000-01-01"
end_date = "2024-01-01"

window_size = 60
train_ratio = 0.7  # chronological split
output_dir = "./spy_real_data"


# ==========================================================
# 1. Download SPY prices
# ==========================================================
print("Downloading SPY data...")
data = yf.download(ticker, start=start_date, end=end_date)

prices = data["Adj Close"].values

# ==========================================================
# 2. Compute log returns
# ==========================================================
log_returns = np.diff(np.log(prices))

print(f"Total returns: {len(log_returns)}")


# ==========================================================
# 3. Chronological split
# ==========================================================
split_index = int(len(log_returns) * train_ratio)

train_returns = log_returns[:split_index]
test_returns = log_returns[split_index:]

print(f"Train size: {len(train_returns)}")
print(f"Test size: {len(test_returns)}")


# ==========================================================
# 4. Build sliding windows (inside each segment only)
# ==========================================================
def build_windows(returns, window_size):
    X = []
    y = []

    for i in range(len(returns) - window_size - 1):
        window = returns[i:i + window_size]
        next_return = returns[i + window_size]

        label = 1 if next_return > 0 else 0

        X.append(window)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y


X_train, y_train = build_windows(train_returns, window_size)
X_test, y_test = build_windows(test_returns, window_size)

print(f"Train windows: {X_train.shape}")
print(f"Test windows: {X_test.shape}")


# ==========================================================
# 5. Convert to UCR-style format
# ==========================================================
train_ucr = np.column_stack((y_train, X_train))
test_ucr = np.column_stack((y_test, X_test))


# ==========================================================
# 6. Save in UCR folder format
# ==========================================================
os.makedirs(output_dir, exist_ok=True)

train_path = os.path.join(output_dir, "SPY_TRAIN.txt")
test_path = os.path.join(output_dir, "SPY_TEST.txt")

np.savetxt(train_path, train_ucr)
np.savetxt(test_path, test_ucr)

print("Done.")
print(f"Saved to: {output_dir}")
