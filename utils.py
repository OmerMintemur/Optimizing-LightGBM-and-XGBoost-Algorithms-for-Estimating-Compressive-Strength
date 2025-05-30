# utils.py

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from config import RESULTS_DIR

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_results(model, X, y, filename):
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    with open(f"{RESULTS_DIR}{filename}", 'w') as f:
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write("Model Parameters:\n")
        for param, value in model.get_params().items():
            f.write(f"{param}: {value}\n")

def plot_predictions(y_true, y_pred, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.savefig(f"{RESULTS_DIR}{filename}", dpi=300)
    plt.close()
