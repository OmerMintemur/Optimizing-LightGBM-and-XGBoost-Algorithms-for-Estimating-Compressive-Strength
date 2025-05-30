# shap_analysis.py

import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import FEATURE_NAMES, RESULTS_DIR, BEFORE_AFTER, METRIC, MODEL_TYPE

def compute_shap_values(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap_values.feature_names = FEATURE_NAMES
    return shap_values

def plot_shap_summary(shap_values, X):
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    fig = plt.gcf()
    fig.set_size_inches(17, 8)
    plt.title(f"SHAP Summary Plot - {BEFORE_AFTER} Augmentation - {METRIC.upper()}\n{MODEL_TYPE}")
    plt.savefig(f"{RESULTS_DIR}shap_summary_{BEFORE_AFTER}.pdf", dpi=300)
    # plt.close()

def plot_shap_bar(shap_values, X):
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    plt.title(f"SHAP Feature Importance - {BEFORE_AFTER} Augmentation - {METRIC.upper()}\n{MODEL_TYPE}")
    plt.savefig(f"{RESULTS_DIR}shap_bar_{BEFORE_AFTER}.pdf", dpi=300)
    # plt.close()

def plot_correlation_matrix(shap_values, X):
    shap_df = pd.DataFrame(shap_values.values, columns=FEATURE_NAMES)
    corr = shap_df.corr().abs()
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title(f"SHAP Feature Correlation - {BEFORE_AFTER} Augmentation - {METRIC.upper()}\n{MODEL_TYPE}")
    plt.savefig(f"{RESULTS_DIR}shap_correlation_{BEFORE_AFTER}.pdf", dpi=300)
    # plt.close()
