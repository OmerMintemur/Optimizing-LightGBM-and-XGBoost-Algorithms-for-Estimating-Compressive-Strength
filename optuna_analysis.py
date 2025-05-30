import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import FEATURE_NAMES, RESULTS_DIR, BEFORE_AFTER, METRIC, MODEL_TYPE
import optuna


def visualize_progress(study):
    fig = optuna.visualization.plot_param_importances(study)
    fig.update_layout(width=int(2000),height=int(750))
    fig.update_layout(title=dict(text=f"Parameter Importance History According to {METRIC}<br>Model {MODEL_TYPE} <br>{BEFORE_AFTER} Augmentation"))
    fig.write_image(f'{RESULTS_DIR}Parameter_Importance.pdf')

def optimization_history(study):
    fig = optuna.visualization.plot_optimization_history(study)
    fig.update_layout(width=int(2000),height=int(750))
    fig.update_layout(title=dict(text=f"Optimization History According to {METRIC}<br>Model {MODEL_TYPE} <br>{BEFORE_AFTER} Augmentation"))
    fig.write_image(f'{RESULTS_DIR}Optimization_History.pdf')