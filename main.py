# main.py

from data_loader import load_data, scale_features
from model_trainer import objective, train_best_model
from shap_analysis import compute_shap_values, plot_shap_summary, plot_shap_bar, plot_correlation_matrix
from utils import ensure_dir, save_results, plot_predictions
from config import RESULTS_DIR, METRIC,N_TRIALS
from optuna_analysis import optimization_history, visualize_progress
import optuna


def main():
    # Load and preprocess data
    X, y = load_data(augmented=True)
    X_scaled, scaler = scale_features(X)

    # Ensure results directory exists
    ensure_dir(RESULTS_DIR)

    # Hyperparameter optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_scaled, y), n_trials=N_TRIALS)

    visualize_progress(study)
    optimization_history(study)

    # Train best model
    best_model = train_best_model(X_scaled, y, study.best_params)

    # Save results
    save_results(best_model, X_scaled, y, 'model_results.txt')

    # Plot predictions
    predictions = best_model.predict(X_scaled)
    plot_predictions(y, predictions, 'actual_vs_predicted.png')

    # SHAP analysis
    shap_values = compute_shap_values(best_model, X_scaled)
    plot_shap_summary(shap_values, X)
    plot_shap_bar(shap_values, X)
    plot_correlation_matrix(shap_values, X)


if __name__ == "__main__":
    main()
