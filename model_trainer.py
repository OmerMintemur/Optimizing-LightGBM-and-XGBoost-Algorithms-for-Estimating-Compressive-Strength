# model_trainer.py

import optuna
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import METRIC,MODEL_TYPE

if MODEL_TYPE == 'Xgboost':
    import xgboost as xgb
    print("Xgboost selected")
elif MODEL_TYPE == 'Lightgbm':
    import lightgbm as lgb
    print("Ligthgbm selected")
else:
    raise ValueError(f"Unsupported model type: {MODEL_TYPE}")


def define_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=2),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
        "max_depth": trial.suggest_int("max_depth", 2, 300),
        "rate_drop":trial.suggest_float("rate_drop", 0.0, 1.0),
        "verbose": -1,
        'boosting_type':'dart',
        'reg_lambda':trial.suggest_float("reg_lambda", 0, 2, step=0.05)
    }


def objective(trial, X, y):
    params = define_search_space(trial)
    if MODEL_TYPE == 'Xgboost':
        model = xgb.XGBRegressor(**params)
    else:
        model = lgb.LGBMRegressor(objective="regression", **params)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    if METRIC == 'mae':
        scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=kf)
    elif METRIC == 'mape':
        scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_percentage_error', cv=kf)
        scores = scores*100
    else:  # RMSE
        scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=kf)

    return -scores.mean()


def train_best_model(X, y, best_params):
    if MODEL_TYPE == 'Xgboost':
        model = xgb.XGBRegressor(**best_params)
    else:
        model = lgb.LGBMRegressor(**best_params)

    model.fit(X, y)
    return model
