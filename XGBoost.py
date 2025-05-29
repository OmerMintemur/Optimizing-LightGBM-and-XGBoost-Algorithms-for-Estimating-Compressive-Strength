# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:17:24 2023

@author: OMER
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:05:05 2023

@author: OMER
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import optuna
from sklearn.model_selection import cross_val_score
from optuna.samplers import TPESampler
import shap

# Parameters to be optimized
def return_param(trial):
    param_grid = {
        #         "device_type": trial.suggest_categorical("device_type", ['gpu']),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=2),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
        "max_depth": trial.suggest_int("max_depth", 2, 300),
        "rate_drop":trial.suggest_float("rate_drop", 0.0,1.0),
        'reg_lambda':trial.suggest_float("reg_lambda", 0, 2, step=0.05),
        'boosting_type':'dart',
        "verbosity":0
    }
    return param_grid


# Save the best model (callback function)
best_booster = None
model = None
mae_values = []
r_2_scores = []
path = ""
metric = ""


# Chose one of them
MAE = False

MAPE = False

RMSE = True

for_optimization = ""

before_after = 'After'

optimization_history_title = f"Optimization Process of XGBOOST {before_after} Data Augmentation"
parameter_importance_title = f"Parameter Importance of XGBOOST {before_after} Data Augmentation"


if MAE == True:
    path = 'XGBOOST/Augmentation/MAE/'
    metric = 'mae'
    for_optimization = "MAE"
if MAPE == True:
    path = 'XGBOOST/Augmentation/MAPE/'
    metric='mape'
    for_optimization = "MAPE"
if RMSE == True:
    path = 'XGBOOST/Augmentation/RMSE/'
    metric = 'rmse'
    for_optimization = "RMSE"


# https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/
# Our objective function is to lower the error
# Also do a kfold cross validation
# https://slundberg.github.io/shap/notebooks/Census%20income%20classification%20with%20XGBoost.html
# https://stackoverflow.com/questions/62514395/score-obtained-from-cross-val-score-is-rmse-or-mse
def objective(trial):
    global model    
    """ data = pd.read_excel('Veri_Son.xlsx')
    data = data.drop(["No"], axis=1)
    data = data.drop(["Toplam"], axis=1)
    X, y = data.iloc[:, 0:7], data.iloc[:, 7:] """

    data = pd.read_csv('AugmentedData.csv')
    data = data.sample(frac=1)

    X, y = data.iloc[:, 1:8], data.iloc[:, 8:]


    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    param_grid = return_param(trial)
    model = xgb.XGBRegressor(objective="reg:squarederror", **param_grid)
    k_fold = KFold(n_splits=10, shuffle=True)

    # optional parameters
    if MAE == True:
        mae = cross_val_score(model,X,y.values.ravel(), scoring='neg_mean_absolute_error', cv=k_fold)
        return -1 * mae.mean()
    elif MAPE == True:
        mape = cross_val_score(model,X,y.values.ravel(), scoring='neg_mean_absolute_percentage_error', cv=k_fold)
        return -1 * mape.mean()*100
    elif RMSE == True:
        res = cross_val_score(model, X, y.values.ravel(), scoring='neg_root_mean_squared_error', cv=k_fold)
        return res.mean() * -1


def callback(study, trial):
    global best_booster
    if study.best_trial == trial:
        best_booster = model


# Create a study
direction = ''
if MAE==True or RMSE==True or MAPE==True:
    direction='minimize'


study = optuna.create_study(sampler=TPESampler(multivariate=True,constant_liar=True), 
                            direction=direction, study_name="XGBOOST Regressor")
study.optimize(objective, n_trials=1000, callbacks=[callback])

# Visualize the progress
fig = optuna.visualization.plot_optimization_history(study)
fig.update_layout(title=dict(text=f"Optimization History According to {for_optimization}<br>Model XGBOOST <br>{before_after} Augmentation"))
fig.write_image(f'{path}Optimization_History.png')

# Show the best model

# 
print(best_booster)
with open(f'{path}Results.txt', 'w') as f:
    f.write(f"\tBest value ({metric}): {study.best_value:.5f}\n")
    f.write(f"\tBest params:\n")
    for key, value in study.best_params.items():
        f.write(f"\t\t{key}: {value}\n")
# The file is automatically closed when the 'with' block ends
    
# Parameter Importance
fig = optuna.visualization.plot_param_importances(study)
fig.update_layout(title=dict(text=f"Parameter Importance History According to {for_optimization}<br>Model XGBOOST <br>{before_after} Augmentation"))
fig.write_image(f'{path}Parameter_Importance.png')
# show(fig)

# Again Read the data
# If augmentation false

""" data = pd.read_excel('Veri_Son.xlsx')
data = data.drop(["No"], axis=1)
data = data.drop(["Toplam"], axis=1)
data = data.sample(frac=1)
X, y = data.iloc[:, 0:7], data.iloc[:, 7:] """


# if Augmentation True
data = pd.read_csv('AugmentedData.csv')
data = data.sample(frac=1)
X, y = data.iloc[:, 1:8], data.iloc[:, 8:]


# Best Model Params
param = {'learning_rate': study.best_params['learning_rate'],
         'max_depth': study.best_params['max_depth'],
         'n_estimators': study.best_params['n_estimators'],
         'rate_drop': study.best_params['rate_drop'],
         'objective': 'reg:squarederror',
         'verbose': -1,
         'reg_lambda':study.best_params['reg_lambda'],
         'metric': metric}

# https://xgboost.readthedocs.io/en/stable/python/python_intro.html#py-data
# Fit the best model

# train_data = xgb.DMatrix(data, label=y)
# model = xgb.train(param, train_data)


xgb_ = xgb.XGBRegressor(**param)
model = xgb_.fit(X,y)

k_fold = KFold(n_splits=10, shuffle=True)
r_2 = cross_val_score(model,X,y.values.ravel(), scoring='r2', cv=k_fold)

with open(f'{path}Results.txt', 'a') as f:
    f.write('\n')
    f.write(f'R2 - {r_2.mean()}')


# Utilize SHAP Values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
print(shap_values.shape)

# Correlation
features = ["Cement", "Silica Fume", "Quartz Sand","Silica Sand","Superplasticizer","Micro Steel Fiber" , "Water"]
""" features = data.columns
features = features.to_list()[1:8] """
shap_df = pd.DataFrame(shap_values, columns=pd.Index(features, name='features'))


plt.figure(figsize=(12,9))
feat_order = shap_df.abs().mean().sort_values().index.tolist()
ax=sns.heatmap(shap_df.corr().abs().loc[feat_order, feat_order], cbar=True,annot=True,cmap="crest")

ax.set_yticklabels(ax.get_yticklabels(), rotation = 30, fontsize = 10)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, fontsize = 10)
plt.title(f"Feature Correlation Matrix According to Shapley Values\nModel XGBOOST\n{before_after} Augmentation \n According to {for_optimization}", fontsize=16)
plt.ylabel("")
plt.xlabel("")

plt.savefig(f'{path}Correlation_{before_after}_Augmentation.png', dpi=300,bbox_inches='tight')
plt.show()
# Correlation Ends

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

""" data_list = ["Cement", "Silica Fume", "Quartz Sand", "Silica Sand",
             "Superplasticizer", "Micro Steel Fiber", "Water"]

for data_x in data_list:
    for data_y in data_list:
        if data_y == data_x:
            continue
        else:
            shap.plots.scatter(shap_values[:, f"{data_x}"], color=shap_values[:, f"{data_y}"], show=False)
            plt.savefig(f'{path}DependencePlot_{data_x}_{data_y}.png', dpi=300) """

# Bar Plots
fig = plt.figure(1, figsize=(20, 20))
shap.plots.bar(shap_values,show=False)
fig = plt.gcf()
fig.set_size_inches(15, 12)
plt.title(f"Feature Importance \n Model XGBOOST \n {before_after} Augmentation \n According to {for_optimization}",fontsize=16)
plt.savefig(f"{path}Feature Importance Model XGBOOST {before_after} Augmentation.png",format="png", dpi=300 )
plt.show()

# BeeSwarm Plots
shap.plots.beeswarm(shap_values, max_display=99, show=False)
fig = plt.gcf()
fig.set_size_inches(15, 12)
plt.title(f"Effect of Features\nModel XGBOOST \n {before_after} Augmentation \n According to {for_optimization}",fontsize=16)
plt.savefig(f"{path}Effect of Features Model XGBOOST {before_after} Augmentation.png",format="png", dpi=300 )
plt.show()


plt.subplot(2, 1, 1)
plt.gcf()
shap.plots.bar(shap_values.abs.max(0), max_display=99, show=False)
plt.subplot(2, 1, 2)
shap.plots.beeswarm(
    shap_values.abs, color="shap_red", max_display=99, show=False, plot_size=None
)
ax = plt.gca()
masv = {}
for feature in ax.get_yticklabels():
    name = feature.get_text()
    col_ind = X.columns.get_loc(name)
    mean_abs_sv = np.mean(np.abs(shap_values.values[:, col_ind]))
    masv[name] = mean_abs_sv
ax.scatter(
    masv.values(),
    [i for i in range(len(X.columns))],
    zorder=99,
    label="Mean Absolute SHAP Value",
    c="k",
    marker="|",
    linewidths=3,
    s=100,
)
ax.legend(frameon=True)
plt.suptitle(f"Maximum and Mean Effect of Attributes\n Model XGBOOST \n{before_after} Augmentation \n According to {for_optimization}",fontsize=16)
fig = plt.gcf()
fig.set_size_inches(15, 12)
plt.savefig(f"{path}bar_beeswarm.png",format="png", dpi=300)

""" n = 2
fig, ax = plt.subplots(1, n, figsize=(15, 3))

for i, (k, v) in enumerate(sorted(masv.items(), key=lambda x: x[1], reverse=True)):
    if i < n:
        shap.plots.scatter(shap_values[:, k], ax=ax[i], show=False, alpha=0.6)
        ax[i].grid(axis="y")
        if i != 0:
            ax[i].set_ylabel("")
            ax[i].spines["left"].set_visible(False)
            ax[i].set_ylim(ax[0].get_ylim())
            ax[i].set_yticklabels(["" for _ in range(len(ax[0].get_yticks()))])
        else:
            ax[i].set_ylabel("SHAP value")
fig.savefig(f"{path}Result_4.png", format='png', dpi=300) """


# Draw Regession Lines
# If augmentation
X, y_real = data.iloc[:, 1:8], data.iloc[:, 8:]
# else
""" X, y_real = data.iloc[:, 0:7], data.iloc[:, 7:] """

y_predict = model.predict(X)
y_predict = y_predict.reshape(X.shape[0], 1)

plt.figure(figsize=(10, 10))
plt.scatter(y_real, y_predict, c='crimson')
plt.grid('on')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_predict), max(y_real.values))
p2 = min(min(y_predict), min(y_real.values))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.title(f"True Values and Predictions of the XGBOOST Model for MPa Values\nModel XGBOOST\n{before_after} Augmentation \n According to {for_optimization}", fontsize=24)
plt.xlabel('Measured Compressive Strength (MPa)', fontsize=12)
plt.ylabel('Predicted Compressive Strength (MPa)', fontsize=12)
plt.axis('equal')

plt.savefig(f'{path}Result_10_300DPI.png', format='png', dpi=300)
plt.show()

# Draw Predicted and Real Values Comparison
x_axis = np.arange(0,X.shape[0])
plt.figure(figsize=(22, 8))
plt.title(f"Predicted and Real Values of Compressive Strength (MPa)\nModel XGBOOST\n{before_after} Augmentation",fontsize=16)
plt.plot(x_axis, y_predict, label="Predicted Values")
plt.plot(x_axis, y_real, label="Real Values")
# sns.regplot(x_axis, y_predict,scatter=False,label="Reg Line - CI=%95",robust=True)
plt.ylabel("Compressive Strength (MPa)",fontsize=16)
plt.xlabel("Data Samples",fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid("on")
plt.legend(loc = 4, prop={'size': 18})
plt.savefig(f'{path}Result_5_300DPI.png', format='png', dpi=300)
plt.show()
