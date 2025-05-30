# data_loader.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import AUGMENTED_DATA_PATH, FEATURE_NAMES

def load_data(augmented=True):
    if augmented:
        data = pd.read_csv(AUGMENTED_DATA_PATH)
        data = data.sample(frac=1).reset_index(drop=True)
        X = data[FEATURE_NAMES]
        y = data['Compressive Strength']
    else:
        data = pd.read_excel('Veri_Son.xlsx')
        data = data.drop(["No", "Toplam"], axis=1)
        data = data.sample(frac=1).reset_index(drop=True)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    return X, y

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler