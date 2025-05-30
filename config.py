# config.py

# Metric selection
METRIC = 'rmse'  # Options: 'mae', 'mape', 'rmse'
BEFORE_AFTER = 'After'  # Options: 'Before', 'After'
MODEL_TYPE = 'Xgboost'  # or 'Xgboost'

# File paths
AUGMENTED_DATA_PATH = 'AugmentedData.csv'
ORIGINAL_DATA_PATH = 'Veri_Son.xlsx'
RESULTS_DIR = f'D:\\Projeler\\Duygu_Hoca++\\Revizyon_18.09.2024\\LIGHTGBM\\Deneme\\{METRIC.upper()}\\'

# Feature names
FEATURE_NAMES = ["Cement", "Silica Fume", "Quartz Sand", "Silica Sand",
                 "Superplasticizer", "Micro Steel Fiber", "Water"]

N_TRIALS = 5 # Change as needed