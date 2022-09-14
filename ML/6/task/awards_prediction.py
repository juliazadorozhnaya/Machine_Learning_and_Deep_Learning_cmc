from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from numpy import ndarray

from lightgbm import LGBMRegressor

def train_model_and_predict(train_file: str, test_file: str) -> ndarray:

    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)

    df_train.fillna('label_empty')
    df_test.fillna('label_empty')

    for col in ['genres', 'directors', 'filming_locations', 'keywords']:
        for i in range(df_train.shape[0]):
            if df_train[col][i] != 'label_empty':
                df_train[col][i] = ','.join(data for data in df_train[col][i] if data)
        df_train[col] = df_train[col].astype('category')

        for i in range(df_test.shape[0]):
            if df_test[col][i] != 'label_empty':
                df_test[col][i] = ','.join(data for data in df_test[col][i] if data)
        df_test[col] = df_test[col].astype('category')

    for i in range(3):
        df_train[f"actor_{i}_gender"] = df_train[f"actor_{i}_gender"].astype('category')
        df_test[f"actor_{i}_gender"] = df_test[f"actor_{i}_gender"].astype('category')

    y_train = df_train["awards"]
    del df_train["awards"]

    regressor = LGBMRegressor()

    regressor.fit(df_train, y_train)

    return regressor.predict(df_test)