#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 00:43:08 2025

@author: caio
"""

from catboost import CatBoostRegressor
import pandas as pd 
import numpy as np 
from sklearn.model_selection import KFold
import time
from sklearn.metrics import mean_squared_error

# %%
data_treino = pd.read_csv("/home/caio/github/Kaggle S5E5/data/train.csv")
data_teste = pd.read_csv("/home/caio/github/Kaggle S5E5/data/test.csv")

numerical_features = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp"]

def create_columns(data):
    data['Age_Height'] = data['Age'] * data['Height']
    data['Age_Weight'] = data['Age'] * data['Weight']
    data['Age_Duration'] = data['Age'] * data['Duration']
    data['Age_Heart_Rate'] = data['Age'] * data['Heart_Rate']
    data['Age_Body_Temp'] = data['Age'] * data['Body_Temp']
    data['Height_Weight'] = data['Height'] * data['Weight']
    data['Height_Duration'] = data['Height'] * data['Duration']
    data['Height_Heart_Rate'] = data['Height'] * data['Heart_Rate']
    data['Height_Body_Temp'] = data['Height'] * data['Body_Temp']
    data['Weight_Duration'] = data['Weight'] * data['Duration']
    data['Weight_Heart_Rate'] = data['Weight'] * data['Heart_Rate']
    data['Weight_Body_Temp'] = data['Weight'] * data['Body_Temp']
    data['Duration_Heart_Rate'] = data['Duration'] * data['Heart_Rate']
    data['Duration_Body_Temp'] = data['Duration'] * data['Body_Temp']
    data['Heart_Rate_Body_Temp'] = data['Heart_Rate'] * data['Body_Temp']
    return data
#%%    
treino = create_columns(data_treino)
teste = create_columns(data_teste)

treino['Sex'] = data_treino['Sex'].replace({'male': 1.2, 'female': 1})
teste['Sex'] = data_teste['Sex'].replace({'male': 1.2, 'female': 1})
treino['Sex'] = treino['Sex'].astype("category")
teste['Sex'] = teste['Sex'].astype("category")

feature_treino = treino.drop(columns=['id', 'Calories'])
target_treino = np.log1p(treino["Calories"])
feature_teste = teste.drop(columns=["id"]) # Coletando features do conjunto de teste (pra que não sei)
#%%    
# Converta a coluna categórica para string
feature_treino["Sex"] = feature_treino["Sex"].astype(str)
feature_teste["Sex"] = feature_teste["Sex"].astype(str)

#features1 = feature_treino.tolist() 
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=69)

oof = np.zeros(len(data_treino))   # Armazena previsões para o conjunto de val de cada fold
pred = np.zeros(len(data_teste))  # previsoes feitas no test, para tirar media final

for i, (train_idx, valid_idx) in enumerate(kf.split(feature_treino, target_treino)):
    print(f"\n{'#'*10} Fold {i+1} {'#'*10}")
    x_train = feature_treino.iloc[train_idx].copy()
    y_train = target_treino.iloc[train_idx]
    x_valid = feature_treino.iloc[valid_idx].copy()
    y_valid = target_treino.iloc[valid_idx]
    x_test = feature_teste.copy()
    
    start = time.time()
    
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.02,
        depth=10,
        l2_leaf_reg=3,  # Regularização equivalente ao gamma
        subsample=0.9,
        colsample_bylevel=0.75,
        early_stopping_rounds=30,
        loss_function='RMSE',
        eval_metric='RMSE',
        verbose=100,
    )
    
    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        cat_features=["Sex"],
        early_stopping_rounds=30,
        verbose=100
        )
    
    oof[valid_idx] = model.predict(x_valid)
    pred += model.predict(x_test)
    
    rmse = np.sqrt(mean_squared_error(y_valid, oof[valid_idx]))
    print(f"Fold {i+1} RMSE: {rmse:.4f}")
    print(f"Feature engineering & training time: {time.time() - start:.1f} sec")
pred /= FOLDS
full_rmse = np.sqrt(mean_squared_error(target_treino, oof))
print(f"\nFinal CV RMSE: {full_rmse:.4f}")

# Reverte a transformação log1p feita nos targets (np.expm1 desfaz np.log1p)
final_predictions = np.expm1(pred)

# Cria o DataFrame de submissão com id e Calories
submission_df = pd.DataFrame({
    "id": data_teste["id"],
    "Calories": final_predictions
})

# Salva o DataFrame como CSV no formato exigido
submission_df.to_csv("submission002.csv", index=False)

print("Arquivo submission.csv criado com sucesso!")














































    