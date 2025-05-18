#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 02:31:29 2025

@author: caio
"""


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from catboost import CatBoostRegressor

# Subindo CSV
data_treino = pd.read_csv('/home/caio/github/Kaggle S5E5/data/final.csv')
data_teste = pd.read_csv('/home/caio/github/Kaggle S5E5/data/csv_para_previsao.csv')

# Separando feature e target
target = data_treino['Calories']
feature = data_treino.drop('Calories', axis=1)

# Criar o modelo
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    verbose=0,
    random_state=42
)

# Validação cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, feature, target, cv=kf, scoring='neg_root_mean_squared_error')
print(f'RMSE médio (CV): {-np.mean(scores):.4f}')
# 3.57 = 0.11

# Treinar no conjunto completo
model.fit(feature, target)

# Previsão no conjunto de teste
preds = model.predict(data_teste)

# 12. Gerar CSV de submissão
submission = pd.DataFrame({
    'id': data_teste['id'],  # Verifique se a coluna 'id' está correta
    'Calories': preds
})

# 13. Salvar submissão
submission.to_csv('/home/caio/github/Kaggle S5E5/data/catboost.csv', index=False)