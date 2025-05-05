#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 00:12:03 2025

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

# modelo zerado
modelo = LinearRegression()

# 4. K-Fold cross-validation: 4 dobras
kf = KFold(n_splits=4, shuffle=True, random_state=69)

# 5. Validação cruzada com RMSE (negativo porque o sklearn usa maximização)
scores = cross_val_score(
    modelo,
    feature,
    target,
    cv=kf,
    scoring='neg_root_mean_squared_error'
)

# 6. Resultados
print("RMSE por dobra:", -scores)
print(f"RMSE médio: {-np.mean(scores):.4f}")
print(f"Desvio padrão do RMSE: {np.std(-scores):.4f}")

# 7. Treinar modelo final com todos os dados
modelo.fit(feature, target)

# 8. Previsão nos dados finais
previsao = modelo.predict(data_teste)
previsao = np.maximum(previsao, 0)  # Isso substitui qualquer valor negativo por 0

# 9. Gerar CSV de submissão
submission = pd.DataFrame({
    'id': data_teste['id'],  # Verifique se a coluna 'Id' existe
    'Calories': previsao
})

# 10. Submissão
submission.to_csv('/home/caio/github/Kaggle S5E5/data/submissao_linear.csv', index=False)
print("Arquivo de submissão salvo em 'submissao_linear.csv'")