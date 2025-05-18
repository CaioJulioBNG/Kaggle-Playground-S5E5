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
modelo_linear = LinearRegression()

# 4. K-Fold cross-validation: 4 dobras
kf = KFold(n_splits=4, shuffle=True, random_state=69)

# 5. Validação cruzada com RMSE (negativo porque o sklearn usa maximização)
scores = cross_val_score(
    modelo_linear,
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
modelo_linear.fit(feature, target)

# 8. Previsão com modelo linear nos dados de treino (para calcular resíduos)
previsao_treino = modelo_linear.predict(feature)
residuos = target - previsao_treino

# 9. Treinar modelo CatBoost sobre os resíduos
modelo_catboost = CatBoostRegressor(verbose=0)
modelo_catboost.fit(feature, residuos)

# 10. Previsão no conjunto de teste
previsao_linear_teste = modelo_linear.predict(data_teste)
residuos_previstos = modelo_catboost.predict(data_teste)

# 11. Previsão final somando os dois modelos
previsao_final = previsao_linear_teste + residuos_previstos
previsao_final = np.maximum(previsao_final, 0)  # Evita valores negativos

# 12. Gerar CSV de submissão
submission = pd.DataFrame({
    'id': data_teste['id'],  # Verifique se a coluna 'id' está correta
    'Calories': previsao_final
})

# 13. Salvar submissão
submission.to_csv('/home/caio/github/Kaggle S5E5/data/submissao_stacking.csv', index=False)
print("Arquivo de submissão salvo em 'submissao_stacking.csv'")
