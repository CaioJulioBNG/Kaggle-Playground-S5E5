#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 00:12:03 2025

@author: caio
"""

import pandas as pd
import numpy as np
import optuna
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from catboost import CatBoostRegressor

# Subindo CSV
data_treino = pd.read_csv('/home/caio/github/Kaggle S5E5/data/final.csv')
data_teste = pd.read_csv('/home/caio/github/Kaggle S5E5/data/csv_para_previsao.csv')

# Separando feature e target
target = data_treino['Calories']
feature = data_treino.drop('Calories', axis=1)

# K-Fold cross-validation
kf = KFold(n_splits=4, shuffle=True, random_state=69)

# Função de otimização com Optuna
def objective(trial):
    scaler_name = trial.suggest_categorical("scaler", ["standard", "minmax", "none"])
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
    positive = trial.suggest_categorical("positive", [True, False])

    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = "passthrough"

    model = Pipeline([
        ("scaler", scaler),
        ("regressor", LinearRegression(fit_intercept=fit_intercept, positive=positive))
    ])

    scores = cross_val_score(
        model,
        feature,
        target,
        cv=kf,
        scoring='neg_root_mean_squared_error'
    )

    return -np.mean(scores)

# Rodando o Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Melhores hiperparâmetros encontrados:")
print(study.best_params)

# Criando o modelo com os melhores hiperparâmetros
best_params = study.best_params

if best_params["scaler"] == "standard":
    scaler_final = StandardScaler()
elif best_params["scaler"] == "minmax":
    scaler_final = MinMaxScaler()
else:
    scaler_final = "passthrough"

modelo_linear = Pipeline([
    ("scaler", scaler_final),
    ("regressor", LinearRegression(
        fit_intercept=best_params["fit_intercept"],
        positive=best_params["positive"]
    ))
])

# Treinando modelo final com todos os dados
modelo_linear.fit(feature, target)

# Previsão no treino (para resíduos)
previsao_treino = modelo_linear.predict(feature)
residuos = target - previsao_treino

# Modelo CatBoost sobre resíduos
modelo_catboost = CatBoostRegressor(verbose=0)
modelo_catboost.fit(feature, residuos)

# Previsão no conjunto de teste
previsao_linear_teste = modelo_linear.predict(data_teste)
residuos_previstos = modelo_catboost.predict(data_teste)

# Previsão final com média para evitar outliers negativos/positivos
previsao_final = (previsao_linear_teste + residuos_previstos) / 2
previsao_final = np.maximum(previsao_final, 0)

# Gerar CSV de submissão
submission = pd.DataFrame({
    'id': data_teste['id'],
    'Calories': previsao_final
})

# Salvar submissão
submission.to_csv('/home/caio/github/Kaggle S5E5/data/submissao_stacking.csv', index=False)
print("Arquivo de submissão salvo em 'submissao_stacking.csv'")
