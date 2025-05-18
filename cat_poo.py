#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 03:04:59 2025

@author: caio
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from catboost import CatBoostRegressor

class ModeloCat:
    def __init__(self, data_treino_path, data_teste_path):
        self.data_treino_path = data_treino_path
        self.data_teste_path = data_teste_path
        self.modelo = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='RMSE',
            verbose=0,
            random_state=42
            )
        
    def carregar_data(self):
        self.data_treino = pd.read_csv(self.data_treino_path)
        self.data_teste = pd.read_csv(self.data_teste_path)
        self.feature = self.data_treino.drop('Calories', axis=1)
        self.target = self.data_treino['Calories']
        
    def validacao_cruzada(self, dobras=5):
        kf = KFold(n_splits=dobras, shuffle=True, random_state=69)
        scores = cross_val_score(self.modelo, self.feature, self.target, cv=kf, scoring="neg_root_mean_squared_error")
        print(f'RMSE Médio: {-np.mean(scores):.4f}')
        # RMSE padrão: 3.5579
        # RMSE Médio: 3.5596 -> imc e age_sex
        
    def treino(self):
        self.modelo.fit(self.feature, self.target)
        
    def salvar(self):
        preds = self.modelo.predict(self.data_teste)
        submission = pd.DataFrame({
            'id': self.data_teste['id'],  # Verifique se a coluna 'id' está correta
            'Calories': preds
        })

        # 13. Salvar submissão
        submission.to_csv('/home/caio/github/Kaggle S5E5/data/catboost.csv', index=False)
        
if __name__ == "__main__":
    preditor = ModeloCat(
        data_treino_path = '/home/caio/github/Kaggle S5E5/data/treino_tratado.csv',
        data_teste_path = '/home/caio/github/Kaggle S5E5/data/teste_tratado.csv'
        )
    
    #preditor.carregar_data()
   #preditor.validacao_cruzada()
    #preditor.treino()
    #preditor.salvar()
    import matplotlib.pyplot as plt

plt.hist(target, bins=50)
plt.title("Distribuição de Calories - Treino")
plt.show()

