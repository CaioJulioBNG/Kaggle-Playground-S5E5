#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 04:00:31 2025

@author: caio
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class EDA:
    def __init__(self, data_treino_path, data_teste_path):
        self.data_treino_path = data_treino_path
        self.data_teste_path = data_teste_path
        self.carregar_data()
        self.replace_sex_treino()
        self.criar_colunas(self.data_treino)
        self.criar_colunas(self.data_teste)
        self.save()
        print('Sucesso')
        
    def carregar_data(self):
        self.data_treino = pd.read_csv(self.data_treino_path)
        self.data_teste = pd.read_csv(self.data_teste_path)
        
    def replace_sex_treino(self):
        self.data_treino['Sex'] = self.data_treino['Sex'].replace({'male': 1.2, 'female': 1})
        self.data_teste['Sex'] = self.data_teste['Sex'].replace({'male': 1.2, 'female': 1})
        
    def criar_colunas(self, data):
        teste = data['Height'] / 100
        data['imc'] = data['Weight'] / (teste ** 2)
        data['sex_age'] = data['Sex'] * data['Age']
        
    def heatmap(self):
        
        plt.figure(figsize=(10, 8))
        # Criar o mapa de calor com melhor formatação
        sns.heatmap(
            data.corr(numeric_only=True),
            annot=True,
            fmt=".2f",                # Formato dos números
            cmap='coolwarm',
            linewidths=0.5,           # Linhas entre os blocos
            linecolor='gray',
            cbar_kws={"shrink": 0.8}, # Tamanho da barra de cores
            square=True               # Manter células quadradas
        )
        
    def save(self):
        self.data_treino.to_csv("data/treino_tratado.csv",index=False)
        self.data_teste.to_csv('data/teste_tratado.csv', index=False)
        
          
if __name__ == "__main__":
    eda = EDA(
        data_treino_path = '/home/caio/github/Kaggle S5E5/data/train.csv',
        data_teste_path = '/home/caio/github/Kaggle S5E5/data/test.csv'
        )
    
    data = eda.data_treino

    
    