#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 00:00:29 2025

@author: caio
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('/home/caio/github/Kaggle S5E5/data/train.csv')
data1 = pd.read_csv('/home/caio/github/Kaggle S5E5/data/test.csv')
data['Sex'] = data['Sex'].replace({'male': 0, 'female': 1})

print("\nInformações do DataFrame:")
print(data.info())

print("\nEstatísticas descritivas:")
print(data.describe(include='all'))

# 4. Verificar valores ausentes
print("\nValores ausentes:")
print(data.isnull().sum())

# 5. Verificar valores únicos por coluna
print("\nValores únicos:")
print(data.nunique())

# 6. Contagem de valores categóricos (ex: gênero)
if 'gender' in data.columns:
    print("\nContagem de gêneros:")
    print(data['gender'].value_counts())

# 7. Correlação entre variáveis numéricas
print("\nCorrelação entre variáveis numéricas:")
print(data.corr(numeric_only=True))

# 8. Visualizações básicas

# Histograma das colunas numéricas
data.hist(bins=20, figsize=(10, 8))
plt.tight_layout()
plt.show()

# Gráfico de barras para gênero, se existir
if 'gender' in data.columns:
    sns.countplot(data=data, x='gender')
    plt.title("Distribuição por gênero")
    plt.show()

# Mapa de calor da correlação
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

plt.title("Mapa de Calor das Correlações", fontsize=14, pad=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

data.to_csv("data/final.csv",index=False)
data1['Sex'] = data['Sex'].replace({'male': 0, 'female': 1})
data1.to_csv('data/csv_para_previsao.csv', index=False)