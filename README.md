
# 🔥 Predição de Calorias Queimadas com CatBoost

Este projeto tem como objetivo prever a quantidade de **calorias queimadas** durante atividades físicas com base em características fisiológicas e dados da atividade. O modelo utiliza **CatBoostRegressor** com validação cruzada (K-Fold) e engenharia de atributos para melhorar a performance.

[Link da Competição](https://www.kaggle.com/competitions/playground-series-s5e5/overview)

Arquivos utilizados:

- `train.csv`: Conjunto de treino com as colunas `id`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`, `Sex`, `Calories`.
- `test.csv`: Conjunto de teste com as mesmas colunas, exceto `Calories`.
- `modelo1.py, modelo4.py, modelo3.py, modelo4.py`: Diferentes versões, utilizando diferentes abordagens para a resolução do problema

## ⚙️ Funcionalidades dos Scripts

- Carregamento dos dados de treino e teste.
- Engenharia de atributos combinando colunas numéricas com multiplicações cruzadas.
- Codificação da variável categórica `Sex` para uso com o CatBoost.
- Treinamento de um modelo `CatBoostRegressor` com validação cruzada (`KFold`).
- Avaliação com a métrica RMSE (Root Mean Squared Error).
- Aplicação de transformação logarítmica nos targets para melhorar a estabilidade do modelo.
- Geração de previsões no conjunto de teste.
- Criação do arquivo `submission002.csv` pronto para submissão.

## 🔍 Engenharia de Atributos

Foram adicionadas várias colunas derivadas da multiplicação entre pares de atributos numéricos, como por exemplo:

- `Age_Height` = `Age` × `Height`
- `Weight_Duration` = `Weight` × `Duration`
- `Heart_Rate_Body_Temp` = `Heart_Rate` × `Body_Temp` (opcional)

Essas features ajudam o modelo a capturar interações não-lineares entre os dados.

## 🧠 Modelo: CatBoostRegressor

Parâmetros principais:

```python
iterations=3000
learning_rate=0.02
depth=10
l2_leaf_reg=3
subsample=0.9
colsample_bylevel=0.75
early_stopping_rounds=30
loss_function='RMSE'
```

## 📊 Validação

A validação é feita com **K-Fold Cross-Validation**, usando 10 folds em uma das versões e 3 folds em outra. O RMSE de cada fold é impresso no console, além do RMSE final médio do modelo.

## 📤 Submissão

As previsões são revertidas da transformação logarítmica (`np.expm1`) e salvas no formato:

| id  | Calories |
|-----|----------|
| 1   | 125.6    |
| 2   | 330.4    |
| ... | ...      |

## 🧪 Como executar

Certifique-se de instalar as dependências:

```bash
pip install pandas numpy scikit-learn catboost
```

E então execute o script:

```bash
python nome_do_script.py
```
