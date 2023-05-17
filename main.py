import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Base de Dados SPAMbase
csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv'

# Recuperando os dados da URI
clinical_records = pd.read_csv(csv_url)
clinical_records.columns
clinical_records.shape

#Informações sobre o Dataframe
clinical_records.info()

# Visualizar as colunas do DataFrame
clinical_records.columns

# Input dos dados
dados_da_base = clinical_records.drop('DEATH_EVENT', axis=1)
dados_da_base = np.array(dados_da_base, dtype=int)

# Output dos dados
objetivo = clinical_records['DEATH_EVENT']
objetivo = np.array(objetivo, dtype=int)

# Dividindo os dados para treinamento e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(dados_da_base, objetivo, test_size=0.2, random_state=4)
x_treino = normalize(x_treino)
x_teste = normalize(x_teste)

# Algoritmo de Naive Bayes
classificador = GaussianNB()
classificador.fit(x_treino, y_treino)
y_predict  =  classificador.predict(x_teste)

precisao = accuracy_score(y_teste,y_predict)
print("Precisão da previsão: " ,precisao)