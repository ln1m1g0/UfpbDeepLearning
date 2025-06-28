# Importar as bibliotecas necessárias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Para os gráficos serem renderizados no próprio Jupyter
# %matplotlib inline

# Carregando os dois DataSets que serão utilizados

df_train = pd.read_csv('titanic/train.csv')
df_test = pd.read_csv('titanic/test.csv')

""" Criando um DataFrame para entender os tipos de dados 
    e os valores missing no Dataset train"""

df_aux_train = pd.DataFrame({'Colunas': df_train.columns,
                       'Tipos': df_train.dtypes,
                       'Percentual_faltantes': df_train.isna().sum() / df_train.shape[0]})

# print(df_aux_train.head)

""" Criando um DataFrame para entender os tipos de dados 
    e os valores missing no Dataset train"""

df_aux_test = pd.DataFrame({'colunas': df_test.columns,
                            'tipos': df_test.dtypes,
                            'Percentual_faltantes': df_test.isna().sum() / df_test.shape[0]})
# print(df_aux_test.head)
print('Aqui:')
print(df_train.head())

# print(df_train.describe())
print(df_test.describe())

# df_train.hist(figsize=(12,8))
# print(df_train.hist(figsize=(12,8)))

# plt.hist(df_train['Survived'], bins=10, color='c') 

# # Adicionar os rótulos(labels)
# plt.xlabel('Notas')
# plt.ylabel('Frequencia')
# plt.title('Exemplo de Histograma')

# # Exibir o histograma
# plt.show()


# Verificar quais grupos entre a coluna 'Sex' mais tiveram sobreviventes

# female = df_train[df_train['Sex'] == 'female'][['Survived']]
female = df_train[df_train['Sex'] == 'female']['Survived']
rate_fem = sum(female)/len(female)
print('% das mulheres das sobreviveram: {}'.format(rate_fem))

male = df_train[df_train['Sex'] == 'male']['Survived']
# print(male[1:10])
rate_male = sum(male)/len(male)
print('% dos homens que sobreviveram: {}'.format(rate_male))

# # verificar a distribuição de Suvived x Sex

# sns.barplot(x='Sex', y='Survived', data=df_train)
# plt.show()

# # verificar a distribuição de Suvived x Pclass
# sns.barplot(x='Pclass', y='Survived', data=df_train)
# plt.show()

# # Plotar o gráfico de heatmap para verificarmos as relações entre as variáveis
# sns.heatmap(df_train.corr(), square = True, linewidths= .5, annot=True, fmt='.2f')
# plt.show()


# from pandas_profiling import ProfileReport
# profile = ProfileReport(df_train, title='Relatório - Pandas Profiling', html={'style':{'full_width':True}})
# profile


# Guardando a variável 'PassengerId' (esqueci na primeira vez rsrs)
passengerId = df_test['PassengerId']


# Dropando as variáveis dos dois Datasets
df_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

"""Verificando a quantidade de dados nulos em cada feature
do Dataset df_train"""
print(df_train.isna().sum())

"""Verificando a quantidade de dados nulos em cada feature
do Dataset df_test"""
print(df_test.isna().sum())

# 'Age'
# df_train['Age'].fillna(df_train['Age'].median(), inplace=True)
# df_test['Age'].fillna(df_test['Age'].median(), inplace=True)

df_train['Age'].fillna(df_train['Age'].mean(), inplace=True)
df_test['Age'].fillna(df_test['Age'].mean(), inplace=True)



# print(df_train.isna().sum())
# print(df_test.isna().sum())

# 'Fare'
df_test['Fare'].fillna(df_test['Fare'].median(), inplace=True)


# 'Embarked'
df_train.dropna(axis=0, inplace=True)



# Guardando a variável 'Suvived'

target = df_train['Survived']
print('---------------Target----------------')
print(target)
# Tirando a variável do Datafram Train

df_train.drop(['Survived'], axis=1, inplace=True)

# print(df_train.isna().sum())
# print(df_test.isna().sum())

print(df_train.columns == df_test.columns)

# Importanto o LabelEncoder do Sklearn

from sklearn.preprocessing import LabelEncoder

# Utilizando o LabelEncoder

# Dataset train

df_train['Sex'] = df_train[['Sex']].apply(LabelEncoder().fit_transform)
# df_train['Embarked'] = df_train[['Embarked']].apply(LabelEncoder().fit_transform)


# Dataset test

df_test['Sex'] = df_test[['Sex']].apply(LabelEncoder().fit_transform)
# df_test['Embarked'] = df_test[['Embarked']].apply(LabelEncoder().fit_transform)

# df_copy['City'].replace('New York', 'NY', inplace=True)
df_train['Embarked'].replace('S', 7, inplace=True)
df_train['Embarked'].replace('C', 2, inplace=True)
df_train['Embarked'].replace('Q', 1, inplace=True)

df_test['Embarked'].replace('S', 7, inplace=True)
df_test['Embarked'].replace('C', 2, inplace=True)
df_test['Embarked'].replace('Q', 1, inplace=True)





# Após fazer as mudanças no Dataset Train

print(df_train.head())

# Após fazer as mudanças no Dataset Test
# print('-----------------test------------------------')
# print(df_test.head())
# print(set(df_test['Embarked']))

# Importando a biblioteca do Modelo de Machine Learning

from sklearn.linear_model import LogisticRegression

############################################# MLP ########################################
# Criando o Modelo de Rl
# lr_model = LogisticRegression(solver='liblinear')
from sklearn.neural_network import MLPClassifier
solver_ = 'adam'
lr_model = MLPClassifier(hidden_layer_sizes = (20, 30, 20),
                activation = 'relu',
                solver = 'adam', #'sgd',#'adam',
                alpha = 0.0001,
                learning_rate_init = 0.001,
                max_iter = 5000,
                tol = 1e-4,
                n_iter_no_change = 10,
                # verbose = True,
                random_state = 87)


lr_model.fit(df_train, target)

y_pred_train = lr_model.predict(df_train)
# Verificar a acurácia
# ac_lr = round(lr_model.score(df_train, target) * 100, 2)
train_acc = np.sum(y_pred_train == target)/len(target)
print("Acurácia do treino: ", train_acc)


# print("Acurácia do Modelo MLP: {}". format(train_acc))


# Criando o modelo de predição e gerando o arquivo para submissão

y_pred_lr = lr_model.predict(df_test)

submission = pd.DataFrame({"PassengerId": passengerId,
                           "Survived": y_pred_lr})

# Gerando o arquivo

submission.to_csv('submission_lr.csv', index=False)

from sklearn.metrics import confusion_matrix
import pylab as pl
cm = confusion_matrix(target, y_pred_train)
# pl.matshow(cm)
# pl.title('Confusion matrix of the classifier')
# pl.colorbar()
# pl.show()

# import pylab as pl
import seaborn as sns

pl.matshow(cm)
label = 'Matrix de Confusão - Conjunto de Treinamento'
pl.title(label)# + str(vertice_original))

sns.heatmap(cm,annot=True, xticklabels=['morreram', 'sobreviveram'], yticklabels=['morreram', 'sobreviveram'], fmt=".0f")
# sns.heatmap(cm,annot=True, fmt=".0f")

pl.xlabel('Valor Predito')
# pl.xticks(rotation=45)
pl.ylabel('Valor Real')

pl.show()

