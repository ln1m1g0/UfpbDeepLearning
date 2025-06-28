import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import Perceptron


vertices_originais = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 1],
])
#print((vertices_originais[0,:]==vertices_originais[1,:]).all())
raio_ruido = 0.1
num_pontos_ruido = 1000

vertices_com_ruido = []
vertices_sem_ruido = []
rotulo = []
for vertice_original in vertices_originais:
    for _ in range(num_pontos_ruido):
        ruido = np.random.rand(3) * raio_ruido - raio_ruido / 2
        vertice_com_ruido = vertice_original + ruido
        vertices_com_ruido.append(vertice_com_ruido)
        vertices_sem_ruido.append(vertice_original)
        if (vertice_original==vertices_originais[7,:]).all():
            rotulo.append(1)
        else:
            rotulo.append(-1)


vertices_com_ruido = np.array(vertices_com_ruido)
vertices_sem_ruido = np.array(vertices_sem_ruido)
rotulo = np.array(rotulo)
print(vertices_com_ruido.shape)
print(rotulo.shape)
print(vertices_com_ruido[500:505, :])
print(vertices_sem_ruido[500:505, :])
print(rotulo[500:505])

w = np.random.rand(3)

# Criando DataFrame com dados e rótulos
data = {
  "train": list(vertices_com_ruido),
  "label": rotulo
}


#df_dados = pd.DataFrame({'coordenadas': dados.tolist(), 'rotulo': rotulo})
df = pd.DataFrame(data)
print(df.head())
df = shuffle(df)
print(df.head())

X = np.array([np.array(x) for x in df['train']])
y = np.array([np.array(x) for x in df['label']])

print(X[0:5,:])
#
print('---------') 
print(y[0:5])


X_treino, X_validacao, y_treino, y_validacao = train_test_split(X, y, test_size=0.2, random_state=42)
# Definição e Treinamento do Perceptron
#perceptron = Perceptron(max_iter=len(rotulo), eta0 = 0.01, tol=0.009, random_state=42)
perceptron = Perceptron(max_iter=len(rotulo), eta0 = 0.01, random_state=42)
perceptron.fit(X_treino, y_treino)

# Avaliação do Perceptron
y_pred = perceptron.predict(X_validacao)
acuracia = accuracy_score(y_validacao, y_pred)
matriz_confusao = confusion_matrix(y_validacao, y_pred)

print("Acurácia:", acuracia)
print("Matriz de Confusão:")
print(matriz_confusao)