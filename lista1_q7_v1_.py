import os
import requests
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def download_dataset(url, filename):
    """
    Baixa um arquivo do URL fornecido.
    """
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

# URL para o conjunto de dados CIFAR-10
cifar_10_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# Diretório para salvar os arquivos
cifar_10_dir = './cifar-10/'

# Criar diretório se não existir
os.makedirs(cifar_10_dir, exist_ok=True)

# Baixar e extrair CIFAR-10
cifar_10_filename = os.path.join(cifar_10_dir, 'cifar-10-python.tar.gz')
if not os.path.exists(os.path.join(cifar_10_dir, 'cifar-10-batches-py')):
    print("Baixando CIFAR-10...")
    download_dataset(cifar_10_url, cifar_10_filename)
    with tarfile.open(cifar_10_filename, 'r:gz') as tar:
        tar.extractall(cifar_10_dir)
    print("CIFAR-10 baixado e extraído com sucesso.")
else:
    print("CIFAR-10 já foi baixado e extraído.")

# Carregar o dataset CIFAR-10
def load_cifar_10_batch(filename):
    """
    Carrega um lote do conjunto de dados CIFAR-10.
    """
    with open(filename, 'rb') as f:
        batch = np.load(f, encoding='bytes', allow_pickle=True)
        data = batch[b'data']
        labels = batch[b'labels']
        return data, labels

def load_cifar_10():
    """
    Carrega todo o conjunto de dados CIFAR-10.
    """
    x_train = []
    y_train = []
    for i in range(1, 6):
        data, labels = load_cifar_10_batch(os.path.join(cifar_10_dir, 'cifar-10-batches-py', f'data_batch_{i}'))
        x_train.append(data)
        y_train.append(labels)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    with open(os.path.join(cifar_10_dir, 'cifar-10-batches-py', 'test_batch'), 'rb') as f:
        test_batch = np.load(f, encoding='bytes', allow_pickle=True)
        x_test = test_batch[b'data']
        y_test = test_batch[b'labels']
    return (x_train, y_train), (x_test, y_test)

# Normalizar os dados
def normalize_data(x):
    """
    Normaliza os dados.
    """
    return x.astype('float32') / 255.0

# Nome das classes para CIFAR-10
class_names_cifar_10 = ['avião', 'automóvel', 'pássaro', 'gato', 'veado', 'cão', 'rã', 'cavalo', 'navio', 'caminhão']

# Carregar conjunto de dados CIFAR-10
(x_train_cifar_10, y_train_cifar_10), (x_test_cifar_10, y_test_cifar_10) = load_cifar_10()

# Normalizar os dados
x_train_cifar_10 = normalize_data(x_train_cifar_10)
x_test_cifar_10 = normalize_data(x_test_cifar_10)

# Redimensionar os dados para a forma correta
x_train_cifar_10 = x_train_cifar_10.reshape(-1, 32, 32, 3)
x_test_cifar_10 = x_test_cifar_10.reshape(-1, 32, 32, 3)

# Converter rótulos para one-hot encoding
y_train_cifar_10 = tf.keras.utils.to_categorical(y_train_cifar_10, len(class_names_cifar_10))
y_test_cifar_10 = tf.keras.utils.to_categorical(y_test_cifar_10, len(class_names_cifar_10))

# Construção do modelo
model_cifar_10 = Sequential()

# Camadas de convolução e pooling
model_cifar_10.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model_cifar_10.add(MaxPooling2D(pool_size=(2, 2)))
model_cifar_10.add(Dropout(0.25))

model_cifar_10.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model_cifar_10.add(MaxPooling2D(pool_size=(2, 2)))
model_cifar_10.add(Dropout(0.25))

model_cifar_10.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model_cifar_10.add(MaxPooling2D(pool_size=(2, 2)))
model_cifar_10.add(Dropout(0.25))

# Camada totalmente conectada
model_cifar_10.add(Flatten())
model_cifar_10.add(Dense(512, activation='relu'))
model_cifar_10.add(Dropout(0.5))
model_cifar_10.add(Dense(len(class_names_cifar_10), activation='softmax'))

# Compilar o modelo
model_cifar_10.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history_cifar_10 = model_cifar_10.fit(x_train_cifar_10, y_train_cifar_10, batch_size=16, epochs=25, validation_split=0.2)  # as epocas podem ser trocado para um número menor para dimiuir o tempo da compilação

# Plotar a curva de erro médio
plt.figure(figsize=(12, 4))
plt.plot(history_cifar_10.history['loss'], label='Training Loss')
plt.plot(history_cifar_10.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Prever as classes no conjunto de teste
y_pred_cifar_10 = model_cifar_10.predict(x_test_cifar_10)
y_pred_classes_cifar_10 = np.argmax(y_pred_cifar_10, axis=1)
y_true_cifar_10 = np.argmax(y_test_cifar_10, axis=1)

# Calcular a matriz de confusão
cm_cifar_10 = confusion_matrix(y_true_cifar_10, y_pred_classes_cifar_10)
disp_cifar_10 = ConfusionMatrixDisplay(confusion_matrix=cm_cifar_10, display_labels=class_names_cifar_10)

# Plotar a matriz de confusão
plt.figure(figsize=(10, 10))
disp_cifar_10.plot(cmap=plt.cm.Blues, values_format='d')
plt.xticks(rotation=45)
plt.title('Confusion Matrix')
plt.show()

num_amostras_treinamento = x_train_cifar_10.shape[0]
print("Número de amostras nos dados de treinamento:", num_amostras_treinamento)