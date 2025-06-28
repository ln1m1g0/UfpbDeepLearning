import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':

    X, y = make_blobs(n_samples=500, n_features=2, cluster_std=0.1, centers= [(0,0), (0,1), (1,0),(1,1)])
    # print(type(X))
    # print(y)
    y[y == 2] = 1
    y[y == 3] = 0

    X0 = []
    X1 = []
    y0 = []
    y1 = []
    for i, temp in enumerate(y):
        if temp==1:
            X1.append(X[i, :])
            y1.append(temp)
        else:
            X0.append(X[i, :])
            y0.append(temp)

    X0 = np.array(X0)
    X1 = np.array(X1)

    
    # plt.scatter(X[:, 0], X[:, 1], c=y, s=25, edgecolors='k')
    plt.scatter(X0[:, 0], X0[:, 1], c=y0, s=25, edgecolor='blue')
    plt.scatter(X1[:, 0], X1[:, 1], c=y1, s=25, edgecolor='red')
    plt.title("Dados de treinamento da função XOR.")
    plt.legend(["zero", "um"])
    
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=19)

    model = MLPClassifier(hidden_layer_sizes=(20, 20), learning_rate='adaptive', epsilon=0.01) #AdaGrad
    model.fit(X_train, y_train)

    #train
    y_pred_train = model.predict(X_train)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=25, edgecolors='k')
    plt.show()
    train_acc = np.sum(y_pred_train == y_train)/y_train.shape[0]
    print("train accuracy: ", train_acc)

    #test
    



    y_pred_test = model.predict(X_test)
    # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=25, edgecolors='k')
    # plt.show()
    test_acc = np.sum(y_pred_test == y_test)/y_test.shape[0]
    print("test  accuracy: ", test_acc)


    y = y_pred_test
    X = X_test
    X0 = []
    X1 = []
    y0 = []
    y1 = []
    for i, temp in enumerate(y):
        if temp==1:
            X1.append(X[i, :])
            y1.append(temp)
        else:
            X0.append(X[i, :])
            y0.append(temp)

    X0 = np.array(X0)
    X1 = np.array(X1)

    
    # plt.scatter(X[:, 0], X[:, 1], c=y, s=25, edgecolors='k')
    plt.scatter(X0[:, 0], X0[:, 1], c=y0, s=25, edgecolor='blue')
    plt.scatter(X1[:, 0], X1[:, 1], c=y1, s=25, edgecolor='red')
    plt.title("Resultado do teste da aproximação da função XOR.")
    plt.legend(["zero", "um"])
    
    plt.show()