import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':

    # X, y = make_blobs(n_samples=500, n_features=2, cluster_std=0.1, centers= [(0,0), (0,1), (1,0),(1,1)])
    X, y = make_blobs(n_samples=20000, n_features=100, cluster_std=1, centers= [(0,0)])
    # print(X)
    print(set(y))
    c0 = np.array(range(1001)) - 500
    c0= c0/500
    c1 = []
    for temp in c0:
        c1.append((1 - temp**2)**0.5)

    c1 = np.array(c1)
    # print(c0[0:10])
    # print(c1[0:10])
    # plt.plot(c0, c1, 'k')
    # plt.plot(c0, -c1, 'k')
    # plt.show()



    # y[y == 2] = 1
    # y[y == 3] = 0

    X0 = []
    X_out = []
    y0 = []
    y_out = []
    for i, temp in enumerate(y):
        if (X[i, 0]**2 + X[i, 1]**2)>1:
            X_out.append(X[i, :])
            y_out.append(0)
        else:
            X0.append(X[i, :])
            y0.append(1)


    X0 = np.array(X0)
    print('Tamanho da base de dados: ', X0.shape)

    X_out = np.array(X_out)


    Xr = [] #right side
    Xl = [] #left side
    yr = []
    yl = []


    for i, temp in enumerate(y0):        
        if X0[i, 0]>0:
            Xr.append(X0[i, :])
            yr.append(1)
        else:
            Xl.append(X0[i, :])
            yl.append(3)
    
    Xr = np.array(Xr)
    Xl = np.array(Xl)

    Xru = [] #right upper side
    Xrd = [] #right down side
    yru = []
    yrd = []

    for i, temp in enumerate(yr):
        if Xr[i, 1]>0:
            Xru.append(Xr[i, :])
            yru.append(1)
        else:
            Xrd.append(Xr[i, :])
            yrd.append(2)        

    Xru = np.array(Xru)
    Xrd = np.array(Xrd)

    Xlu = [] #left upper side
    Xld = [] #left down side
    ylu = []
    yld = []

    for i, temp in enumerate(yl):
        if Xl[i, 1]>0:
            Xlu.append(Xl[i, :])
            ylu.append(1)
        else:
            Xld.append(Xl[i, :])
            yld.append(2)        

    Xlu = np.array(Xlu)
    Xld = np.array(Xld)

    
    Xruu = [] #right upper side above losango
    Xrud = [] #right upper side under losango
    yruu = [] # 1
    yrud = [] # 2

    for i, temp in enumerate(yru):
        if Xru[i, 0] + Xru[i, 1] > 1:
            Xruu.append(Xru[i, :])
            yruu.append(1)
        else:
            Xrud.append(Xru[i, :])
            yrud.append(2)

    Xruu = np.array(Xruu)
    Xrud = np.array(Xrud)

    Xrdu = [] #right down side above losango
    Xrdd = [] #right down side under losango
    yrdu = []
    yrdd = []

    for i, temp in enumerate(yrd):
        if -Xrd[i, 0] + Xrd[i, 1] > -1:
            Xrdu.append(Xrd[i, :])
            yrdu.append(3)
        else:
            Xrdd.append(Xrd[i, :])
            yrdd.append(4)

    # Xruu = np.array(Xruu)
    # Xrud = np.array(Xrud)

    Xrdu = np.array(Xrdu)
    Xrdd = np.array(Xrdd)



    Xluu = [] #left upper side above losango
    Xlud = [] #left upper side under losango
    yluu = []
    ylud = []

    for i, temp in enumerate(ylu):
        if -Xlu[i, 0] + Xlu[i, 1] > 1:
            Xluu.append(Xlu[i, :])
            yluu.append(5)
        else:
            Xlud.append(Xlu[i, :])
            ylud.append(6)

    Xluu = np.array(Xluu)
    Xlud = np.array(Xlud)
    

    Xldu = [] #left down side above losango
    Xldd = [] #left down side under losango
    yldu = []
    yldd = []

    for i, temp in enumerate(yld):
        if Xld[i, 0] + Xld[i, 1] > -1:
            Xldu.append(Xld[i, :])
            yldu.append(7)
        else:
            Xldd.append(Xld[i, :])
            yldd.append(8)

    

    Xldu = np.array(Xldu)
    Xldd = np.array(Xldd)





    # plt.scatter(X[:, 0], X[:, 1], c=y, s=25, edgecolors='blue')
    # plt.plot(c0, c1, 'k', linewidth=4)
    # plt.plot(c0, -c1, 'k', linewidth=4)
    # plt.scatter(X0[:, 0], X0[:, 1], c=y0, s=25, edgecolor='blue')
    plt.scatter(Xruu[:, 0], Xruu[:, 1], c=yruu, s=25, edgecolor='blue')
    plt.scatter(Xrud[:, 0], Xrud[:, 1], c=yrud, s=25, edgecolor='tab:orange')

    plt.scatter(Xrdu[:, 0], Xrdu[:, 1], c=yrdu, s=25, edgecolor='cyan')
    plt.scatter(Xrdd[:, 0], Xrdd[:, 1], c=yrdd, s=25, edgecolor='tab:brown')
    # plt.scatter(Xl[:, 0], Xl[:, 1], c=yl, s=25, edgecolor='yellow')

    # plt.scatter(Xlu[:, 0], Xlu[:, 1], c=ylu, s=25, edgecolor='yellow')
    plt.scatter(Xluu[:, 0], Xluu[:, 1], c=yluu, s=25, edgecolor='tab:pink')
    plt.scatter(Xlud[:, 0], Xlud[:, 1], c=ylud, s=25, edgecolor='yellow')
   
    # plt.scatter(Xld[:, 0], Xld[:, 1], c=yld, s=25, edgecolor='green')
    plt.scatter(Xldu[:, 0], Xldu[:, 1], c=yldu, s=25, edgecolor='tab:purple')
    plt.scatter(Xldd[:, 0], Xldd[:, 1], c=yldd, s=25, edgecolor='tab:gray')

    plt.scatter(X_out[:, 0], X_out[:, 1], c=y_out, s=25, edgecolor='white', linewidth=6)
    plt.title("Distribuição dos dados de treinamento.")
    # plt.legend(["Dentro do Raio", "Fora do Raio (outlier)"])
    print('yruu: ', set(yruu))
    print('yrud: ', set(yrud))
    print('yrdu: ', set(yrdu))
    print('yrdd: ', set(yrdd))
    print('yluu: ', set(yluu))
    print('ylud: ', set(ylud))
    print('yldu: ', set(yldu))
    print('yldd: ', set(yldd))


    plt.show()
    
    print(y.shape)


    y = np.append(yruu, yrud, axis = 0)
    y = np.append(y, yrdu, axis = 0)
    y = np.append(y, yrdd, axis = 0)
    y = np.append(y, yluu, axis = 0)
    y = np.append(y, ylud, axis = 0)
    y = np.append(y, yldu, axis = 0)
    y = np.append(y, yldd, axis = 0)


    X = np.append(Xruu, Xrud, axis = 0)
    X = np.append(X, Xrdu, axis = 0)
    X = np.append(X, Xrdd, axis = 0)
    X = np.append(X, Xluu, axis = 0)
    X = np.append(X, Xlud, axis = 0)
    X = np.append(X, Xldu, axis = 0)
    X = np.append(X, Xldd, axis = 0)


    # print(X.shape)  
    # print(X[0:5])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=19)

    # print(Xruu[0, :])
    print(X_train[0])
    print(y_train[0])
    solver_ = 'adam'
    model = MLPClassifier(hidden_layer_sizes = (20, 3),
                    activation = 'relu',
                    solver = solver_,#'sgd',#'adam',
                    alpha = 0.0001,
                    learning_rate_init = 0.001,
                    max_iter = 5000,
                    tol = 1e-4,
                    n_iter_no_change = 10,
                    # verbose = True,
                    random_state = 87)
        
    # hidden_layer_sizes=(20, 3), learning_rate='adaptive', epsilon=0.01, max_iter=1000)
    model.fit(X_train, y_train)

    # #train
    y_pred_train = model.predict(X_train)
    y_pt = []
    
    
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred_train, s=25, edgecolors='k')
    plt.title("Predição com os dados de treinamento.")
    # # plt.scatter(X_train[:, 0], X_train[:, 1], s=25, edgecolors='k')
    plt.show()


    train_acc = np.sum(y_pred_train == y_train)/y_train.shape[0]
    print("train accuracy: ", train_acc)

    from sklearn.metrics import confusion_matrix
    import pylab as pl

    pred = model.predict(X_test)

    plt.scatter(X_test[:, 0], X_test[:, 1], c=pred, s=25, edgecolors='k')
    # plt.scatter(X_test[:, 0], X_test[:, 1], c=pred, s=25)

    plt.title("Predição com os dados de teste.")
    # # plt.scatter(X_train[:, 0], X_train[:, 1], s=25, edgecolors='k')
    plt.show()


    cm = confusion_matrix(y_test, pred)
    # pl.matshow(cm)
    # pl.title('Confusion matrix of the classifier')
    # pl.colorbar()
    # pl.show()

    # import pylab as pl
    import seaborn as sns
    # cm = confusion_matrix(y_test, y_pred, labels=[-1,1])
    pl.matshow(cm)
    label = 'Matrix de Confusão com ADAM'
    pl.title(label)# + str(vertice_original))
    # pl.colorbar()

    # sns.heatmap(cm,annot=True, xticklabels=[-1,1], yticklabels=[-1,1], fmt=".0f")
    sns.heatmap(cm,annot=True, fmt=".0f")

    pl.xlabel('Valor Predito')
    # pl.xticks(rotation=45)
    pl.ylabel('Valor Real')
    # pl.ioff()


    pl.show()
    # print(set(y_test))

    # print(label)