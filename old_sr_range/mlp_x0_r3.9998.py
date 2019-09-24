import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt

np.random.seed(10)

def get_next(x0,fut):
    r = 3.9998
    for i in range(fut):
        x1=x0*r*(1-x0)
        x0=x1
    return(x1)

x0 = np.loadtxt("esn_random_x0.1_r3.9998.csv", delimiter=" ")

xn = np.empty((x0.shape[0]))

for future in range(1,5):

    for i in range(x0.shape[0]):
        xn[i] = get_next(x0[i],future)

    x0train,x0test,xntrain,xntest=train_test_split(x0,xn,train_size=0.7,random_state=1)

    n=5

    MSE = []
    MAE = []
    N = []

    while n<=4000:

        model = Sequential()
        model.add(Dense(n, input_shape=(1,), activation='relu'))
        model.add(Dense(1, activation='linear'))

        adam=keras.optimizers.Adam()
        model.compile(optimizer=adam, loss='mean_squared_error')

        print("Nº neurônios:", n,"\n")
        history = model.fit(x0train, xntrain, epochs=100,verbose=0)
        mse = model.evaluate(x0test, xntest)
        msetrain = model.evaluate(x0train, xntrain)

        faprox = model.predict(x0test).T
        mae = np.sum(np.abs(faprox-xntest))/xntest.shape[0]
        print("mse",mse,"mse train",msetrain,"mae",mae)

        N.append(n)
        MSE.append(mse)
        MAE.append(mae)

        if n < 50: n+=5
        elif n < 100: n+=10
        else: n+=100

    RMSE=np.sqrt(MSE)

    if future ==1: file = open('mlp_x1_r3.9_minstable.csv','a+')
    elif future ==2: file = open('mlp_x2_r3.9_minstable.csv','a+')
    elif future ==3: file = open('mlp_x3_r3.9_minstable.csv','a+')
    elif future ==4: file = open('mlp_x4_r3.9_minstable.csv','a+')

    for j in range(len(N)):
        file.write('{}'.format(N[j]))
        file.write(',')
        file.write('{}'.format(MAE[j]))
        file.write(',')
        file.write('{}'.format(MSE[j]))
        file.write(',')
        file.write('{}'.format(RMSE[j]))
    file.write('\n')
    file.close()

    n_min = N[np.argmin(RMSE)]

    model = Sequential()
    model.add(Dense(n_min, input_shape=(1,), activation='relu'))
    model.add(Dense(1, activation='linear'))

    adam=keras.optimizers.Adam()
    model.compile(optimizer=adam, loss='mean_squared_error')

    history = model.fit(x0train, xntrain, epochs=100,verbose=0)
    mse = model.evaluate(x0test, xntest)
    print(N[np.argmin(RMSE)],mse)

    faprox = model.predict(x0test).T
    ae = (faprox-xntest)
    ae = np.abs(ae)
    se = ae**2

    if future==1: file2 = open('mlp_x1_r3.9_x0_ae_se.csv','a+')
    elif future==2: file2 = open('mlp_x2_r3.9_x0_ae_se.csv','a+')
    elif future==3: file2 = open('mlp_x3_r3.9_x0_ae_se.csv','a+')
    elif future==4: file2 = open('mlp_x4_r3.9_x0_ae_se.csv','a+')

    for j in range(len(N)):
        file2.write('{}'.format(x0[j]))
        file2.write(',')
        file2.write('{}'.format(ae[0][j]))
        file2.write(',')
        file2.write('{}'.format(se[0][j]))
    file2.write('\n')
    file2.close()
