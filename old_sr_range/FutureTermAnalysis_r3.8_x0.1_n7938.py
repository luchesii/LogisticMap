import numpy as np
from pyESN import ESN
from matplotlib import pyplot as plt
import random

def get_data(x0):
    r = 3.8
    x = []
    x.append(x0)
    for i in range(1,1100):
        x1=x0*r*(1-x0)
        x0=x1
        x.append(x1)
    x=np.asarray(x)
    return(x)

menor = np.loadtxt("esn_big_minstable_r3.8_x0.1_d1000.csv", delimiter=",")
list_of_x0 = np.loadtxt("esn_random_x0.1_r3.8.csv", delimiter=" ")


print(menor.shape)

def get_array(erro_list):
    erro_array = np.asarray(erro_list).reshape(int(len(erro_list)/5),5)
    return erro_array

def get_array2(erro_list):
    erro_array = np.asarray(erro_list).reshape(int(len(erro_list)),10)
    return erro_array

def get_errors(e0,e1,e2,e3,e4,e5,e6,e7,e8,e9):

    mae=[np.sum(np.abs(e0))/e0.shape[0],np.sum(np.abs(e1))/e1.shape[0],np.sum(np.abs(e2))/e2.shape[0],
       np.sum(np.abs(e3))/e3.shape[0],np.sum(np.abs(e4))/e4.shape[0],np.sum(np.abs(e5))/e5.shape[0],
       np.sum(np.abs(e6))/e6.shape[0],np.sum(np.abs(e7,))/e7.shape[0],np.sum(np.abs(e8))/e8.shape[0],
       np.sum(np.abs(e9))/e9.shape[0]]

    mse=[np.sum(e0**2)/e0.shape[0],np.sum(e1**2)/e1.shape[0],np.sum(e2**2)/e2.shape[0],
       np.sum(e3**2)/e3.shape[0],np.sum(e4**2)/e4.shape[0],np.sum(e5**2)/e5.shape[0],
       np.sum(e6**2)/e6.shape[0],np.sum(e7**2)/e7.shape[0],np.sum(e8**2)/e8.shape[0],
       np.sum(e9**2)/e9.shape[0]]

    rmse=[np.sqrt(np.sum(e0**2)/e0.shape[0]),np.sqrt(np.sum(e1**2)/e1.shape[0]),np.sqrt(np.sum(e2**2)/e2.shape[0]),
       np.sqrt(np.sum(e3**2)/e3.shape[0]),np.sqrt(np.sum(e4**2)/e4.shape[0]),np.sqrt(np.sum(e5**2)/e5.shape[0]),
       np.sqrt(np.sum(e6**2)/e6.shape[0]),np.sqrt(np.sum(e7**2)/e7.shape[0]),np.sqrt(np.sum(e8**2)/e8.shape[0]),
       np.sqrt(np.sum(e9**2)/e9.shape[0])]

    return mae,mse,rmse

future_error = []
future_error_2d = []
future_error_3d = []
future_error_4d = []
future_error_34d = []

for i in range(menor.shape[0]):
    data = get_data(list_of_x0[i])

    future_t = np.zeros((10))

    esn = ESN(n_inputs = 1,
        n_outputs = 1,
      n_reservoir = int(menor[i][1]),
      spectral_radius = menor[i][2],
      random_state=42)
    trainlen = 1000
    future = 10
    pred_training = esn.fit(np.ones(trainlen),data[0:trainlen])

    prediction = esn.predict(np.ones(future))
    error = np.sqrt(np.mean((prediction.flatten() - data[trainlen:trainlen+future])**2))
    print(i,'test error:',error)

    future_error.append(prediction.flatten()-data[trainlen:trainlen+future])
    if error<0.1 and error>=0.01:
        future_error_2d.append(prediction.flatten()-data[trainlen:trainlen+future])
    elif error<0.01 and error>=0.001:
        future_error_3d.append(prediction.flatten()-data[trainlen:trainlen+future])
    elif error<0.001:
        future_error_4d.append(prediction.flatten()-data[trainlen:trainlen+future])
    if error<0.01:
        future_error_34d.append(prediction.flatten()-data[trainlen:trainlen+future])



FUTUREerror = get_array2(future_error)
FUTUREerror_2d = get_array2(future_error_2d)
FUTUREerror_3d = get_array2(future_error_3d)
FUTUREerror_4d = get_array2(future_error_4d)
FUTUREerror_34d = get_array2(future_error_34d)

mae, mse, rmse = get_errors(FUTUREerror[:,0],FUTUREerror[:,1],FUTUREerror[:,2],FUTUREerror[:,3],FUTUREerror[:,4],FUTUREerror[:,5],FUTUREerror[:,6],FUTUREerror[:,7],FUTUREerror[:,8],FUTUREerror[:,9])
print(mae,'\n',mse,'\n',rmse)
