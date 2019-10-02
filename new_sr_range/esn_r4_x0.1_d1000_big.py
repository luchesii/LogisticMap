import numpy as np
from pyESN import ESN
from matplotlib import pyplot as plt
import random

def get_data(x0):
    r = 4
    x = []
    x.append(x0)
    for i in range(1,1100):
        x1=x0*r*(1-x0)
        x0=x1
        x.append(x1)
    x=np.asarray(x)
    return(x)

def find_sr(n,sr,data):

    sr_lim = sr + 0.05
    sr = sr - 0.05
    RMSE = []
    while sr < sr_lim:

        esn = ESN(n_inputs = 1,
              n_outputs = 1,
              n_reservoir = int(n),
              spectral_radius = sr,
              random_state=42)
        trainlen = 1000
        future = 10
        pred_training = esn.fit(np.ones(trainlen),data[0:trainlen])


        prediction = esn.predict(np.ones(future))
        rmse = np.sqrt(np.mean((prediction.flatten() - data[trainlen:trainlen+future])**2))
        RMSE.append(sr)
        RMSE.append(rmse)

        sr = sr + 0.01

    RMSE = np.asarray(RMSE).reshape((int(len(RMSE)/2),2))

    i_min = RMSE[:,1].argmin()



    sr_min = RMSE[i_min,0]
    erro_min = RMSE[i_min,1]
    return sr_min,erro_min


list_of_x0 = np.loadtxt("esn_random_x0.1_r4.csv", delimiter=" ")

menor = np.zeros((4))

#para saber quantos ja tem salvos
file_tocount = np.loadtxt('esn_new_minstable_r4_x0.1_d1000.csv',delimiter=",")
file_siz = file_tocount.shape[0]
print(file_siz)


for ii in range(file_siz,10000):

    data = get_data(list_of_x0[ii])
    erros2 = []
    n = 1

    while n<2000:
        sr = 0.1
        errors = []

        while sr < 1:
            err = []
            esn = ESN(n_inputs = 1,
              n_outputs = 1,
              n_reservoir = n,
              spectral_radius = sr,
              random_state=42)
            trainlen = 1000
            future = 10
            pred_training = esn.fit(np.ones(trainlen),data[0:trainlen])


            prediction = esn.predict(np.ones(future))
            error = np.sqrt(np.mean((prediction.flatten() - data[trainlen:trainlen+future])**2))

            errors.append(sr)
            errors.append(error)


            sr = sr+ 0.1

        erros = np.asarray(errors).reshape((int(len(errors)/2),2))

        i_min = erros[:,1].argmin()


        erros2.append(n)
        erros2.append(erros[i_min,0])
        erros2.append(erros[i_min,1])



        if n<100:
            n = n+1
        else:
            n = n+100


    t = int(len(erros2)/3)
    erros2 = np.asarray(erros2).reshape(t,3)
    i_min = erros2[:,2].argmin()
    menor[0] = list_of_x0[ii]  #x0
    menor[1] = erros2[i_min,0] #n

    print(ii)
    sr_min, rmse_min = find_sr(erros2[i_min,0],erros2[i_min,1],data)

    menor[2] = sr_min
    menor[3] = rmse_min

    file2 = open('esn_new_minstable_r4_x0.1_d1000.csv','a+')
    for j in range(menor.shape[0]):
        file2.write('{}'.format(menor[j]))
        if j<menor.shape[0]-1:
            file2.write(',')
    file2.write('\n')
    file2.close()
