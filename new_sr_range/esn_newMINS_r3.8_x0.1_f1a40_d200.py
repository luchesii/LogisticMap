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

def find_sr(n,sr,data,fut):

    if sr<0.01:
        sr_lim = sr + 0.0008
        sr = sr - 0.0008
        c = 1
    elif sr<0.1:
        sr_lim = sr + 0.008
        sr = sr - 0.008
        c = 2
    else:
        sr_lim = sr + 0.08
        sr = sr - 0.08
        c = 3

    RMSE = []
    while sr < sr_lim:

        esn = ESN(n_inputs = 1,
              n_outputs = 1,
              n_reservoir = int(n),
              spectral_radius = sr,
              random_state=42)
        trainlen = 200
        future = fut
        pred_training = esn.fit(np.ones(trainlen),data[0:trainlen])


        prediction = esn.predict(np.ones(future))
        rmse = np.sqrt(np.mean((prediction.flatten() - data[trainlen:trainlen+future])**2))
        RMSE.append(sr)
        RMSE.append(rmse)

        if c==1:
            sr = sr + 0.0001
        elif c==2:
            sr = sr + 0.001
        else:
            sr = sr + 0.01

    RMSE = np.asarray(RMSE).reshape((int(len(RMSE)/2),2))

    i_min = RMSE[:,1].argmin()


    sr_min = RMSE[i_min,0]
    erro_min = RMSE[i_min,1]
    return sr_min,erro_min


list_of_x0 = np.loadtxt("esn_random_x0.1_r3.8.csv", delimiter=" ")

menor = np.zeros((5))

#para saber quantos ja tem salvos
file_tocount = np.loadtxt('esn_new_minstable_r3.8_x0.1_f1a40_d200.csv',delimiter=",")
file_siz = int(file_tocount.shape[0]/40)

print(file_siz)
c = 0

for ii in range(file_siz,10000):
    if c==0: file_siz2 = int(file_tocount.shape[0]%40)+1
    else: file_siz2 = 1

    for fut in range(file_siz2,41):
        menor[4] = fut
        data = get_data(list_of_x0[ii])
        erros2 = []
        n = 1

        while n<2000:
            sr = 0.001
            errors = []

            while sr < 1:
                err = []
                esn = ESN(n_inputs = 1,
                  n_outputs = 1,
                  n_reservoir = n,
                  spectral_radius = sr,
                  random_state=42)
                trainlen = 200
                future = fut
                pred_training = esn.fit(np.ones(trainlen),data[0:trainlen])


                prediction = esn.predict(np.ones(future))
                error = np.sqrt(np.mean((prediction.flatten() - data[trainlen:trainlen+future])**2))

                errors.append(sr)
                errors.append(error)


                if sr<0.01:
                    sr = sr + 0.003
                elif sr<0.01:
                    sr = sr + 0.015
                else:
                    sr = sr + 0.15

            erros = np.asarray(errors).reshape((int(len(errors)/2),2))

            i_min = erros[:,1].argmin()


            erros2.append(n)
            erros2.append(erros[i_min,0])
            erros2.append(erros[i_min,1])



            if n<100: n = n+1
            else:  n = n+100


        t = int(len(erros2)/3)
        erros2 = np.asarray(erros2).reshape(t,3)
        i_min = erros2[:,2].argmin()
        menor[0] = list_of_x0[ii]  #x0
        menor[1] = erros2[i_min,0] #n

        print(ii,erros2[i_min,0],erros2[i_min,1])
        sr_min, rmse_min = find_sr(erros2[i_min,0],erros2[i_min,1],data,fut)

        menor[2] = sr_min
        menor[3] = rmse_min

        file2 = open('esn_new_minstable_r3.8_x0.1_f1a40_d200.csv','a+')
        for j in range(menor.shape[0]):
            file2.write('{}'.format(menor[j]))
            if j<menor.shape[0]-1:
                file2.write(',')
        file2.write('\n')
        file2.close()
    c = 1
