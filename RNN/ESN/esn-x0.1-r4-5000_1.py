import numpy as np
from pyESN import ESN
from matplotlib import pyplot as plt
#matplotlib inline

data = np.loadtxt("esn_data10000_x0.1_r4.csv", delimiter=" ")

erros2 = []
n = 300

while n<2000:
    sr = 0.001
    errors = []
    while sr <= 0.2:
        esn = ESN(n_inputs = 1,
          n_outputs = 1,
          n_reservoir = n,
          spectral_radius = sr,
          random_state=42)
        trainlen = 5000
        future = 10
        pred_training = esn.fit(np.ones(trainlen),data[:trainlen])

        print('\n n',n,'spectral radius:',sr,'\n')
        prediction = esn.predict(np.ones(future))
        error = np.sqrt(np.mean((prediction.flatten() - data[trainlen:trainlen+future])**2))
        print('test error:',error)
        errors.append(sr)
        errors.append(error)

        plt.figure(figsize=(11,1.5))
        plt.plot(range(trainlen,trainlen+future),data[trainlen:trainlen+future],'k',label="target system")
        plt.plot(range(trainlen,trainlen+future),prediction,'r', label="free running ESN")
        lo,hi = plt.ylim()
        plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
        plt.legend(loc=(0.61,1.1),fontsize='x-small')
        plt.show()

        if sr<0.01:
            if error>0.05:
                sr= sr + 0.005
            else:
                sr = sr + 0.001
        elif sr>0.05:
            sr = sr+ 0.05
        else:
            if error>0.05:
                sr = sr+0.05
            sr = sr+ 0.01

    erros = np.asarray(errors).reshape((int(len(errors)/2),2))
    plt.title('train:10 pred:10')
    plt.xlabel('spectral radius')
    plt.ylabel('mse')
    plt.plot(erros[:,0],erros[:,1])
    plt.show()

    print(erros)
    i_min = erros[:,1].argmin()

    print('Menor:', erros[i_min,0], erros[i_min,1])

    erros2.append(n)
    erros2.append(erros[i_min,0])
    erros2.append(erros[i_min,1])

    print (erros2)

    n = n+100

print (erros2,'\n')
l = int(len(erros2)/3)
erros2 = np.asarray(erros2).reshape(l,3)


file2 = open('esn_results-x0.1-r4-5000_1.csv','w+')
for i in range(l):
    for j in range(3):
        file2.write('{}'.format(erros2[i,j]))
        file2.write(',')
    file2.write('\n')


file2.close()
