import numpy as np
import random

r = 4
x0=0.1

n=10000 #número de dados no dataset


file = open('esn_data10000_x0.1_r4.csv','w+')

for i in range(200):
    x1=x0*r*(1-x0)
    x0=x1

x=[]
for i in range(n*10):
    x1=x0*r*(1-x0)
    x0=x1
    x.append(x1)

x=np.asarray(x)

for i in range(n):
    file.write('{}'.format(x[i]))
    file.write('\n')

file.close()
