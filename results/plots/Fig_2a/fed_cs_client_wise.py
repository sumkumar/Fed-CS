import numpy as np
import matplotlib.pyplot as plt
import os

for i in range(1,1001):
    val = {}
    loss = []
    flag = False
    for j in range(1,20):
        file_name = 'lnmse_new_global'+str(j)+'_'+str(i)+'.npz'
        if os.path.exists(file_name):
            flag = True
            inp = np.load(file_name)['arr_0']
            val[j]=inp[10]
            loss.append(inp[10])
    if flag ==True:
        x = list(val.keys())
        plt.xlabel('Num of clients')
        plt.ylabel('NMSE (dB)')
        plt.plot(x, loss, label="Epochs = "+str(i))
        plt.legend()