import numpy as np
import matplotlib.pyplot as plt
import os

Comm = np.arange(10,110,10)
for i in range(1,1001):
    file_name = 'Layer_lnmse_10_'+str(i)+'.npz'
    if os.path.exists(file_name):
        inp = np.load(file_name)['arr_0']
        plt.plot(Comm, inp, label="Epochs = "+str(i))
plt.xlabel('Num of Comm rounds')
plt.ylabel('lnmse (dB)')
plt.legend()
plt.show()

Lay = np.arange(1,11,1)
for i in range(1,1001):
    file_name = 'Layer_lnmse_10_'+str(i)+'.npz'
    if os.path.exists(file_name):
        inp = np.load(file_name)['arr_0']
        plt.plot(Lay, inp, label="Epochs = "+str(i))
plt.xlabel('Num of Layers')
plt.ylabel('lnmse (dB)')
plt.legend()
plt.show()