import numpy as np
import matplotlib.pyplot as plt
import os

fed_file1 = 'Cl-10 Ep-500 Comm-10.npz'
fed_file2 = 'Cl-10 Ep-500 Comm-20.npz'
fed_file3 = 'Cl-10 Ep-500 Comm-25.npz'
non_fed_file = 'Non-Fed Ep-500.npz'
ista_file = 'nmse_ista.npz'

Comm = np.arange(0,11,1)
inp1 = np.load(fed_file1)['arr_0']
plt.plot(Comm, inp1, label="clients = 10 & C=10")
inp2 = np.load(fed_file2)['arr_0']
plt.plot(Comm, inp2, label="clients = 10 & C=20")
inp3 = np.load(fed_file3)['arr_0']
plt.plot(Comm, inp3, label="clients = 10 & C=25")
inp4 = np.load(non_fed_file)['arr_0']
plt.plot(Comm, inp4, label="LISTA")
inp5 = np.load(ista_file)['arr_0']
plt.plot(Comm, inp5, label="ISTA")
plt.xlabel('Num of Layers L')
plt.ylabel('NMSE (dB)')
plt.legend()
plt.show()