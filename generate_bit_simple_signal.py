import numpy as np
import torch
from scipy import signal

def bit_signal(t, f):
    #return signal.square(2*np.pi*f*t, duty=0.5).astype('float64')
    return signal.square(t).astype('float64')

def bit_signal_simple(t):
    return signal.square(t, duty=0.5).astype('float64')    
    #return signal.square(t).astype('float64')

np.random.seed(2)

COEF = 10
T = 4 #int(16/COEF) #need to check
L = 100*COEF
N = 100
MAX = 20*COEF

x = np.empty((N, L), 'int64')
#x[:] = np.array(range(0, L, 0.1)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
x[:] = np.array(np.linspace(0, MAX, L)) + np.random.randint(-1 * T, 1 * T, N).reshape(N, 1)
#data = bit_signal_simple(t=x, unit_length=10, upbit_points=[1,3])
data = bit_signal_simple(t=x)
#data = bit_signal_rect(t=x, lst=[1,4])
torch.save(data, open('traindata_bit_simple.pt', 'wb'))

import matplotlib.pyplot as plt
#print(data[0])
plt.plot(data[1])
plt.show()
# plt.plot(data[2])
# plt.show()

