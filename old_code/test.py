from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from random import randint

data = loadmat('veh_combined.mat')['veh_combined']

t_start = randint(20,len(data)-300-1)
#t_start = 4801

states = []
for i in range(301):
    s = []
    for car in data[t_start+i][0:7]:
        v = car[3]
        x = car[1]
        a = car[5]
        #s = s + [v,x,a]
        s = s + [a]
    states.append(s)

run = np.array([s[3] for s in states])
print(run)
print(t_start)
np.flip(run)
plt.title("Acceleration During Episode [BAD]")
plt.plot(run,"r")
plt.ylabel("Acceleration")
plt.xlabel("Time")
plt.show()