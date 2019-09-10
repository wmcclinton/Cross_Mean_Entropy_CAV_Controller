from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat('veh_combined.mat')['veh_combined']
# time_index, location, distance to prior vehicle, velocity, veocity difference to prior vehicle, acceleration
#-300 - 6000 location

for i in range(12):
    d = []
    for car_time in data:
        #for car in car_time:
        d.append(car_time[i][2])

    plt.plot(d,"r")
    plt.show()

quit()

s = []
t = 0
cav_pos = data[t][0:3][1][1]
for car in data[t][0:3]:
    #print(car)
    v = car[3]
    x = car[1]
    a = car[5]
    s = s + [v,x,a]
    
print(s)
quit()