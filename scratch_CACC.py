from CAVSimulator0910 import Simulator
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

##### Controller HYPERPARAMETERS FOR TUNING

start_from_init = True
num_leading_vehicle = 3
num_following_vehicle = 0
num_eps = 1000

print("Controller Hyperparameters")
print()
print("#"*30)
######

data_t = []
data_d = []
start_disp = None

def add2loc_map(env):
    t = deepcopy(env.t)
    d = deepcopy(env.current_disps)
    data_t.append(t)
    data_d.append(d)

def create_loc_map(env):
    plt.title("Location Graph")

    for n in range(env.num_vehicles):
        if(n < env.num_leading_cars):
            plt.plot(np.array(data_d)[:,n] + start_disp[n*3 + 1], color='b')
        elif(n == env.num_leading_cars):
            plt.plot(np.array(data_d)[:,n] + start_disp[n*3 + 1],"g")
        else:
            plt.plot(np.array(data_d)[:,n] + start_disp[n*3 + 1],"r")
    plt.ylabel("Location")
    plt.xlabel("Time")
    plt.show()

# CAV Simulator (Generates Fake Data now)
env = Simulator(num_leading_vehicle,num_following_vehicle)
env.normalize = False
#env.verbose = True
num_episodes = num_eps
rewards = []

for i in range(num_episodes):
    #
    data_t = []
    data_d = []
    start_disp = None
    #
    s = env.reset()
    #
    env.normalize = True
    start_disp = env.center_state(env.current_states[0])
    env.normalize = False
    #

    done = 0
    i = 0
    reward = None
    while not done:
        #print(env.t)
        #env.render()
        # For graph
        add2loc_map(env)
        #print(s)
        v, x, a = env.CACC(s,env.num_leading_cars)
        #print(v)
        #a = min(-2,max(a,2))
        s, reward, done, info = env.step(a,controller="CACC")
        #print(reward)
        #print()
        i = i + 1

    print(reward)
    rewards.append(reward)
    print(i)
print("CACC")
print("Average Reward:",np.mean(rewards))
print("Median Reward:",np.median(rewards))
print("SE Reward:",np.std(rewards)/(len(rewards))**0.5)
#print(rewards)
plt.hist(rewards, bins='auto')
plt.show()

# For graph
create_loc_map(env)

plt.title("Controller Rewards")
plt.plot(rewards,"r")
plt.ylabel("Reward")
plt.xlabel("Episode")
plt.show()
quit()

