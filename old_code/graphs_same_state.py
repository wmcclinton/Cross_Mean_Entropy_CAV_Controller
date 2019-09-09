from CAVSimulator import Simulator
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import gym
from gym import wrappers
from collections import deque
import torch
import torch.nn as nn

from collections import deque
from random import randint
from scipy.io import loadmat

data = loadmat('veh_combined.mat')['veh_combined']

window_size = 1
window = deque(maxlen=window_size)
for i in range(window_size):
    window.appendleft(None)

def deque2state(env):
    state = window[0]
    d_t = 1
    for i in range(1, window_size):
        if window[i] is None:
            # on start get data directly
            #window[i] = torch.zeros(len(window[0]))
            window[i] = torch.Tensor(np.array(env.center_state(env.get_state(env.t_start - d_t))))
            d_t = d_t + 1
        state = torch.cat([state,window[i]])
    return state

class Agent(nn.Module):
    def __init__(self,
                 env: gym.wrappers.time_limit.TimeLimit,
                 hidden_size: int = 16) -> None:
        super(Agent, self).__init__()
        self.env = env
        self.space_size = env.observation_space.shape[0] * window_size
        self.hidden_size = hidden_size
        self.action_size = env.action_space.n

        self.fc1 = nn.Linear(self.space_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.action_size)

        self.eval_long = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.fc1(x))
        out = torch.tanh(self.fc2(out)) # try removing
        out = torch.softmax(out, dim=0)
        return out

    def set_weights(self, weights: np.ndarray) -> None:
        s = self.space_size
        h = self.hidden_size
        a = self.action_size
        # separate the weights for each layer
        fc1_end = (s * h) + h
        fc1_W = torch.from_numpy(weights[:s * h].reshape(s, h))
        fc1_b = torch.from_numpy(weights[s * h:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(h*a)].reshape(h, a))
        fc2_b = torch.from_numpy(weights[fc1_end+(h*a):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

    def get_weights_dim(self) -> int:
        return (self.space_size + 1) * self.hidden_size + \
            (self.hidden_size + 1) * self.action_size

data_t = []
data_d = []
start_disp = None

def add2loc_map(env):
    t = deepcopy(env.t)
    d = deepcopy(env.current_disps)
    data_t.append(t)
    data_d.append(d)
    
#######################

env = Simulator(3,3)
env.normalize = False
#env.verbose = True
num_episodes = 1
rewards = []

######################
t_start = randint(20,len(data)-env.max_t-1)
t_start = 4200
#####################

for i in range(num_episodes):
    #
    data_t = []
    data_d = []
    start_disp = None
    #
    s = env.reset_from_state(t_start)
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
        print("S")
        env.render()
        # For graph
        add2loc_map(env)
        #print(s)
        v, x, a = env.CACC(s,env.num_leading_cars)
        #print(v)
        print("Acc:::")
        print(a)
        input()
        s, reward, done, info = env.step(a)
        #print(reward)
        #print()
        i = i + 1

    print(reward)
    rewards.append(reward)
    print(i)

# For CACC
#create_loc_map(env)

rewards_CACC = rewards
data_d_CACC = deepcopy(data_d)
data_t_CACC = deepcopy(data_t)
s_d_CACC = start_disp

print("DONE with CACC")
#################################

#########################
env.normalize = True
env.verbose = True

rewards = []

agent = Agent(env)

# evaluate
# load the weights from file
#agent.load_state_dict(torch.load('./cem_cartpole.pth'))
#agent.load_state_dict(torch.load('./cem_cartpole_5.pth')) # Path to load model from
agent.load_state_dict(torch.load('./cem_cartpole_-23.pth'))

for i in range(num_episodes):
    #
    data_t = []
    data_d = []
    start_disp = None
    #

    state = env.reset_from_state(t_start)
    # For Graph
    env.verbose = True
    start_disp = env.center_state(env.current_states[0])
    #

    reward = None
    t = 0
    while True:
        with torch.no_grad():
            # For Graph
            add2loc_map(env)
            #
            #env.render()
            window.appendleft(torch.Tensor(state))
            action_probs = agent(deque2state(env)).detach().numpy()
            action = np.argmax(action_probs)
            a = (env.a_max - env.a_min)*((action)/(agent.action_size - 1)) + env.a_min
            #for i in range(agent.action_size):
                #print((env.a_max - env.a_min)*((i)/(agent.action_size - 1)) + env.a_min)
            #quit()
            #print(a)
            #input()
            next_state, reward, done, _ = env.step(a)
            
            state = next_state
            t = t + 1
            if done:
                break
    print(t)
    print(reward)
    rewards.append(reward)

rewards_ours = rewards
data_d_ours = deepcopy(data_d)
data_t_ours = deepcopy(data_t)
s_d_ours = start_disp

print("DONE with Ours")

##############################
env.normalize = False
env.verbose = True

rewards = []

for i in range(num_episodes):
    #
    data_t = []
    data_d = []
    start_disp = None
    #
    s = env.reset_from_state(t_start)
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
        s_ = env.get_state(env.t_start + i + 1)
        v = s_[env.num_leading_cars*3+0]
        x = s_[env.num_leading_cars*3+1]
        a = s_[env.num_leading_cars*3+2]
        #print(s_)
        #print(v)
        #print(a)
        s, reward, done, info = env.step(a,human=True)
        #print(s)
        #print(reward)
        #print()
        i = i + 1

    print(reward)
    rewards.append(reward)
    print(i)

rewards_human = rewards
data_d_human = deepcopy(data_d)
data_t_human = deepcopy(data_t)
s_d_human = start_disp

print("DONE with Human")
#########################

np.savetxt("rew_human.csv", rewards_human, delimiter=",")
np.savetxt("rew_cacc.csv", rewards_CACC, delimiter=",")
np.savetxt("rew_ours.csv", rewards_ours, delimiter=",")

print("HUMAN")
print("Average Reward:",np.mean(rewards_human))
print("SE Reward:",np.std(rewards_human)/(len(rewards_human))**0.5)
print()
print("CACC")
print("Average Reward:",np.mean(rewards_CACC))
print("SE Reward:",np.std(rewards_CACC)/(len(rewards_CACC))**0.5)
#print(np.sort(rewards_CACC))
print()
print("OURS")
print("Average Reward:",np.mean(rewards_ours))
print("SE Reward:",np.std(rewards_ours)/(len(rewards_ours))**0.5)
print()
print(t_start)
#print(np.sort(rewards_ours))
#print(rewards)

plt.hist(rewards_CACC, bins='auto',color="r")
#plt.hist(rewards_human, bins='auto',color="b")
plt.hist(rewards_ours, bins='auto',color="g")
plt.show()

#plt.title("Controller Rewards")
#plt.plot(rewards_CACC,"r")
#plt.plot(rewards_human,"b")
#plt.plot(rewards_ours,"g")
#plt.ylabel("Reward")
#plt.xlabel("Episode")
#plt.show()

# For graph

plt.title("Location Graph")

for n in range(env.num_vehicles):
    if(n < env.num_leading_cars):
        plt.plot(np.array(data_d_human)[:,n] + s_d_human[n*3 + 1],'b')
    elif(n == env.num_leading_cars):
        plt.plot(np.array(data_d_human)[:,n] + s_d_human[n*3 + 1],color="g",linestyle="--")
        plt.plot(np.array(data_d_CACC)[:,n] + s_d_CACC[n*3 + 1],color="g",linestyle=":")
        plt.plot(np.array(data_d_ours)[:,n] + s_d_ours[n*3 + 1],color="g")
    else:
        plt.plot(np.array(data_d_human)[:,n] + s_d_human[n*3 + 1],color="r",linestyle="--")
        plt.plot(np.array(data_d_CACC)[:,n] + s_d_CACC[n*3 + 1],color="r",linestyle=":")
        plt.plot(np.array(data_d_ours)[:,n] + s_d_ours[n*3 + 1],color="r")
plt.ylabel("Location")
plt.xlabel("Time")
plt.show()

for n in range(env.num_vehicles):
    if(n < env.num_leading_cars):
        plt.plot(np.array(data_d_human)[:,n] + s_d_human[n*3 + 1],'b')
    elif(n == env.num_leading_cars):
        #plt.plot(np.array(data_d_human)[:,n] + s_d_human[n*3 + 1],color="g",linestyle="--")
        plt.plot(np.array(data_d_CACC)[:,n] + s_d_CACC[n*3 + 1],color="g",linestyle=":")
        #plt.plot(np.array(data_d_ours)[:,n] + s_d_ours[n*3 + 1],color="g")
    else:
        #plt.plot(np.array(data_d_human)[:,n] + s_d_human[n*3 + 1],color="r",linestyle="--")
        plt.plot(np.array(data_d_CACC)[:,n] + s_d_CACC[n*3 + 1],color="r",linestyle=":")
        #plt.plot(np.array(data_d_ours)[:,n] + s_d_ours[n*3 + 1],color="r")
plt.ylabel("Location")
plt.xlabel("Time")
plt.show()

for n in range(env.num_vehicles):
    if(n < env.num_leading_cars):
        plt.plot(np.array(data_d_human)[:,n] + s_d_human[n*3 + 1],'b')
    elif(n == env.num_leading_cars):
        #plt.plot(np.array(data_d_human)[:,n] + s_d_human[n*3 + 1],color="g",linestyle="--")
        #plt.plot(np.array(data_d_CACC)[:,n] + s_d_CACC[n*3 + 1],color="g",linestyle=":")
        plt.plot(np.array(data_d_ours)[:,n] + s_d_ours[n*3 + 1],color="g")
    else:
        #plt.plot(np.array(data_d_human)[:,n] + s_d_human[n*3 + 1],color="r",linestyle="--")
        #plt.plot(np.array(data_d_CACC)[:,n] + s_d_CACC[n*3 + 1],color="r",linestyle=":")
        plt.plot(np.array(data_d_ours)[:,n] + s_d_ours[n*3 + 1],color="r")
plt.ylabel("Location")
plt.xlabel("Time")
plt.show()

for n in range(env.num_vehicles):
    if(n < env.num_leading_cars):
        plt.plot(np.array(data_d_human)[:,n] + s_d_human[n*3 + 1],'b')
    elif(n == env.num_leading_cars):
        plt.plot(np.array(data_d_human)[:,n] + s_d_human[n*3 + 1],color="g",linestyle="--")
        #plt.plot(np.array(data_d_CACC)[:,n] + s_d_CACC[n*3 + 1],color="g",linestyle=":")
        #plt.plot(np.array(data_d_ours)[:,n] + s_d_ours[n*3 + 1],color="g")
    else:
        plt.plot(np.array(data_d_human)[:,n] + s_d_human[n*3 + 1],color="r",linestyle="--")
        #plt.plot(np.array(data_d_CACC)[:,n] + s_d_CACC[n*3 + 1],color="r",linestyle=":")
        #plt.plot(np.array(data_d_ours)[:,n] + s_d_ours[n*3 + 1],color="r")
plt.ylabel("Location")
plt.xlabel("Time")
plt.show()

quit()
#-759874.2689289242 1,1 100
#-3092599.9404430203 3,3 100
#-18945850.40532393 3,3 2000
#-3804.086092265367 3,3 100
#-72983.58234821797 3,3 2000
#$$$$
#148.92242153042025
#1.772965201233505
#-190355.0
#150.69538673165377
#-190204.30461326835
#$$$$

