# modify from https://github.com/udacity/deep-reinforcement-learning/blob/master/cross-entropy/CEM.ipynb
import numpy as np
import gym
from gym import wrappers
from collections import deque
import torch
import torch.nn as nn

import random

from CAVSimulator0910 import Simulator

from collections import deque

import torch.optim as optim

import torch.nn.functional as F

##### Controller HYPERPARAMETERS FOR TUNING

start_from_init = True
num_leading_vehicle = 3
num_following_vehicle = 0


print("Controller Hyperparameters")
print(start_from_init, num_leading_vehicle, num_following_vehicle)
print("#"*30)
######

def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

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

def win_deque2state(env,tmp_window):
    state = tmp_window[0]
    d_t = 1
    for i in range(1, window_size):
        if tmp_window[i] is None:
            # on start get data directly
            #window[i] = torch.zeros(len(window[0]))
            tmp_window[i] = torch.Tensor(np.array(env.center_state(env.get_state(env.t_start - d_t))))
            d_t = d_t + 1
        state = torch.cat([state,tmp_window[i]])
    return state

from copy import deepcopy
import matplotlib.pyplot as plt

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


class Agent(nn.Module):
    def __init__(self,
                 env: gym.wrappers.time_limit.TimeLimit,
                 hidden_size: int = 256) -> None:
        super(Agent, self).__init__()
        self.env = env
        self.space_size = env.observation_space.shape[0] * window_size
        self.hidden_size = hidden_size
        self.action_size = env.action_space.n

        self.fc1 = nn.Linear(self.space_size, self.hidden_size)
        self.fc1_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc1_3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc1_4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc1_5 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.action_size)

        self.eval_long = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.fc1(x))
        out = torch.relu(self.fc1_2(out))
        out = torch.relu(self.fc1_3(out))
        out = torch.relu(self.fc1_4(out))
        out = torch.relu(self.fc1_5(out))
        out = torch.tanh(self.fc2(out)) # try removing
        #out = torch.softmax(out, dim=0)
        return out

    def set_weights(self, weights: np.ndarray) -> None:
        s = self.space_size
        h = self.hidden_size
        a = self.action_size
        # separate the weights for each layer
        fc1_end = (s * h) + h
        fc1_2_end = fc1_end + (h * h) + h
        fc1_3_end = fc1_2_end + (h * h) + h
        fc1_4_end = fc1_2_end + (h * h) + h
        fc1_5_end = fc1_2_end + (h * h) + h

        fc1_W = torch.from_numpy(weights[:s * h].reshape(s, h))
        fc1_b = torch.from_numpy(weights[s * h:fc1_end])

        fc1_2_W = torch.from_numpy(weights[fc1_end:fc1_end+(h*h)].reshape(h, h))
        fc1_2_b = torch.from_numpy(weights[fc1_end + (h*h):fc1_2_end])

        fc1_3_W = torch.from_numpy(weights[fc1_2_end:fc1_2_end+(h*h)].reshape(h, h))
        fc1_3_b = torch.from_numpy(weights[fc1_2_end + (h*h):fc1_3_end])

        fc1_3_W = torch.from_numpy(weights[fc1_3_end:fc1_3_end+(h*h)].reshape(h, h))
        fc1_3_b = torch.from_numpy(weights[fc1_3_end + (h*h):fc1_4_end])

        fc1_3_W = torch.from_numpy(weights[fc1_4_end:fc1_2_end+(h*h)].reshape(h, h))
        fc1_3_b = torch.from_numpy(weights[fc1_4_end + (h*h):fc1_5_end])
        
        fc2_W = torch.from_numpy(weights[fc1_5_end:fc1_5_end+(h*a)].reshape(h, a))
        fc2_b = torch.from_numpy(weights[fc1_5_end+(h*a):])

        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))

        self.fc1_2.weight.data.copy_(fc1_2_W.view_as(self.fc1_2.weight.data))
        self.fc1_2.bias.data.copy_(fc1_2_b.view_as(self.fc1_2.bias.data))

        self.fc1_3.weight.data.copy_(fc1_3_W.view_as(self.fc1_3.weight.data))
        self.fc1_3.bias.data.copy_(fc1_3_b.view_as(self.fc1_3.bias.data))

        self.fc1_4.weight.data.copy_(fc1_3_W.view_as(self.fc1_4.weight.data))
        self.fc1_4.bias.data.copy_(fc1_3_b.view_as(self.fc1_4.bias.data))

        self.fc1_5.weight.data.copy_(fc1_3_W.view_as(self.fc1_5.weight.data))
        self.fc1_5.bias.data.copy_(fc1_3_b.view_as(self.fc1_5.bias.data))

        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

    def get_weights_dim(self) -> int:
        return (self.space_size + 1) * self.hidden_size + (self.hidden_size + 1) * self.hidden_size + (self.hidden_size + 1) * self.hidden_size + (self.hidden_size + 1) * self.hidden_size + (self.hidden_size + 1) * self.hidden_size + \
            (self.hidden_size + 1) * self.action_size

data_loss = []
data_acc = []
def mimic_optimize(env,agent,Replay_Buffer,buffer_size):
    optimizer = optim.Adagrad(agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    if buffer_size == None:
        buffer_size = len(Replay_Buffer)

    for _ in range(1):
        batch = random.sample(Replay_Buffer,buffer_size)
        inp = torch.cat([win_deque2state(env,x[0]).reshape(1,-1) for x in batch],0)
        target = torch.Tensor([[((x[1] - env.a_min) / (env.a_max - env.a_min)) *  (agent.action_size - 1)] for x in batch]).int().long()
        t_onehot = torch.FloatTensor(len(batch), agent.action_size)

        # In your for loop
        t_onehot.zero_()
        t_onehot.scatter_(1, torch.clamp(target,0,14), 1)


        optimizer.zero_grad()   # zero the gradient buffers
        output = agent(inp)

        loss = criterion(output, t_onehot)
        loss.backward()
        optimizer.step()    # Does the update
        print("Loss",loss)
  
        acc = (((torch.argmax(F.softmax(output), 1, keepdim=True)  == target).float().sum()).numpy()/ (len(target))) * 100
        print("Accuracy",str(acc)+"%")
        #print(random.sample([x[1] for x in batch],10))
        data_loss.append(loss.detach().numpy())
        data_acc.append(acc/100)

    return loss.detach().numpy(), acc

if __name__ == "__main__":
    env = Simulator(num_leading_vehicle,num_following_vehicle)

    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    agent = Agent(env)

    # evaluate
    # load the weights from file
    #agent.load_state_dict(torch.load('./cem_cartpole.pth'))
    #agent.load_state_dict(torch.load('./cem_cartpole_5.pth')) # Path to load model from
    #agent.load_state_dict(torch.load('./cem_cartpole.pth'))
    num_episodes = 1000
    rewards = []

    Replay_Buffer = deque(maxlen=10000)

    for i in range(num_episodes):
        #
        data_t = []
        data_d = []
        start_disp = None
        #

        state = env.reset()
        # For Graph
        #env.verbose = True
        start_disp = env.center_state(env.current_states[0])
        #

        reward = None
        t = 0
        while True:
            with torch.no_grad():
                #env.render()
                window.appendleft(torch.Tensor(state))
                #action_probs = agent(deque2state(env)).detach().numpy()
                #action = np.argmax(action_probs)
                #a = (env.a_max - env.a_min)*((action)/(agent.action_size - 1)) + env.a_min
                #for i in range(agent.action_size):
                    #print((env.a_max - env.a_min)*((i)/(agent.action_size - 1)) + env.a_min)
                #quit()
                #print(a)
                #input()
                
                #v, x, a = env.CACC(state,env.num_leading_cars)
                a = env.get_state(env.t_start+env.t)[3*num_leading_vehicle + 2]
                Replay_Buffer.appendleft([window,a])
                #print(v)

                next_state, reward, done, _ = env.step(a,human=True)
                # For Graph
                add2loc_map(env)
                #
                state = next_state
                t = t + 1
                if done:
                    break
        #print(t)
        #print(reward)
        rewards.append(reward)
        if(len(Replay_Buffer) > 2048):
            mimic_optimize(env,agent,Replay_Buffer,2048)

    print("DONE")
    acc = 0
    loss = 0
    n = 0
    while acc < 90 or loss > 0.02:
        loss, acc = mimic_optimize(env,agent,Replay_Buffer,2048)
        n = n + 1
        print("Epoch",n)
        print("Loss",loss," Accuracy",str(acc)+"%")
        print()

    torch.save(agent.state_dict(), './mimic_cav_' + str(int(acc)) + '_.pth') # Path to save model to

window_size = 10

plt.title("Learning Curve")
plt.plot(moving_average(data_loss,window_size)[window_size:-window_size],"r")
plt.plot(moving_average(data_acc,window_size)[window_size:-window_size],"g")
plt.ylabel("Acc/Loss")
plt.xlabel("Step")
plt.show()

quit()
print("Average Reward:",np.mean(rewards))
print("SE Reward:",np.std(rewards)/(len(rewards))**0.5)
#print(rewards)
rewards = np.sort(rewards)
plt.hist(rewards, bins='auto')
plt.show()

create_loc_map(env)

# TODO
#Reward graph CACC, OURS, and Human 
#Combine Location Graphs CACC, OURS, and Human (All dataset)
#Add layers to Network
#Gather data controller + train our controller to begin. Check MSE
#Add constraints to the controller (Add gap rectifier no penalty)
