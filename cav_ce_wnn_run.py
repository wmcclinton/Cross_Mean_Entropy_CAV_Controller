# modify from https://github.com/udacity/deep-reinforcement-learning/blob/master/cross-entropy/CEM.ipynb
import numpy as np
import gym
from gym import wrappers
from collections import deque
import torch
import torch.nn as nn

from CAVSimulator0910 import Simulator

from collections import deque

##### Controller HYPERPARAMETERS FOR TUNING

start_from_init = False
num_leading_vehicle = 3
num_following_vehicle = 0
num_eps = 300


print("Controller Hyperparameters")
print()
print("#"*30)
######

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
                 hidden_size: int = 64) -> None:
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

    def evaluate(self,
                 weights: torch.Tensor,
                 gamma: float = 1.0,
                 max_t: float = 5000) -> float:
        self.set_weights(weights)
        episode_return = 0.0
        num_episodes = 10
        if self.eval_long:
            num_episodes =  300
        rewards = []
        for i in range(num_episodes):
            state = self.env.reset()
            for t in range(max_t):
                state = torch.Tensor(state)
                window.appendleft(state)
                action_probs = self.forward(deque2state(env))
                action = np.argmax(action_probs.detach().numpy())
                a = (env.a_max - env.a_min)*((action)/(agent.action_size - 1)) + env.a_min
                state, reward, done, _ = self.env.step(a)
                episode_return += reward * gamma**t
                if done:
                    rewards.append(reward)
                    break
        if self.eval_long:
            print('Long Eval: Average Score: {:.2f}\tSE Score: {:.2f}'.\
                format(np.mean(rewards), np.std(rewards)/(len(rewards)**0.5)))
            print('Long Eval: Median Score: {:.2f}'.\
                format(np.median(rewards)))
            print(len(rewards))
        return episode_return/num_episodes


def cem(agent: Agent,
        n_iters: int = 500,
        max_t: int = 1000,
        gamma: float = 1,
        pop_size: int = 50,
        elite_frac: int = 0.2,
        std: float = 0.5):
    """
    PyTorch implementation of the cross-entropy method.
    Params
    ======
        n_iter: maximum number of training iterations
        max_t: maximum number of timesteps per episode
        gamma: discount rate
        print_every: how often to print average score (over last 100 episodes)
        pop_size: size of population at each iteration
        elite_frac: percentage of top performers to use in update (0.2)
        std: standard deviation of additive noise (0.5)
    """
    n_elite=int(pop_size * elite_frac)
    scores_deque = deque(maxlen=10)
    scores = []
    best_weight = std * np.random.randn(agent.get_weights_dim())
    elite_weights = [std * np.random.randn(agent.get_weights_dim()) for i in range(n_elite)]

    for i_iter in range(n_iters):
        weights_pop = [best_weight +
                       (std * np.random.randn(agent.get_weights_dim()))
                       for i in range(pop_size - n_elite)] + elite_weights

        rewards = np.array([agent.evaluate(weights, gamma, max_t)
                            for weights in weights_pop])

        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        best_weight = np.array(elite_weights).mean(axis=0)
        
        agent.eval_long = True
        reward = agent.evaluate(best_weight, gamma=1.0)
        agent.eval_long = False
        scores_deque.append(reward)
        scores.append(reward)

        torch.save(agent.state_dict(), './cem_cartpole.pth') # Path to save model to

        print('Episode {}\tBest Average Score: {:.2f}'.\
              format(i_iter, np.mean(scores_deque)))
        print('Episode {}\tAll Average Score: {:.2f}\tAll SE Score: {:.2f}'.\
              format(i_iter, np.mean(rewards), np.std(rewards)/(len(rewards)**0.5)))


    return agent, scores

if __name__ == "__main__":
    # Variable to designate train or just load from path and test
    train = True
    #
    env = Simulator(num_leading_vehicle,num_following_vehicle)

    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    agent = Agent(env)
    #
    if start_from_init:
        print("Started from CACC initialization")
        agent.load_state_dict(torch.load('./mimic_cav_90_.pth'))
    #
    if train:
        agent, scores = cem(agent)

    # evaluate
    # load the weights from file
    #agent.load_state_dict(torch.load('./cem_cartpole.pth'))
    #agent.load_state_dict(torch.load('./cem_cartpole_5.pth')) # Path to load model from
    #agent.load_state_dict(torch.load('./mimic_cav_90_.pth'))
    num_episodes = num_eps
    rewards = []

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
                action_probs = agent(deque2state(env)).detach().numpy()
                action = np.argmax(action_probs)
                a = (env.a_max - env.a_min)*((action)/(agent.action_size - 1)) + env.a_min
                #for i in range(agent.action_size):
                    #print((env.a_max - env.a_min)*((i)/(agent.action_size - 1)) + env.a_min)
                #quit()
                #print(a)
                #input()
                next_state, reward, done, _ = env.step(a)
                # For Graph
                add2loc_map(env)
                #
                state = next_state
                t = t + 1
                if done:
                    break
        print(t)
        print(reward)
        rewards.append(reward)
        

print("Average Reward:",np.mean(rewards))
print("SE Reward:",np.std(rewards)/(len(rewards))**0.5)
quit()
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
