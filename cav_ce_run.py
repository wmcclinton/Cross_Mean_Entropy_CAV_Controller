# modify from https://github.com/udacity/deep-reinforcement-learning/blob/master/cross-entropy/CEM.ipynb
import numpy as np
import gym
from gym import wrappers
from collections import deque
import torch
import torch.nn as nn

from CAVSimulator import Simulator

class Agent(nn.Module):
    def __init__(self,
                 env: gym.wrappers.time_limit.TimeLimit,
                 hidden_size: int = 32) -> None:
        super(Agent, self).__init__()
        self.env = env
        self.space_size = env.observation_space.shape[0]
        self.hidden_size = hidden_size
        self.action_size = env.action_space.n

        self.fc1 = nn.Linear(self.space_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.fc1(x))
        out = torch.tanh(self.fc2(out))
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
        for i in range(num_episodes):
            state = self.env.reset()
            for t in range(max_t):
                state = torch.Tensor(state)
                action_probs = self.forward(state)
                action = np.argmax(action_probs.detach().numpy())
                state, reward, done, _ = self.env.step(action)
                episode_return += reward * gamma**t
                if done:
                    break
        return episode_return/num_episodes


def cem(agent: Agent,
        n_iters: int = 1000,
        max_t: int = 1000,
        gamma: float = 1,
        pop_size: int = 50,
        elite_frac: int = 0.05,
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
        elite_frac: percentage of top performers to use in update
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

        reward = agent.evaluate(best_weight, gamma=1.0)
        scores_deque.append(reward)
        scores.append(reward)

        torch.save(agent.state_dict(), './cem_cartpole.pth') # Path to save model to

        print('Episode {}\tAverage Score: {:.2f}'.\
              format(i_iter, np.mean(scores_deque)))


    return agent, scores


if __name__ == "__main__":
    # Variable to designate train or just load from path and test
    train = False
    #
    env = Simulator(3,3)

    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    agent = Agent(env)
    if train:
        agent, scores = cem(agent)

    # evaluate
    # load the weights from file
    #agent.load_state_dict(torch.load('./cem_cartpole.pth'))
    #agent.load_state_dict(torch.load('./cem_cartpole.pth')) # Path to load model from

    state = env.reset()
    reward = None
    t = 0
    while True:
        with torch.no_grad():
            env.render()
            action_probs = agent(torch.Tensor(state)).detach().numpy()
            action = np.argmax(action_probs)
            next_state, reward, done, _ = env.step(action)
            print(state)
            input()
            state = next_state
            t = t + 1
            if done:
                break
    print(t)
    print(reward)