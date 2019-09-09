from CAVNNQL import NNQLearner
from CAVSimulator import Simulator
import numpy as np
#import gym

episodes = 1000
max_steps = 2500
n_tests = 1

# General Table Q-Learner (Safer than Function Approximators)
Q = NNQLearner()

# OpenAI GYM
#env = gym.make('Taxi-v2')

# CAV Simulator (Generates Fake Data now)
env = Simulator(3,3)
#env.verbose = True

# Gets rewards as training
timestep_reward = Q.run(env, episodes, max_steps, n_tests, test = True)
print(timestep_reward)

#

# Plot
import matplotlib.pyplot as plt

N = 5000
plt.plot(timestep_reward)
plt.ylabel("Reward")
plt.xlabel("Num of Episodes")
plt.show()