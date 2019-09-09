import time

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from os import system
import copy


# Hyper Parameters for QN

GAMMA = 0.95
LEARNING_RATE = 0.0003

MEMORY_SIZE = 1000
BATCH_SIZE = 64

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DQNSolver:
    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        avg_loss = 100
        while avg_loss < 0.5:
            batch = random.sample(self.memory, BATCH_SIZE)
            avg_loss = 100
            for state, action, reward, state_next, terminal in batch:
                q_update = reward
                q_values = self.model.predict(state)
                q_values[0][action] = q_update
                #print(q_values)
                history = self.model.fit(state, q_values, verbose=0)
                avg_loss = 0.5*avg_loss + 0.5*float(history.history['loss'][0])
            print(avg_loss)
        input()
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

class NNQLearner():
    def discertize(self,env,a):
        observation_space, action_space = env.observation_space.n, env.action_space.n
        bins = [(env.a_max - env.a_min)*((a+1)/observation_space) + env.a_min for a in range(action_space)]
        dis_a = 0
        dist = abs(bins[0] - a)
        for i in range(action_space-1):
            if abs(bins[i+1] - a) < dist:
                dis_a = i+1
                dist = abs(bins[i+1] - a)

        return dis_a

    def run(self, env, episodes, max_steps, n_tests, render = True, test=False):
        observation_space, action_space = env.observation_space.n, env.action_space.n
        self.DQN = DQNSolver(observation_space, action_space)
        timestep_reward = []
        for episode in range(episodes): # One Trial (100000)
            print(f"Episode: {episode}")
            s = env.reset()
            # First Action
            if (episode%env.warmup_gap < env.warmup):
                print("WARMUP")
                v, x, a = env.CACC(s,env.num_leading_cars)
                a = self.discertize(env,a)
                s = np.reshape(env.center_state(s), [1, observation_space])
            else:
                print("Q-Controller")
                s = np.reshape(env.center_state(s), [1, observation_space])
                a = self.DQN.act(s)
            print(a)
            t = 0
            total_reward = 0
            done = False
            episode_mem = []
            while t < max_steps: # One Interval (10)
                if render:
                    #env.render()
                    null = 0
                t += 1
                
                s_, reward, done, info = env.step((env.a_max - env.a_min)*((a+1)/observation_space) + env.a_min) # One Step (1)

                total_reward += reward
                if (episode%env.warmup_gap < env.warmup):
                    v_, x_, a_ = env.CACC(s_,env.num_leading_cars)
                    a_ = self.discertize(env,a_)
                    s_ = np.reshape(env.center_state(s_), [1, observation_space])
                else:
                    s_ = np.reshape(env.center_state(s_), [1, observation_space])
                    a_ = self.DQN.act(s)
                episode_mem.append([s, a, None, s_, done])
                s, a = s_, a_
                if done:
                    if render:
                        print("Run: " + str(episode) + ", exploration: " + str(self.DQN.exploration_rate) + ", score: " + str(total_reward))
                        print(f"This episode took {t} timesteps and reward: {total_reward}")
                    timestep_reward.append(total_reward)
                    break
            for ep in episode_mem:
                self.DQN.remember(ep[0], ep[1], total_reward, ep[3], ep[4])
            self.DQN.experience_replay()
        if test:
            self.test_agent(env, n_tests)
        return timestep_reward

    def test_agent(self,env, n_tests, delay=0):
        observation_space, action_space = env.observation_space.n, env.action_space.n
        for test in range(n_tests):
            print(f"Test #{test}")
            s = env.reset()
            s = np.reshape(s, [1, observation_space])
            done = False
            while True:
                time.sleep(delay)
                env.render()
                a = self.DQN.act(s)
                print(f"Chose action {a} for state {s}")
                s, reward, done, info = env.step((env.a_max - env.a_min)*(a/(action_space-1)) + env.a_min)
                s = np.reshape(s, [1, observation_space])
                if done:
                    print("Reward",reward)
                    print()
                    time.sleep(3)
                    break
