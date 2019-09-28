from __future__ import division

# modify from https://github.com/udacity/deep-reinforcement-learning/blob/master/cross-entropy/CEM.ipynb
import numpy as np
import gym
from gym import wrappers
from collections import deque
import torch
import torch.nn as nn

from CAVSimulator0910_old import Simulator

env = Simulator(3,0)

import argparse
import sys
sys.path.append('../../keras-rl')
from PIL import Image
import numpy as np
import gym
from keras.models import Model
from keras.layers import Flatten, Convolution2D, Input, Dense
from keras.optimizers import Adam
import keras.backend as K
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import TrainEpisodeLogger, ModelIntervalCheckpoint

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Softmax
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory

#We downsize the atari frame to 84 x 84 and feed the model 4 frames at a time for
#a sense of direction and speed.
INPUT_SHAPE = (env.observation_space.shape)
WINDOW_LENGTH = 1

#Standard Atari processing
class AtariProcessor(Processor):
    def process_observation(self, observation):
        observation
        processed_observation = np.array(observation)
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') #/ 255.
        return processed_batch.reshape([-1,12])

    def process_reward(self, reward):
        #return np.clip(reward, -10., 1000.)
        return reward

    def process_action(self, action):
        a = (env.a_max - env.a_min)*((action)/(env.action_space.n - 1)) + env.a_min
        return a


# Get the environment and extract the number of actions.
nb_actions = env.action_space.n
print("NUMBER OF ACTIONS: " + str(nb_actions))

#Standard DQN model architecture.
input_shape = (WINDOW_LENGTH*INPUT_SHAPE[0],)
frame = Input(shape=(input_shape))
dense = Dense(512, activation='relu')(frame)
dense = Dense(512, activation='relu')(dense)
buttons = Dense(nb_actions, activation='linear')(dense)
buttons = Softmax()(buttons)
model = Model(inputs=frame,outputs=buttons)
print(model.summary())

processor = AtariProcessor()

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = EpisodeParameterMemory(limit=100000, window_length=WINDOW_LENGTH)

cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory, processor=processor,
               batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
cem.compile()

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
cem.fit(env, nb_steps=1000000, visualize=False, verbose=2)

# After training is done, we save the best weights.
cem.save_weights('cem_{}_params.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
cem.test(env, nb_episodes=5, visualize=True)