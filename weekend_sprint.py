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

#We downsize the atari frame to 84 x 84 and feed the model 4 frames at a time for
#a sense of direction and speed.
INPUT_SHAPE = (env.observation_space.shape)
WINDOW_LENGTH = 4

#Standard Atari processing
class AtariProcessor(Processor):
    def process_observation(self, observation):
        observation
        processed_observation = np.array(observation)
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') #/ 255.
        return processed_batch.reshape([-1,48])

    def process_reward(self, reward):
        return reward #np.clip(reward, -1., 1.)

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
model = Model(inputs=frame,outputs=buttons)
print(model.summary())

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

processor = AtariProcessor()

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1250000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, enable_double_dqn=False, enable_dueling_network=False, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)

dqn.compile(Adam(lr=.00025), metrics=['mae'])

folder_path = '../model_saves/Vanilla/'

train = True

if train:
    weights_filename = folder_path + 'dqn_{}_weights.h5f'.format("CAV_Controller")
    checkpoint_weights_filename = folder_path + 'dqn_' + "CAV_Controller" + '_weights_{step}.h5f'
    log_filename = folder_path + 'dqn_' + "CAV_Controller" + '_REWARD_DATA.txt'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=500000)]
    #callbacks += [TrainEpisodeLogger(log_filename)]
    dqn.fit(env, callbacks=callbacks, nb_steps=10000000, verbose=6, nb_max_episode_steps=20000)

else:
    weights_filename = folder_path + 'dqn_MsPacmanDeterministic-v4_weights_10000000.h5f'
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True, nb_max_start_steps=80)

quit()

num_episodes = 100
rewards = []

for i in range(num_episodes):
    s = env.reset()

    done = 0
    i = 0
    while not done:
        #env.render()
        v, x, a = env.CACC(s,env.num_leading_cars)
        print(a)
        s, reward, done, info = env.step(a,controller="CACC")
        i = i + 1

    print(reward)
    quit()