import numpy as np

import gym
from gym import wrappers

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from CAVSimulator import Simulator

class MujocoProcessor(WhiteningNormalizerProcessor):
    def process_action(self, action):
        max_acc = 2.0
        return int(np.clip(action, -1., 1.) * max_acc)


# Get the environment and extract the number of actions.
env = Simulator(3,3)
np.random.seed(123)

nb_actions = 1

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + (env.observation_space.shape[0],)))
actor.add(Dense(20))
actor.add(Activation('relu'))
actor.add(Dense(20))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('tanh'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + (env.observation_space.shape[0],), name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Dense(20)(flattened_observation)
x = Activation('relu')(x)
x = Concatenate()([x, action_input])
x = Dense(20)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000, gamma=1, target_model_update=1e-3,
                  processor=MujocoProcessor())
agent.compile([Adam(lr=1e-5), Adam(lr=1e-4)], metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=100000, visualize=False, verbose=1)

# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)