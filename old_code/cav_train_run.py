import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory
from rl.processors import WhiteningNormalizerProcessor

from CAVSimulator import Simulator

class MujocoProcessor(WhiteningNormalizerProcessor):
    def process_action(self, action):
        return (2.0 - (-2.0))*((np.argmax(action))/(15 - 1)) + 2.0

# Get the environment and extract the number of actions.
env = Simulator(3,3)
np.random.seed(123)

nb_actions = 15
obs_dim = env.observation_space.shape[0]

# Option 1 : Simple model
#model = Sequential()
#model.add(Flatten(input_shape=(1,) + (env.observation_space.shape[0],)))
#model.add(Dense(nb_actions))
#model.add(Activation('tanh'))

# Option 2: deep network
model = Sequential()
model.add(Flatten(input_shape=(1,) + (env.observation_space.shape[0],)))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))


print(model.summary())


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = EpisodeParameterMemory(limit=10000, window_length=1)

cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory,
               batch_size=1000, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05, noise_decay_const=0.0, noise_ampl=1.0, processor=MujocoProcessor())
cem.compile()

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
cem.fit(env, nb_steps=100000, visualize=False, verbose=2)

# After training is done, we save the best weights.
cem.save_weights('cem_CAV_params.h5f', overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
cem.test(env, nb_episodes=5, visualize=True)