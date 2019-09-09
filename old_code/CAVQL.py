import numpy as np
import time

class QLearner():

    def init_q(self,s, a, type="ones"):
        if type == "ones":
            return np.ones((s, a))
        elif type == "random":
            return np.random.random((s, a))
        elif type == "zeros":
            return np.zeros((s, a))

    def epsilon_greedy(self,Q, epsilon, n_actions, s, test=False):
        if test or np.random.rand() < epsilon:
            action = np.argmax(Q[s, :])
        else:
            action = np.random.randint(0, n_actions)
        return action

    def check_constraints(self, a, state):
        # TODO Need to check 3 contraints
        return a, 0

    def run(self, env, alpha, gamma, epsilon, episodes, max_steps, n_tests, render = False, test=False):
        n_states, n_actions = env.observation_space.n, env.action_space.n
        Q = self.init_q(n_states, n_actions, type="ones")
        timestep_reward = []
        for episode in range(episodes): # One Trial (100000)
            print(f"Episode: {episode}")
            s = env.reset()
            # First Action
            a = self.epsilon_greedy(Q, epsilon * (episode/episodes), n_actions, s)
            t = 0
            total_reward = 0
            done = False
            while t < max_steps: # One Interval (10)
                if render:
                    env.render()
                t += 1
                ######
                a, c_reward = self.check_constraints(a, env.current_states[-1])
                ######

                s_, reward, done, info = env.step(a) # One Step (1)
                #print(reward)

                ######
                reward = reward + c_reward
                ######

                total_reward += reward
                # Next action
                a_ = self.epsilon_greedy(Q, epsilon * (episode/episodes), n_actions, s_)
                if done:
                    # TODO Update NN
                    # Update Q-Table
                    Q[s, a] += alpha * ( reward  - Q[s, a] )
                else:
                    # TODO Update Q-Table
                    Q[s, a] += alpha * ( reward + (gamma * Q[s_, a_]) - Q[s, a] )
                s, a = s_, a_
                if done:
                    if render:
                        print(f"This episode took {t} timesteps and reward: {total_reward}")
                    timestep_reward.append(total_reward)
                    break
        if render:
            print(f"Here are the Q values:\n{Q}\nTesting now:")
        if test:
            self.test_agent(Q, env, n_tests, n_actions)
        return timestep_reward

    def test_agent(self,Q, env, n_tests, n_actions, delay=1):
        for test in range(n_tests):
            print(f"Test #{test}")
            s = env.reset()
            done = False
            epsilon = 0
            while True:
                time.sleep(delay)
                env.render()
                a = self.epsilon_greedy(Q, epsilon, n_actions, s, test=True)
                print(f"Chose action {a} for state {s}")
                s, reward, done, info = env.step(a)
                if done:
                    print("Reward",reward)
                    print()
                    time.sleep(3)
                    break
