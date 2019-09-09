from random import randint
from scipy.io import loadmat
import os
import pygame
import numpy as np
import math
pygame.init()

data = loadmat('veh_combined.mat')['veh_combined']
# time_index, location, distance to prior vehicle, velocity, veocity difference to prior vehicle, acceleration

class observation_space:
    def __init__(self, num_leading_cars, num_following_cars):
        # (Velocity, Distance, Acceleration) per vehicle + (V,A) for car
        self.size = 3 * (num_leading_cars + num_following_cars + 1)
        self.n = self.size
        self.shape = (self.size,1)

class action_space:
    def __init__(self):
        # Acceleration
        self.n = 15

# Simulator needed for CAVQL
class Simulator():
    def __init__(self, num_leading_cars = 0, num_following_cars = 0):
        print("Initializing Environment...")
        # Window Size for visualization
        self.win_width = 1200
        self.win_height = 500
        self.win = None

        self.num_leading_cars = num_leading_cars
        self.num_following_cars = num_following_cars
        self.num_vehicles = self.num_leading_cars + 1 + self.num_following_cars
        self.observation_space = observation_space(num_leading_cars, num_following_cars)
        self.action_space = action_space()

        # Past stats and current displacement
        self.current_states = []
        self.current_disps = []

        # Start marker for data
        self.t_start = 10

        # Current time step
        self.t = 0

        # Max number of time steps
        self.max_t = 300

        # Parameters for reward
        # For weighting acceleration cost
        self.LAMBDA = 1 #100
        # To prevent division by zero in acceleration avg.
        self.EPSILON = 0.0001
        # To scale reward to be smaller (Helps with learning)
        self.rew_normalize = 1
        # Keeps track of negative rewards
        self.neg_rewards = []
        # Checks if crashed
        self.is_crashed = False


        # Max and Min Vel and Acc
        self.a_max = 2.0 #2.0
        self.a_min = -2.0 #-2.0
        self.dt = 0.1 # time interval
        self.v_max = 16 #16
        self.v_min = 0 #0

        # Allows printing of Vel, Acc, and Gap warnings
        self.verbose = False
        # Allows for centering of states
        self.normalize = True  

        # Amount of time steps to use CACC to prior to learning
        self.warmup = 100
        # Amoun of time steps to wait before using CACC again
        self.warmup_gap = 200

    def get_state(self,t):
        s = []
        for car in data[t][0:self.num_vehicles]:
            v = car[3]
            x = car[1]
            a = car[5]
            s = s + [v,x,a]
        return s

    def center_state(self,s):
        new_s = s
        if self.normalize:
            cav_pos = s[(self.num_leading_cars)*3+1]
            shift = [0,cav_pos,0]*self.num_vehicles
            new_s = [a - b for a, b in zip(s,shift)]
        return new_s

    def update_state(self,s,a,human=False):
        rew = 0 # penalty reward for one step
        # for leading vehicles
        v_set_next = data[self.t_start+self.t+1][0:self.num_leading_cars][:,3].reshape((1,-1))[0]
        # d = distance between any vehicle and the controlled vehicle?
        x_set_next = data[self.t_start+self.t+1][0:self.num_leading_cars][:,1].reshape((1,-1))[0] # better to use x than d, otherwise d for leading vehicles have to be calculated
        a_set_next = data[self.t_start+self.t+1][0:self.num_leading_cars][:,5].reshape((1,-1))[0]
        
        s_ = [0 for i in range(len(s))] # next state
        for n in range(self.num_leading_cars): # from observation in reality, from data in simulation
            s_[n*3+0] = v_set_next[n] # t: current time index
            s_[n*3+1] = x_set_next[n]
            s_[n*3+2] = a_set_next[n]

        # for controlled CAV
        v = s[self.num_leading_cars*3+0] # current Location of the controlled vehicle
        x = s[self.num_leading_cars*3+1] # if size of s is 3*(num_leading_cars+1+num_following_cars)
        #a = s[self.num_leading_cars*3+2]
        # or def CACC
        # DRL controller
        pn_a_max = -5 #pn_a_max = -self.a_max*1000
        pn_a_min = -5 #pn_a_min = -abs(self.a_min)*1000
        # penalty for violating constraint for acceleration
        if a > self.a_max: # or use rectify function
            a_ = self.a_max
            rew = rew + pn_a_max
            if(self.verbose):
                print("Violates Max Acceleration")
                #print(a)
                #quit()
        elif a < self.a_min:
            a_ = self.a_min
            rew = rew + pn_a_min
            if(self.verbose):
                print("Violates Min Acceleration")
                #print(a)
                #quit()
        else:
            a_ = a
        
        v_ = s[self.num_leading_cars*3+0] + a_*self.dt
        if human:
            v_ = self.get_state(self.t_start+self.t+1)[self.num_leading_cars*3+0]
        self.v_max = 16
        self.v_min = 0
        pn_v_max = -5 # pn_v_max = -self.v_max*1000
        pn_v_min = -5 # pn_v_min = -2000
        # penalty for violating constraint for velocity
        if v_ > self.v_max:
            v_ = self.v_max
            a_ = (v_-s[self.num_leading_cars*3+0]) / self.dt
            rew = rew + pn_v_max
            if(self.verbose):
                print("Violates Max Vel")
                #quit()
        elif v_ < self.v_min:
            v_ = self.v_min
            a_ = (v_-s[self.num_leading_cars*3+0]) / self.dt
            rew = rew + pn_v_min
            if(self.verbose):
                print("Violates Min Vel")
                #quit()
       
        x_ = x + v*self.dt + (1/2)*a*self.dt**2
        if human:
            x_ = self.get_state(self.t_start+self.t+1)[self.num_leading_cars*3+1]
        tau = int(1/self.dt) # reaction time = 1 sec
        x_pre = data[self.t_start+self.t+1-tau][self.num_leading_cars-1][1]
        min_s = 2.0 # safe gap
        l = 6.85
        gap = x_pre - x_ - l - min_s - v*self.dt 
        pn_safe = -5 # pn_safe = -10000
        # penalty for violating constraint for safety
        #print(gap)
        if gap < -1e-10:
            if(self.verbose):
                print("#")
                print(gap)
                print("Warning Within Gap")
                #quit()
            gap = 0
            x_ = x_pre - l - min_s - v*self.dt
            a_ = (x_ - x - v*self.dt)*2/(self.dt**2)
            if a_<self.a_min or a_>self.a_max:
                if(self.verbose):
                    print("safety wrong acc")
                    # TODO
                    #quit()
            v_ = v + a_*self.dt
            if v_<self.v_min or v_>self.v_max:
                if(self.verbose):
                    print("safety wrong vel")
                    # TODO
                    #quit()
            rew = rew + pn_safe

        gap = x_pre - x_ - l - min_s - v*self.dt
        if gap < -1e-10:
            if(self.verbose):
                print("Crashed")
            quit()
            self.is_crashed = True
            rew = rew + -100000
        
        if gap > 8:
            rew = rew + -5
            if(self.verbose):
                print("Warning Gap Too Large")

        s_[self.num_leading_cars*3+0] = v_
        s_[self.num_leading_cars*3+1] = x_
        s_[self.num_leading_cars*3+2] = a_
        if human:
            s_[self.num_leading_cars*3+0] = self.get_state(self.t_start+self.t+1)[self.num_leading_cars*3+0]
            s_[self.num_leading_cars*3+1] = self.get_state(self.t_start+self.t+1)[self.num_leading_cars*3+1]
            s_[self.num_leading_cars*3+2] = self.get_state(self.t_start+self.t+1)[self.num_leading_cars*3+2]
        # for following num_vehicles
        f_controllers = "IDM" # Designates what controller to use for following
        for n in range(self.num_leading_cars+1,self.num_leading_cars + self.num_following_cars + 1):
            v_temp = None
            x_temp = None
            a_temp = None
            if f_controllers == "IDM":
                v_temp, x_temp, a_temp = self.IDM(s,n,s_)
            elif f_controllers == "CACC":
                v_temp, x_temp, a_temp = self.CACC(s,n)
            # Add new controllers here ***
            s_[n*3+0] = v_temp
            s_[n*3+1] = x_temp
            s_[n*3+2] = a_temp
        return s,a,rew,s_,a_

    def IDM(self,s,n,s_):###### update by adding s_ for safety constraint
        # model specific parameters
        l = 6.85 # vehicle effective length
        s_0 = 2 # minimum gap
        T = 1.0 # time gap is different from dt

        # car following model: IDM
        # REALLY BAD!!!
        vf = s[n*3+0]
        xf = s[n*3+1]
        af = s[n*3+2]
        dx = s[(n-1)*3+1] - xf - l
        dv = s[(n-1)*3+0] - vf
        s_star = s_0 + max(0, vf*T + vf*dv/(2*((self.a_max*abs(self.a_min))**0.5)))
        a_temp = self.a_max*(1-(vf/self.v_max)**4 -(s_star/dx)**2) + randint(0,21)*0.0001 #### update by adding random number (0,0.02) 0606
        a_temp = max(a_temp,-vf/self.dt) # forget why

        # Update by adding constraint checking and rectifying
        ####### update from here
        # constraint for acceleration
        if a_temp > self.a_max: # or use rectify function
            a_temp = self.a_max
            if(self.verbose):#??????
                print("FC Violates Max Acceleration")
        elif a_temp < self.a_min:
            a_temp = self.a_min
            if(self.verbose):
                print("FC Violates Min Acceleration")

        v_temp =  vf + a_temp*self.dt
        # constraint for velocity
        if v_temp > self.v_max:
            v_temp = self.v_max
            a_temp = (v_temp-vf) / self.dt
            if(self.verbose):
                print("FC Violates Max Vel")
        elif v_temp < self.v_min:
            v_temp = self.v_min
            a_temp = (v_temp-vf) / self.dt
            if(self.verbose):
                print("FC Violates Min Vel")

        x_temp = (vf+v_temp)/2*self.dt + xf
        # contraint for safety
        ### can i add s_ to the function input?????
        xl_ = s_[(n-1)*3+1] # location of the leading vehicle in the pair at next time point
        min_s = 7.0 # safe gap
        gap = xl_ - x_temp - min_s - vf*self.dt # safe buffer coz traditional safety constraints donot work
        #since we can not use future information in reality and current value is meaningless if collision is happening
        if gap < 0:
            if(self.verbose):
                print("FC Violates safety")
            x_temp = xl_ - min_s - vf*self.dt
            a_temp = (x_temp - xf - vf*self.dt)*2/(self.dt**2)
            if a_temp<self.a_min or a_temp>self.a_max:
                if(self.verbose):
                    print("FC safety wrong acc")
                    #print(a_temp)
                    #input()
            v_temp = vf + a_temp*self.dt
            if v_temp<self.v_min or v_temp>self.v_max:
                if(self.verbose):
                    print("FC safety wrong vel")
                    #print(a_temp)
                    #input()
        ######## update end here
        # Add penalty for too large gap
        ######

        return v_temp, x_temp, a_temp

    def CACC(self,s,n): # traditional controller to choose the acceleration
        l = 6.85
        k = 0.3
        ka = 1.0
        kv = 3.0
        kd = 0.2
        v = s[(n)*3+0]
        x = s[(n)*3+1]
        a_v = k * (self.v_max - v)
        vp = s[(n-1)*3+0]
        xp = s[(n-1)*3+1]
        ap = s[(n-1)*3+2]
        r_ref = 2
        a_d = ka*ap + kv*(vp-v) + kd*(xp-x-l -r_ref)
        a_ = min(a_v,a_d)
        # update by adding rectify 06/13/19
        ####### update from here
        # constraint for acceleration
        if a_ > self.a_max: # or use rectify function
            a_ = self.a_max
            if(self.verbose):#??????
                print("CACC Violates Max Acceleration")
        elif a_ < self.a_min:
            a_ = self.a_min
            if(self.verbose):
                print("CACC Violates Min Acceleration")
        v_ =  v + a_*self.dt
        # constraint for velocity
        if v_ > self.v_max:
            v_ = self.v_max
            a_ = (v_-v) / self.dt
            if(self.verbose):
                print("CACC Violates Max Vel")
        elif v_ < self.v_min:
            v_ = self.v_min
            a_ = (v_-v) / self.dt
            if(self.verbose):
                print("CACC Violates Min Vel")

        x_ = (v+v_)/2*self.dt + x
        # contraint for safety
        ### can i add s_ to the function input????
        min_s = 2.0 # safe gap including vehicle length
        tau = int(1/self.dt)
        xp_ = data[self.t_start+self.t+1-tau][self.num_leading_cars-1][1]
        #xp_ = xp + vp*self.dt + 1/2*ap*(self.dt**2) # location of leading vehicle at next time point
        gap = xp_ - x_ - l - min_s - v*self.dt #safety constraint????????
        if gap < 0:
            if(self.verbose):
                print("#")
                print(gap)
                print("CACC Violates safety")
            x_ = xp_ - l - min_s - v*self.dt
            a_ = (x_ - x - v*self.dt)*2/(self.dt**2)
            if a_<self.a_min or a_>self.a_max:
                if(self.verbose):
                    print("CACC safety wrong acc")
                    #quit()
            v_ = v + a_*self.dt
            if v_<self.v_min or v_>self.v_max:
                if(self.verbose):
                    print("CACC safety wrong vel")
                    #quit()
        ######## update end here
        return v_, x_, a_

    def reward_function(self, states, disps):
        size = len(states)
        sum_squared_acc = 0
        #print("$$$")
        accels = []
        for s in range(size):
            state = []
            for i in range(self.num_vehicles):
                tmp = states[s][:3]
                states[s] = states[s][3:]
                state.append(tmp)

            # Calculate SUM of SQUARED ACCELERATION
            for n in range(self.num_vehicles - self.num_leading_cars):
                sum_squared_acc = sum_squared_acc + (1**n)*(state[self.num_leading_cars + n][2] ** 2)
                #print(n)
                accels.append((n,state[self.num_leading_cars + n][2]))
                #if(state[self.num_leading_cars + n][2] > 10 or state[self.num_leading_cars + n][2] < -10):
                    #print(accels)
                    #print(self.t_start)
                    #print(len(accels)/(self.num_vehicles - self.num_leading_cars))
                    #input()

        # Calculate SUM of CAR DISPLACEMENT
        sum_car_disp = sum([(1**i)*disps[self.num_leading_cars:][i] for i in range(self.num_vehicles - self.num_leading_cars)])

        reward = sum_car_disp - self.LAMBDA * ((sum_squared_acc)/((self.num_vehicles - self.num_leading_cars)*size))
        #if(reward < -100):
            #print("$$$$")
            #print(sum_car_disp)
            #print(sum_squared_acc)
            #print(self.LAMBDA * ((sum_squared_acc)/((self.num_vehicles - self.num_leading_cars)*size)))
            #print(self.LAMBDA * ((self.num_vehicles - self.num_leading_cars)*size)/(sum_squared_acc + self.EPSILON) )
            #print(sum(self.neg_rewards))
            #print(reward)
            #print((reward + sum(self.neg_rewards)))
            #print((reward + sum(self.neg_rewards))/self.rew_normalize)
            #print("$$$$")
            #print(accels)
            #input()
            #quit()

        return (reward + sum(self.neg_rewards))/self.rew_normalize

    def reset(self):
        # Resets cumilators for this time frame
        #print("Environment Reset")
        self.t_start = randint(20,len(data)-self.max_t-1)
        #while (self.t_start > 4000 and self.t_start < 5000) or (self.t_start > 6000 and self.t_start < 7000) or (self.t_start > 8000 and self.t_start < 9000):
            #self.t_start = randint(20,len(data)-self.max_t-1)
        self.current_states = [self.get_state(self.t_start)]
        self.current_disps = [0 for i in range(self.num_vehicles)]
        self.t = 0
        self.neg_rewards = []
        self.is_crashed = False
        return self.center_state(self.current_states[-1])

    def reset_from_state(self,t_start):
        # Resets cumilators for this time frame
        #print("Environment Reset")
        self.t_start = t_start
        #while (self.t_start > 4000 and self.t_start < 5000) or (self.t_start > 6000 and self.t_start < 7000) or (self.t_start > 8000 and self.t_start < 9000):
            #self.t_start = randint(20,len(data)-self.max_t-1)
        self.current_states = [self.get_state(self.t_start)]
        self.current_disps = [0 for i in range(self.num_vehicles)]
        self.t = 0
        self.neg_rewards = []
        self.is_crashed = False
        return self.center_state(self.current_states[-1])

    def render(self):
        # Prints Visulizations for us based on state
        self.win = pygame.display.set_mode((self.win_width,self.win_height))
        pygame.time.delay(10)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
        
        width = self.win_width * 0.02 / self.num_vehicles
        height = self.win_height * 0.05 / self.num_vehicles
        start = self.current_states[0][3*(self.num_vehicles-1)+1]
        distance = self.max_t * 1
        scale = 0.25

        vel = 5

        self.win.fill((255,255,255))
        for n in range(self.num_leading_cars):
            pygame.draw.rect(self.win, (0, 0, 255), (((self.current_states[-1][3*n+1] - start)/distance) * self.win_width * scale + 50, self.win_height/2, width, height))
        pygame.draw.rect(self.win, (0, 255, 0), (((self.current_states[-1][3*(self.num_leading_cars)+1] - start)/distance) * self.win_width * scale + 50, self.win_height/2, width, height))  
        for n in range(self.num_following_cars):
            pygame.draw.rect(self.win, (255, 0, 0), (((self.current_states[-1][3*(self.num_leading_cars+n+1)+1] - start)/distance) * self.win_width * scale + 50, self.win_height/2, width, height))

        pygame.display.update()
        os.system("clear")
        print("State")
        print()
        for n in range(self.num_vehicles):
            print("Car",n)
            color = None
            if (n < self.num_leading_cars):
                color = "Blue"
            elif (n == self.num_leading_cars):
                color = "Green"
            else:
                color = "Red"
            print("Color",color)
            print("Disp:",self.center_state(self.current_states[-1])[3*n+1])
            print("Vel:",self.current_states[-1][3*n+0])
            print("Acc:",self.current_states[-1][3*n+2])
            print()
        #print(self.center_state(self.current_states[-1]))

    def step(self, a, human=False):
        # returns next_state, reward, is_done, and info base on chosen acceleration
        s, a, r, s_, a_ = self.update_state(self.current_states[-1], a, human)
        
        self.current_states.append(s_)
        self.neg_rewards.append(r)

        # Calc displacement
        for n in range(self.num_vehicles):
            self.current_disps[n] = self.current_disps[n] + (s_[n*3+1] - s[n*3+1])

        self.t = self.t + 1
        if self.t < self.max_t:
            if self.is_crashed:
                reward = self.reward_function(self.current_states, self.current_disps)
                return self.center_state(s_), reward, 1, {"CAV":0}
            else:
                return self.center_state(s_), 0, 0, {"CAV":0}
        else:
            reward = self.reward_function(self.current_states, self.current_disps)
            return self.center_state(s_), reward, 1, {"CAV":reward}